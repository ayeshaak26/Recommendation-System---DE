"""Extract stage tests. A fake session stands in for TMDB, so no network or key is needed."""

from __future__ import annotations

import json
from datetime import date

import pytest
import requests

from pipeline import config, extract

TODAY = date(2026, 11, 20)
BASE = config.TMDB_BASE_URL


class FakeResponse:
    def __init__(self, status: int = 200, payload: dict | None = None, headers: dict | None = None):
        self.status_code = status
        self._payload = payload or {}
        self.headers = headers or {}

    def json(self) -> dict:
        return self._payload


class FakeTMDB:
    """Serves list pages of 2 movies each and a detail response per movie."""

    def __init__(self, list_ids: list[int], changed: list[int] | None = None):
        self.list_ids = list_ids
        self.changed = changed or []
        self.calls: list[str] = []
        self.scripted: dict[str, list] = {}  # path -> responses/exceptions served first

    def get(self, url: str, params: dict, timeout: int) -> FakeResponse:
        path = url.removeprefix(BASE)
        self.calls.append(path)
        if self.scripted.get(path):
            item = self.scripted[path].pop(0)
            if isinstance(item, Exception):
                raise item
            return item
        if path == "/movie/changes":
            return FakeResponse(payload={"results": [{"id": i} for i in self.changed], "total_pages": 1})
        if path in ("/movie/now_playing", "/movie/upcoming", "/discover/movie", "/movie/popular"):
            page = params["page"]
            chunk = self.list_ids[(page - 1) * 2 : page * 2]
            total = max(1, -(-len(self.list_ids) // 2))
            return FakeResponse(payload={"results": [{"id": i} for i in chunk], "total_pages": total})
        movie_id = int(path.rsplit("/", 1)[1])
        return FakeResponse(payload={"id": movie_id, "title": f"Movie {movie_id}"})

    def detail_calls(self) -> list[str]:
        lists = ("/movie/now_playing", "/movie/upcoming", "/movie/popular", "/movie/changes")
        return [c for c in self.calls if c.startswith("/movie/") and c not in lists]


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    """Point the raw layer and database at a temp folder for every test."""
    monkeypatch.setattr(config, "RAW_DIR", tmp_path / "raw")
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "curated" / "movies.db")


def make_client(fake: FakeTMDB) -> tuple[extract.TMDBClient, list[float]]:
    sleeps: list[float] = []
    return extract.TMDBClient("test-key", session=fake, sleep=sleeps.append), sleeps


# ---------------------------- client behaviour ----------------------------- #
def test_429_is_retried_and_honours_retry_after():
    fake = FakeTMDB([])
    fake.scripted["/movie/1"] = [FakeResponse(429, headers={"Retry-After": "3"})]
    client, sleeps = make_client(fake)

    assert client.get("/movie/1")["id"] == 1
    assert fake.calls == ["/movie/1", "/movie/1"]
    assert 3.0 in sleeps


def test_server_and_network_errors_are_retried_then_raised_without_the_key():
    fake = FakeTMDB([])
    leaky = requests.ConnectionError(f"{BASE}/movie/1?api_key=test-key")
    fake.scripted["/movie/1"] = [leaky] + [FakeResponse(503)] * (config.MAX_RETRIES - 1)
    client, _ = make_client(fake)

    with pytest.raises(extract.RetryableError) as err:
        client.get("/movie/1")
    assert len(fake.calls) == config.MAX_RETRIES
    assert "test-key" not in str(err.value)


def test_404_and_401_are_not_retried():
    fake = FakeTMDB([])
    fake.scripted["/movie/1"] = [FakeResponse(404)]
    fake.scripted["/movie/2"] = [FakeResponse(401)]
    client, _ = make_client(fake)

    with pytest.raises(extract.NotFoundError):
        client.get("/movie/1")
    with pytest.raises(extract.AuthError):
        client.get("/movie/2")
    assert fake.calls == ["/movie/1", "/movie/2"]


# ---------------------------- incremental rules ---------------------------- #
def test_plan_fetch_splits_new_changed_and_stale():
    known = {1: date(2026, 11, 19), 2: date(2026, 11, 19), 3: date(2026, 9, 1)}
    plan = extract.plan_fetch([1, 2, 3, 4], known, changed_ids={2}, today=TODAY)

    assert (plan.new, plan.changed, plan.stale) == ([4], [2], [3])


def test_plan_fetch_full_refresh_takes_everything():
    known = {1: TODAY, 2: TODAY}
    plan = extract.plan_fetch([1, 2, 3], known, changed_ids=set(), today=TODAY, full_refresh=True)

    assert sorted(plan.all_ids) == [1, 2, 3]


# ------------------------------ whole stage -------------------------------- #
def test_first_run_saves_raw_files_and_dedupes_ids():
    fake = FakeTMDB([10, 11, 11, 12, 13])  # 11 repeats across pages
    client, _ = make_client(fake)

    result = extract.run_extract(target=4, client=client, today=TODAY)

    movies_dir = config.RAW_DIR / "2026-11-20" / "movies"
    assert sorted(p.stem for p in movies_dir.glob("*.json")) == ["10", "11", "12", "13"]
    assert json.loads((movies_dir / "10.json").read_text("utf-8"))["title"] == "Movie 10"
    assert (result.discovered, result.fetched, result.failed) == (4, 4, 0)
    assert (config.RAW_DIR / "2026-11-20" / "_manifest.json").exists()
    assert list((config.RAW_DIR / "2026-11-20" / "lists").glob("*_page_001.json"))


def test_same_day_rerun_fetches_nothing_new():
    fake = FakeTMDB([10, 11, 12, 13])
    client, _ = make_client(fake)
    extract.run_extract(target=4, client=client, today=TODAY)
    fake.calls.clear()

    result = extract.run_extract(target=4, client=client, today=TODAY)

    assert fake.detail_calls() == []
    assert result.fetched == 0


def test_next_day_run_fetches_only_new_and_changed():
    fake = FakeTMDB([10, 11, 12, 13])
    client, _ = make_client(fake)
    extract.run_extract(target=4, client=client, today=date(2026, 11, 19))

    fake.list_ids = [10, 11, 12, 13, 14]  # 14 is a new release
    fake.changed = [11, 999]              # 11 was edited; 999 is not ours
    fake.calls.clear()
    result = extract.run_extract(target=5, client=client, today=TODAY)

    assert sorted(fake.detail_calls()) == ["/movie/11", "/movie/14"]
    assert (result.planned_new, result.planned_changed, result.fetched) == (1, 1, 2)


def test_deleted_and_failing_movies_do_not_crash_the_run():
    fake = FakeTMDB([10, 11, 12, 13])
    fake.scripted["/movie/11"] = [FakeResponse(404)]
    fake.scripted["/movie/12"] = [FakeResponse(500)] * config.MAX_RETRIES
    client, _ = make_client(fake)

    result = extract.run_extract(target=4, client=client, today=TODAY)

    assert (result.fetched, result.not_found, result.failed) == (2, 1, 1)
    assert result.failed_ids == [12]


def test_bad_api_key_aborts_the_run():
    fake = FakeTMDB([10, 11])
    fake.scripted["/movie/now_playing"] = [FakeResponse(401)]
    client, _ = make_client(fake)

    with pytest.raises(extract.AuthError):
        extract.run_extract(target=2, client=client, today=TODAY)
