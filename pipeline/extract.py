"""EXTRACT stage: pull movies from the TMDB API into the raw layer.

What one run does
-----------------
1. Discover movie IDs by paging through the list endpoints (fresh releases,
   top-voted, popular) until TARGET_MOVIE_COUNT IDs are collected.
2. Decide what actually needs fetching (the incremental part):
     - new      : discovered but not in the database yet
     - changed  : already known, and TMDB's /movie/changes says it was edited
                  since our last run
     - stale    : already known, but not refreshed for REFRESH_STALE_AFTER_DAYS
   With --full-refresh, everything is fetched again.
3. Fetch full details (credits, keywords, videos in ONE call per movie) and save
   each response untouched to data/raw/<run-date>/movies/<id>.json.

Raw layer layout
----------------
data/raw/2026-11-20/
    lists/movie_popular_page_001.json    <- every list page, as returned
    movies/27205.json                    <- one detail response per movie
    _manifest.json                       <- counts for this run

Safe to re-run: a second run on the same day skips movies already saved today,
so a crashed run simply resumes where it stopped.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Iterator

import requests
from tenacity import RetryCallState, Retrying, retry_if_exception_type, stop_after_attempt
from tenacity.wait import wait_exponential_jitter

from pipeline import config
from pipeline.logging_setup import get_logger

log = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class TMDBError(Exception):
    """Any TMDB API failure. Messages never contain the API key."""


class RetryableError(TMDBError):
    """Temporary failure (429, 5xx, network) - worth trying again."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


class NotFoundError(TMDBError):
    """404 - the movie was deleted from TMDB. Skip it, do not retry."""


class AuthError(TMDBError):
    """401 - bad or missing API key. Abort the whole run."""


# --------------------------------------------------------------------------- #
# API client
# --------------------------------------------------------------------------- #
class RateLimiter:
    """Spaces calls evenly so we never exceed N requests per second (thread-safe)."""

    def __init__(self, per_second: float, sleep: Callable[[float], None] = time.sleep) -> None:
        self._interval = 1.0 / per_second
        self._sleep = sleep
        self._lock = threading.Lock()
        self._next_slot = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = max(0.0, self._next_slot - now)
            self._next_slot = max(now, self._next_slot) + self._interval
        if delay:
            self._sleep(delay)


def _wait_strategy(retry_state: RetryCallState) -> float:
    """Honour TMDB's Retry-After header on a 429; otherwise back off exponentially."""
    exc = retry_state.outcome.exception() if retry_state.outcome else None
    retry_after = getattr(exc, "retry_after", None)
    if retry_after is not None:
        return min(float(retry_after), config.BACKOFF_MAX_SECONDS)
    return wait_exponential_jitter(
        initial=config.BACKOFF_MIN_SECONDS, max=config.BACKOFF_MAX_SECONDS
    )(retry_state)


class TMDBClient:
    """Thin TMDB v3 client with rate limiting and retries.

    `session` and `sleep` are injectable so tests can run without the network
    and without real waiting.
    """

    def __init__(
        self,
        api_key: str,
        session: requests.Session | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self._api_key = api_key
        self._session = session or requests.Session()
        self._limiter = RateLimiter(config.REQUESTS_PER_SECOND, sleep=sleep)
        self._retrying = Retrying(
            retry=retry_if_exception_type(RetryableError),
            stop=stop_after_attempt(config.MAX_RETRIES),
            wait=_wait_strategy,
            sleep=sleep,
            before_sleep=self._log_retry,
            reraise=True,  # after the last attempt, raise the real error
        )

    @staticmethod
    def _log_retry(retry_state: RetryCallState) -> None:
        log.warning(
            "Retry %d/%d in %.1fs: %s",
            retry_state.attempt_number,
            config.MAX_RETRIES,
            retry_state.next_action.sleep if retry_state.next_action else 0,
            retry_state.outcome.exception() if retry_state.outcome else "",
        )

    def get(self, path: str, **params: Any) -> dict[str, Any]:
        """GET a TMDB path (e.g. "/movie/popular") and return the parsed JSON."""
        return self._retrying(self._get_once, path, params)

    def _get_once(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        self._limiter.wait()
        try:
            response = self._session.get(
                f"{config.TMDB_BASE_URL}{path}",
                params={"api_key": self._api_key, **params},
                timeout=config.REQUEST_TIMEOUT_SECONDS,
            )
        except (requests.ConnectionError, requests.Timeout) as exc:
            # `from None`: the original message contains the full URL including the key.
            raise RetryableError(f"{type(exc).__name__} on {path}") from None

        status = response.status_code
        if status == 200:
            return response.json()
        if status == 429:
            header = response.headers.get("Retry-After", "")
            retry_after = float(header) if header.replace(".", "", 1).isdigit() else None
            raise RetryableError(f"429 rate limited on {path}", retry_after=retry_after)
        if status >= 500:
            raise RetryableError(f"{status} server error on {path}")
        if status == 401:
            raise AuthError("401 from TMDB - check TMDB_API_KEY in your .env file.")
        if status == 404:
            raise NotFoundError(f"404 not found: {path}")
        raise TMDBError(f"{status} unexpected response on {path}")


# --------------------------------------------------------------------------- #
# Raw layer helpers
# --------------------------------------------------------------------------- #
def _write_json(path: Path, payload: Any) -> None:
    """Write JSON atomically: a crash mid-write never leaves a half file behind."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)


def _iter_pages(
    client: TMDBClient, endpoint: str, lists_dir: Path, max_pages: int, **params: Any
) -> Iterator[dict[str, Any]]:
    """Yield each page of a paginated endpoint, saving every page to the raw layer."""
    slug = endpoint.strip("/").replace("/", "_")
    max_pages = min(max_pages, config.TMDB_MAX_PAGE)
    page = 1
    while page <= max_pages:
        data = client.get(endpoint, page=page, **params)
        _write_json(lists_dir / f"{slug}_page_{page:03d}.json", data)
        yield data
        if page >= int(data.get("total_pages") or 1):
            break
        page += 1


# --------------------------------------------------------------------------- #
# Step 1: discover IDs
# --------------------------------------------------------------------------- #
def discover_movie_ids(client: TMDBClient, lists_dir: Path, target: int) -> list[int]:
    """Collect up to `target` unique movie IDs from the list endpoints.

    A dict keeps insertion order and de-duplicates: the same movie appears in
    several lists, and TMDB pagination can repeat a movie across pages.
    """
    found: dict[int, None] = {}

    def collect(endpoint: str, stop_at: int, max_pages: int, **params: Any) -> None:
        if len(found) >= stop_at:
            return
        before = len(found)
        for data in _iter_pages(client, endpoint, lists_dir, max_pages, **params):
            for row in data.get("results") or []:
                if isinstance(row.get("id"), int):
                    found.setdefault(row["id"])
            if len(found) >= stop_at:
                break
        log.info("%-20s +%d ids (total %d)", endpoint, len(found) - before, len(found))

    collect("/movie/now_playing", target, config.FRESH_LIST_MAX_PAGES)
    collect("/movie/upcoming", target, config.FRESH_LIST_MAX_PAGES)
    top_voted_quota = min(target, len(found) + int(target * config.DISCOVER_SHARE))
    collect("/discover/movie", top_voted_quota, config.TMDB_MAX_PAGE, **config.DISCOVER_PARAMS)
    collect("/movie/popular", target, config.TMDB_MAX_PAGE)

    return list(found)[:target]


# --------------------------------------------------------------------------- #
# Step 2: decide what to fetch (incremental logic)
# --------------------------------------------------------------------------- #
def load_known_movies() -> dict[int, date]:
    """Return {movie_id: date last extracted} for movies we already have.

    Source of truth is the curated database (movies.movie_id, movies.extracted_at).
    Before the first load has ever run there is no database, so fall back to
    scanning the raw layer.
    """
    if config.DB_PATH.exists():
        try:
            with sqlite3.connect(config.DB_PATH) as conn:
                rows = conn.execute("SELECT movie_id, extracted_at FROM movies").fetchall()
            return {int(mid): date.fromisoformat(str(ts)[:10]) for mid, ts in rows if ts}
        except sqlite3.OperationalError:
            log.info("Database has no movies table yet - using the raw layer as state.")

    known: dict[int, date] = {}
    if config.RAW_DIR.exists():
        for run_dir in sorted(p for p in config.RAW_DIR.iterdir() if p.is_dir()):
            try:
                run_date = date.fromisoformat(run_dir.name)
            except ValueError:
                continue  # not a run-date folder
            for file in (run_dir / "movies").glob("*.json"):
                if file.stem.isdigit():
                    known[int(file.stem)] = run_date  # later folders overwrite earlier
    return known


def get_changed_ids(client: TMDBClient, lists_dir: Path, since: date, today: date) -> set[int]:
    """IDs TMDB reports as edited between `since` and today (max 14 days back)."""
    start = max(since, today - timedelta(days=config.TMDB_CHANGES_MAX_DAYS))
    changed: set[int] = set()
    pages = _iter_pages(
        client,
        "/movie/changes",
        lists_dir,
        config.TMDB_MAX_PAGE,
        start_date=start.isoformat(),
        end_date=today.isoformat(),
    )
    for data in pages:
        changed.update(r["id"] for r in data.get("results") or [] if isinstance(r.get("id"), int))
    return changed


@dataclass
class FetchPlan:
    """Which movie IDs this run will fetch, and why."""

    new: list[int] = field(default_factory=list)
    changed: list[int] = field(default_factory=list)
    stale: list[int] = field(default_factory=list)

    @property
    def all_ids(self) -> list[int]:
        return self.new + self.changed + self.stale


def plan_fetch(
    discovered: list[int],
    known: dict[int, date],
    changed_ids: set[int],
    today: date,
    full_refresh: bool = False,
) -> FetchPlan:
    """Pure function (no I/O) that applies the incremental rules. Easy to test."""
    plan = FetchPlan(new=[m for m in discovered if m not in known])
    for movie_id, last_extracted in known.items():
        if full_refresh:
            plan.stale.append(movie_id)
        elif movie_id in changed_ids:
            plan.changed.append(movie_id)
        elif (today - last_extracted).days > config.REFRESH_STALE_AFTER_DAYS:
            plan.stale.append(movie_id)
    return plan


# --------------------------------------------------------------------------- #
# Step 3: fetch details
# --------------------------------------------------------------------------- #
def _fetch_one(client: TMDBClient, movie_id: int, movies_dir: Path, overwrite: bool) -> str:
    """Fetch and save one movie. Returns: fetched | skipped | not_found | failed."""
    target = movies_dir / f"{movie_id}.json"
    if target.exists() and not overwrite:
        return "skipped"  # already saved earlier today - resume support
    try:
        data = client.get(f"/movie/{movie_id}", append_to_response=config.DETAIL_APPEND)
    except NotFoundError:
        return "not_found"
    except AuthError:
        raise  # no point continuing with a bad key
    except TMDBError as exc:
        log.error("Movie %s failed after retries: %s", movie_id, exc)
        return "failed"
    _write_json(target, data)
    return "fetched"


@dataclass
class ExtractResult:
    """Summary of one extract run; the LOG stage writes this to pipeline_runs."""

    run_date: str
    raw_dir: str
    full_refresh: bool
    started_at: str
    finished_at: str = ""
    discovered: int = 0
    planned_new: int = 0
    planned_changed: int = 0
    planned_stale: int = 0
    fetched: int = 0
    skipped_existing: int = 0
    not_found: int = 0
    failed: int = 0
    failed_ids: list[int] = field(default_factory=list)

    @property
    def rows_extracted(self) -> int:
        return self.fetched


def run_extract(
    full_refresh: bool = False,
    target: int | None = None,
    client: TMDBClient | None = None,
    today: date | None = None,
) -> ExtractResult:
    """Run the whole extract stage and return a summary.

    Args:
        full_refresh: re-fetch every movie, ignoring the incremental rules.
        target: how many movie IDs to discover (default: config.TARGET_MOVIE_COUNT).
                Pass a small number for a quick smoke test.
        client, today: injectable for tests.
    """
    today = today or date.today()
    target = target or config.TARGET_MOVIE_COUNT
    client = client or TMDBClient(config.require_api_key())

    run_dir = config.RAW_DIR / today.isoformat()
    lists_dir, movies_dir = run_dir / "lists", run_dir / "movies"
    movies_dir.mkdir(parents=True, exist_ok=True)

    result = ExtractResult(
        run_date=today.isoformat(),
        raw_dir=str(run_dir),
        full_refresh=full_refresh,
        started_at=datetime.now().isoformat(timespec="seconds"),
    )
    log.info("Extract started (target=%d, full_refresh=%s) -> %s", target, full_refresh, run_dir)

    # 1. Discover
    discovered = discover_movie_ids(client, lists_dir, target)
    result.discovered = len(discovered)

    # 2. Plan
    known = load_known_movies()
    changed_ids: set[int] = set()
    if known and not full_refresh:
        changed_ids = get_changed_ids(client, lists_dir, since=max(known.values()), today=today)
    plan = plan_fetch(discovered, known, changed_ids, today, full_refresh)
    result.planned_new = len(plan.new)
    result.planned_changed = len(plan.changed)
    result.planned_stale = len(plan.stale)
    log.info(
        "Plan: %d known | fetch %d new, %d changed, %d stale",
        len(known), len(plan.new), len(plan.changed), len(plan.stale),
    )

    # 3. Fetch details in parallel; the shared rate limiter keeps us within limits.
    ids = plan.all_ids
    with ThreadPoolExecutor(max_workers=config.EXTRACT_WORKERS) as pool:
        futures = {
            pool.submit(_fetch_one, client, movie_id, movies_dir, full_refresh): movie_id
            for movie_id in ids
        }
        try:
            for done, future in enumerate(as_completed(futures), start=1):
                outcome = future.result()
                if outcome == "fetched":
                    result.fetched += 1
                elif outcome == "skipped":
                    result.skipped_existing += 1
                elif outcome == "not_found":
                    result.not_found += 1
                else:
                    result.failed += 1
                    result.failed_ids.append(futures[future])
                if done % 500 == 0:
                    log.info("Progress: %d/%d movies", done, len(ids))
        except AuthError:
            pool.shutdown(wait=False, cancel_futures=True)
            raise

    result.finished_at = datetime.now().isoformat(timespec="seconds")
    _write_json(run_dir / "_manifest.json", asdict(result))
    log.info(
        "Extract finished: %d fetched, %d already saved today, %d not found, %d failed",
        result.fetched, result.skipped_existing, result.not_found, result.failed,
    )
    return result


if __name__ == "__main__":
    # Stand-alone entry point for testing this stage: python -m pipeline.extract --target 40
    parser = argparse.ArgumentParser(description="Run the TMDB extract stage.")
    parser.add_argument("--full-refresh", action="store_true", help="re-fetch every movie")
    parser.add_argument("--target", type=int, default=None, help="number of movies to discover")
    args = parser.parse_args()
    summary = run_extract(full_refresh=args.full_refresh, target=args.target)
    print(json.dumps(asdict(summary), indent=2))
