"""Load + pipeline_runs tests, using a temporary database built from fake raw records."""

from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import date

import pytest

from pipeline import config, extract, load, transform
from test_transform import raw_movie

DAY1, DAY2 = "2026-11-19", "2026-11-20"


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    for name in ("RAW_DIR", "STAGED_DIR", "QUARANTINE_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    monkeypatch.setattr(config, "DB_PATH", tmp_path / "curated" / "movies.db")


def stage(run_date: str, movies: list[dict]) -> None:
    """Write raw files for a date and run them through validate + transform."""
    folder = config.RAW_DIR / run_date / "movies"
    folder.mkdir(parents=True, exist_ok=True)
    for movie in movies:
        (folder / f"{movie['id']}.json").write_text(json.dumps(movie), encoding="utf-8")
    transform.run_transform(run_date)


def query(sql: str, *params) -> list[tuple]:
    with closing(sqlite3.connect(config.DB_PATH)) as conn:
        return conn.execute(sql, params).fetchall()


def table_counts() -> dict[str, int]:
    return {t: query(f"SELECT COUNT(*) FROM {t}")[0][0] for t in ("movies", "genres", "cast_crew", "trailers")}


# --------------------------------- load ------------------------------------ #
def test_first_load_fills_all_tables():
    stage(DAY1, [raw_movie(1), raw_movie(2, title="Tenet")])
    result = load.run_load(DAY1)

    assert (result.rows_loaded, result.rows_inserted, result.rows_updated) == (2, 2, 0)
    assert table_counts() == {"movies": 2, "genres": 4, "cast_crew": 6, "trailers": 2}
    title, year, cast, genres, trailer = query(
        "SELECT title, release_year, top_cast, genres_text, trailer_url FROM movies WHERE movie_id = 1"
    )[0]
    assert (title, year, genres) == ("Inception", 2010, "Action, Science Fiction")
    assert json.loads(cast) == ["Leonardo DiCaprio", "Second Actor"]
    assert trailer == "https://www.youtube.com/watch?v=YoHD9XEInc0"


def test_loading_the_same_data_twice_creates_no_duplicates():
    stage(DAY1, [raw_movie(1), raw_movie(2)])
    load.run_load(DAY1)
    before = table_counts()
    result = load.run_load(DAY1)

    assert table_counts() == before
    assert (result.rows_inserted, result.rows_updated) == (0, 2)


def test_changed_movie_is_updated_and_stale_child_rows_are_removed():
    stage(DAY1, [raw_movie(1)])
    load.run_load(DAY1)
    first_loaded = query("SELECT first_loaded_at FROM movies WHERE movie_id = 1")[0][0]

    recast = {"cast": [{"id": 99, "name": "New Lead", "character": "Cobb", "order": 0}], "crew": []}
    stage(DAY2, [raw_movie(1, title="Inception (Remastered)", credits=recast, videos=None, genres=[])])
    result = load.run_load(DAY2)

    assert (result.rows_inserted, result.rows_updated) == (0, 1)
    assert query("SELECT title, extracted_at, first_loaded_at FROM movies") == [
        ("Inception (Remastered)", DAY2, first_loaded)
    ]
    assert query("SELECT name, role FROM cast_crew") == [("New Lead", "cast")]
    assert table_counts() == {"movies": 1, "genres": 0, "cast_crew": 1, "trailers": 0}


def test_incremental_load_adds_only_the_new_movie():
    stage(DAY1, [raw_movie(1), raw_movie(2)])
    load.run_load(DAY1)
    stage(DAY2, [raw_movie(3, title="Dunkirk")])
    result = load.run_load()  # defaults to the latest staged date

    assert (result.run_date, result.rows_inserted, result.rows_updated) == (DAY2, 1, 0)
    assert result.table_totals["movies"] == 3


def test_load_with_no_staged_data_does_nothing():
    result = load.run_load()

    assert (result.rows_loaded, result.table_totals["movies"]) == (0, 0)


def test_failed_load_rolls_back_everything(monkeypatch):
    stage(DAY1, [raw_movie(1)])

    def boom(*args, **kwargs):
        raise RuntimeError("disk full")

    monkeypatch.setattr(load, "replace_children", boom)
    with pytest.raises(RuntimeError):
        load.run_load(DAY1)

    assert query("SELECT COUNT(*) FROM movies")[0][0] == 0  # the movie upsert was undone too


def test_database_rejects_out_of_range_rating_as_a_last_line_of_defence():
    load.init_db()
    with pytest.raises(sqlite3.IntegrityError), load.connect() as conn:
        conn.execute(
            "INSERT INTO movies (movie_id, title, overview, vote_average, extracted_at, first_loaded_at, updated_at) "
            "VALUES (1, 't', 'o', 42, 'd', 'd', 'd')"
        )


def test_extract_reads_known_movies_from_the_database_after_a_load():
    stage(DAY1, [raw_movie(1), raw_movie(2)])
    load.run_load(DAY1)

    assert extract.load_known_movies() == {1: date(2026, 11, 19), 2: date(2026, 11, 19)}


# ----------------------------- pipeline_runs ------------------------------- #
def test_successful_run_is_recorded_with_its_counts():
    with load.track_run("all", full_refresh=True) as metrics:
        metrics.update(rows_extracted=10, rows_rejected=2, rows_loaded=8, rows_inserted=8, model_refreshed=True)

    row = query(
        "SELECT stage, full_refresh, status, rows_extracted, rows_rejected, rows_loaded, rows_inserted, "
        "model_refreshed, error_message, finished_at IS NOT NULL, duration_seconds >= 0 FROM pipeline_runs"
    )
    assert row == [("all", 1, "success", 10, 2, 8, 8, 1, None, 1, 1)]


def test_failed_run_is_recorded_with_the_error_and_still_raises():
    with pytest.raises(ValueError), load.track_run("extract") as metrics:
        metrics["rows_extracted"] = 5
        raise ValueError("TMDB is down")

    assert query("SELECT status, rows_extracted, error_message FROM pipeline_runs") == [
        ("failed", 5, "ValueError: TMDB is down")
    ]


def test_each_run_gets_its_own_row():
    for _ in range(3):
        with load.track_run():
            pass

    assert [r[0] for r in query("SELECT run_id FROM pipeline_runs ORDER BY run_id")] == [1, 2, 3]
