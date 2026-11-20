"""LOAD stage: upsert the staged Parquet tables into the curated SQLite database,
and record every pipeline run in the pipeline_runs table.

Idempotent by design
--------------------
- movies     : INSERT ... ON CONFLICT(movie_id) DO UPDATE. Loading the same data twice
               changes nothing except updated_at; first_loaded_at is never overwritten,
               so "when did we first see this movie" stays true forever.
- child rows : for each movie in the batch, old genres / cast_crew / trailers rows are
               deleted and the current ones inserted. A recast or a replaced trailer
               therefore never leaves stale rows behind.
- Everything for one load happens in a single transaction: it all lands or none of it does.

pipeline_runs
-------------
`track_run()` wraps a pipeline run. It writes a "running" row at the start, then
"success" with the row counts, or "failed" with the error message if anything raises.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

from pipeline import config
from pipeline.logging_setup import get_logger

log = get_logger(__name__)

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
SCHEMA = """
CREATE TABLE IF NOT EXISTS movies (
    movie_id          INTEGER PRIMARY KEY,
    title             TEXT    NOT NULL,
    original_title    TEXT,
    overview          TEXT    NOT NULL,
    tagline           TEXT,
    release_date      TEXT,                 -- ISO date, YYYY-MM-DD
    release_year      INTEGER,
    vote_average      REAL    NOT NULL CHECK (vote_average BETWEEN 0 AND 10),
    vote_count        INTEGER,
    popularity        REAL,
    runtime           INTEGER,
    original_language TEXT,
    status            TEXT,
    imdb_id           TEXT,
    poster_path       TEXT,
    poster_url        TEXT,
    trailer_key       TEXT,
    trailer_url       TEXT,
    genres            TEXT,                 -- JSON array of names
    genres_text       TEXT,                 -- "Action, Science Fiction" for display
    keywords          TEXT,                 -- JSON array
    top_cast          TEXT,                 -- JSON array, billing order
    director          TEXT,
    tags              TEXT,                 -- TF-IDF input
    embedding_text    TEXT,                 -- sentence-embedding input
    extracted_at      TEXT    NOT NULL,     -- run date the raw data was fetched
    first_loaded_at   TEXT    NOT NULL,
    updated_at        TEXT    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_movies_title ON movies (title);
CREATE INDEX IF NOT EXISTS idx_movies_first_loaded ON movies (first_loaded_at);

CREATE TABLE IF NOT EXISTS genres (
    movie_id   INTEGER NOT NULL REFERENCES movies (movie_id) ON DELETE CASCADE,
    genre_id   INTEGER NOT NULL,
    genre_name TEXT    NOT NULL,
    PRIMARY KEY (movie_id, genre_id)
);
CREATE INDEX IF NOT EXISTS idx_genres_name ON genres (genre_name);

CREATE TABLE IF NOT EXISTS cast_crew (
    movie_id   INTEGER NOT NULL REFERENCES movies (movie_id) ON DELETE CASCADE,
    person_id  INTEGER NOT NULL,
    name       TEXT    NOT NULL,
    role       TEXT    NOT NULL CHECK (role IN ('cast', 'director')),
    character  TEXT,
    cast_order INTEGER,
    PRIMARY KEY (movie_id, person_id, role)
);

CREATE TABLE IF NOT EXISTS trailers (
    movie_id     INTEGER PRIMARY KEY REFERENCES movies (movie_id) ON DELETE CASCADE,
    youtube_key  TEXT    NOT NULL,
    name         TEXT,
    type         TEXT,
    official     INTEGER,
    published_at TEXT,
    url          TEXT    NOT NULL
);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id           INTEGER PRIMARY KEY AUTOINCREMENT,
    run_date         TEXT    NOT NULL,
    stage            TEXT    NOT NULL,      -- all | extract | transform | load
    full_refresh     INTEGER NOT NULL DEFAULT 0,
    started_at       TEXT    NOT NULL,
    finished_at      TEXT,
    duration_seconds REAL,
    status           TEXT    NOT NULL CHECK (status IN ('running', 'success', 'failed')),
    rows_extracted   INTEGER NOT NULL DEFAULT 0,
    rows_rejected    INTEGER NOT NULL DEFAULT 0,
    rows_loaded      INTEGER NOT NULL DEFAULT 0,
    rows_inserted    INTEGER NOT NULL DEFAULT 0,   -- movies new to the database
    rows_updated     INTEGER NOT NULL DEFAULT 0,
    model_refreshed  INTEGER NOT NULL DEFAULT 0,
    error_message    TEXT
);
"""

# staged column -> database column, in insert order
MOVIE_COLUMN_MAP = {
    "movie_id": "movie_id", "title": "title", "original_title": "original_title",
    "overview": "overview", "tagline": "tagline", "release_date": "release_date",
    "release_year": "release_year", "vote_average": "vote_average", "vote_count": "vote_count",
    "popularity": "popularity", "runtime": "runtime", "original_language": "original_language",
    "status": "status", "imdb_id": "imdb_id", "poster_path": "poster_path",
    "poster_url": "poster_url", "trailer_key": "trailer_key", "trailer_url": "trailer_url",
    "genres": "genres", "genres_text": "genres_text", "keywords": "keywords",
    "cast": "top_cast", "director": "director", "tags": "tags",
    "embedding_text": "embedding_text", "extracted_at": "extracted_at",
}
CHILD_TABLES = {
    "genres": ["movie_id", "genre_id", "genre_name"],
    "cast_crew": ["movie_id", "person_id", "name", "role", "character", "cast_order"],
    "trailers": ["movie_id", "youtube_key", "name", "type", "official", "published_at", "url"],
}
RUN_METRICS = (
    "rows_extracted", "rows_rejected", "rows_loaded", "rows_inserted", "rows_updated", "model_refreshed",
)


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


@contextmanager
def connect(db_path: Path | None = None) -> Iterator[sqlite3.Connection]:
    """Open the curated database; commit on success, roll back on error, always close."""
    path = db_path or config.DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    """Create all tables and indexes if they do not exist yet. Safe on every run."""
    with connect() as conn:
        conn.executescript(SCHEMA)


# --------------------------------------------------------------------------- #
# DataFrame -> SQLite values
# --------------------------------------------------------------------------- #
def _to_sql_value(value: Any) -> Any:
    """Convert one pandas/numpy value into something sqlite3 accepts."""
    if isinstance(value, (list, tuple)) or hasattr(value, "tolist") and getattr(value, "ndim", 0) == 1:
        return json.dumps(list(value), ensure_ascii=False)   # list column -> JSON text
    if value is None or value is pd.NaT or (not isinstance(value, str) and pd.isna(value)):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if hasattr(value, "item"):                               # numpy scalar -> python scalar
        return value.item()
    return value


def _rows(df: pd.DataFrame, columns: list[str]) -> list[tuple]:
    return [
        tuple(_to_sql_value(v) for v in row)
        for row in df[columns].itertuples(index=False, name=None)
    ]


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #
@dataclass
class LoadResult:
    """Summary of one load; `rows_inserted > 0` is what triggers the model refresh."""

    run_date: str | None
    rows_loaded: int = 0
    rows_inserted: int = 0
    rows_updated: int = 0
    table_totals: dict[str, int] = field(default_factory=dict)   # row counts after the load


def latest_staged_date() -> str | None:
    """Most recent run-date folder in the staged layer, or None."""
    if not config.STAGED_DIR.exists():
        return None
    dates = [p.name for p in config.STAGED_DIR.iterdir() if (p / "movies.parquet").exists()]
    return max(dates) if dates else None


def upsert_movies(conn: sqlite3.Connection, movies: pd.DataFrame) -> tuple[int, int]:
    """Insert new movies and update existing ones. Returns (inserted, updated)."""
    existing = {row[0] for row in conn.execute("SELECT movie_id FROM movies")}
    batch_ids = {int(i) for i in movies["movie_id"]}
    inserted = len(batch_ids - existing)

    db_cols = list(MOVIE_COLUMN_MAP.values())
    now = _now()
    sql = (
        f"INSERT INTO movies ({', '.join(db_cols)}, first_loaded_at, updated_at) "
        f"VALUES ({', '.join('?' * len(db_cols))}, ?, ?) "
        "ON CONFLICT (movie_id) DO UPDATE SET "
        # first_loaded_at is deliberately absent: it keeps its original value.
        + ", ".join(f"{c} = excluded.{c}" for c in db_cols if c != "movie_id")
        + ", updated_at = excluded.updated_at"
    )
    rows = [row + (now, now) for row in _rows(movies, list(MOVIE_COLUMN_MAP))]
    conn.executemany(sql, rows)
    return inserted, len(batch_ids) - inserted


def replace_children(conn: sqlite3.Connection, table: str, df: pd.DataFrame, movie_ids: list[int]) -> None:
    """Swap the child rows of every movie in this batch for the current ones."""
    conn.executemany(f"DELETE FROM {table} WHERE movie_id = ?", [(i,) for i in movie_ids])
    columns = CHILD_TABLES[table]
    conn.executemany(
        f"INSERT INTO {table} ({', '.join(columns)}) VALUES ({', '.join('?' * len(columns))})",
        _rows(df, columns),
    )


def run_load(run_date: str | None = None) -> LoadResult:
    """Upsert one run date's staged tables into the curated database.

    Args:
        run_date: "YYYY-MM-DD" staged folder to load. Defaults to the latest.
    """
    init_db()
    run_date = run_date or latest_staged_date()
    result = LoadResult(run_date=run_date)

    staged_dir = config.STAGED_DIR / run_date if run_date else None
    if staged_dir is None or not (staged_dir / "movies.parquet").exists():
        log.info("Load: no staged data found - nothing to do.")
    else:
        movies = pd.read_parquet(staged_dir / "movies.parquet")
        log.info("Load started for %s: %d staged movies", run_date, len(movies))
        if not movies.empty:
            movie_ids = [int(i) for i in movies["movie_id"]]
            with connect() as conn:  # one transaction for all four tables
                result.rows_inserted, result.rows_updated = upsert_movies(conn, movies)
                for table in CHILD_TABLES:
                    replace_children(conn, table, pd.read_parquet(staged_dir / f"{table}.parquet"), movie_ids)
            result.rows_loaded = len(movie_ids)

    with connect() as conn:
        for table in ("movies", *CHILD_TABLES):
            result.table_totals[table] = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
    log.info(
        "Load finished: %d new, %d updated | database now holds %d movies",
        result.rows_inserted, result.rows_updated, result.table_totals["movies"],
    )
    return result


# --------------------------------------------------------------------------- #
# pipeline_runs
# --------------------------------------------------------------------------- #
@contextmanager
def track_run(stage: str = "all", full_refresh: bool = False) -> Iterator[dict[str, Any]]:
    """Record a pipeline run in pipeline_runs.

    Usage:
        with track_run("all") as metrics:
            metrics["rows_extracted"] = run_extract().rows_extracted
            ...

    The yielded dict accepts the keys in RUN_METRICS. If the block raises, the run is
    marked "failed" with the error message and the exception continues upward, so
    Task Scheduler still sees a non-zero exit code.
    """
    init_db()
    started = datetime.now()
    with connect() as conn:
        run_id = conn.execute(
            "INSERT INTO pipeline_runs (run_date, stage, full_refresh, started_at, status) "
            "VALUES (?, ?, ?, ?, 'running')",
            (started.date().isoformat(), stage, int(full_refresh), started.isoformat(timespec="seconds")),
        ).lastrowid

    metrics: dict[str, Any] = {}
    status, error = "success", None
    try:
        yield metrics
    except BaseException as exc:  # includes Ctrl+C, so an interrupted run is not left "running"
        status, error = "failed", f"{type(exc).__name__}: {exc}"[:2000]
        raise
    finally:
        finished = datetime.now()
        values = {name: int(metrics.get(name) or 0) for name in RUN_METRICS}
        with connect() as conn:
            conn.execute(
                "UPDATE pipeline_runs SET finished_at = ?, duration_seconds = ?, status = ?, "
                "error_message = ?, " + ", ".join(f"{name} = ?" for name in RUN_METRICS)
                + " WHERE run_id = ?",
                (finished.isoformat(timespec="seconds"), round((finished - started).total_seconds(), 2),
                 status, error, *values.values(), run_id),
            )
        log.info("Run %d recorded as %s %s", run_id, status, values)


if __name__ == "__main__":
    # Stand-alone entry point: python -m pipeline.load [--run-date 2026-11-20]
    parser = argparse.ArgumentParser(description="Load staged data into the curated database.")
    parser.add_argument("--run-date", default=None, help="staged folder to load (default: latest)")
    print(json.dumps(asdict(run_load(parser.parse_args().run_date)), indent=2))
