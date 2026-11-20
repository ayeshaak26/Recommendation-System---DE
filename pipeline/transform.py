"""TRANSFORM stage: raw JSON -> validated, cleaned Parquet tables in the staged layer.

Flow for one run date
---------------------
    raw/<date>/movies/*.json
        -> flatten        one row per movie + child tables (genres, cast_crew, trailers)
        -> validate       bad rows go to quarantine/<date>/rejected_movies.parquet
        -> clean + enrich text cleanup, dates, release year, tags, poster/trailer URLs
        -> staged/<date>/{movies,genres,cast_crew,trailers}.parquet

Two text fields are built for the recommender:
    tags            keyword soup for TF-IDF. Names are squashed into single tokens
                    ("Christopher Nolan" -> "christophernolan") so the director is not
                    confused with every other "Christopher".
    embedding_text  the same facts as a natural sentence, which is what a sentence
                    embedding model is trained on.

Safe to re-run: the staged and quarantine files for the date are overwritten.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline import config
from pipeline.logging_setup import get_logger
from pipeline.validate import validate_movies

log = get_logger(__name__)

MOVIE_COLUMNS = [
    "movie_id", "title", "original_title", "overview", "tagline", "release_date",
    "vote_average", "vote_count", "popularity", "runtime", "original_language",
    "status", "adult", "imdb_id", "poster_path", "genres", "keywords", "cast",
    "director", "trailer_key", "extracted_at",
]
GENRE_COLUMNS = ["movie_id", "genre_id", "genre_name"]
PEOPLE_COLUMNS = ["movie_id", "person_id", "name", "role", "character", "cast_order"]
TRAILER_COLUMNS = ["movie_id", "youtube_key", "name", "type", "official", "published_at"]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def clean_text(value: Any) -> str | None:
    """Unescape HTML entities, collapse whitespace, trim. Empty -> None."""
    if not isinstance(value, str):
        return None
    text = re.sub(r"\s+", " ", html.unescape(value)).strip()
    return text or None


def _token(name: str) -> str:
    """'Christopher Nolan' -> 'christophernolan' (one token for TF-IDF)."""
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _names(items: Any) -> list[str]:
    """Pull cleaned, de-duplicated 'name' values out of a list of TMDB objects."""
    seen: dict[str, None] = {}
    for item in items if isinstance(items, list) else []:
        name = clean_text(item.get("name")) if isinstance(item, dict) else None
        if name:
            seen.setdefault(name)
    return list(seen)


def pick_trailer(videos: Any) -> dict[str, Any] | None:
    """Choose the best YouTube video: a Trailer (else a Teaser), official first, newest first."""
    youtube = [
        v for v in videos if isinstance(v, dict) and v.get("site") == "YouTube" and v.get("key")
    ] if isinstance(videos, list) else []
    for kind in config.TRAILER_TYPE_PREFERENCE:
        candidates = [v for v in youtube if v.get("type") == kind]
        if candidates:
            return max(candidates, key=lambda v: (bool(v.get("official")), v.get("published_at") or ""))
    return None


# --------------------------------------------------------------------------- #
# Read + flatten
# --------------------------------------------------------------------------- #
def load_raw_movies(run_dir: Path) -> tuple[list[dict[str, Any]], list[str]]:
    """Read every raw movie file. Returns (records, names of unreadable files)."""
    records: list[dict[str, Any]] = []
    unreadable: list[str] = []
    for file in sorted((run_dir / "movies").glob("*.json")):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise ValueError("not a JSON object")
            records.append(data)
        except (ValueError, OSError) as exc:
            log.warning("Unreadable raw file %s: %s", file.name, exc)
            unreadable.append(file.name)
    return records, unreadable


def flatten(records: list[dict[str, Any]], extracted_at: str) -> dict[str, pd.DataFrame]:
    """Turn nested TMDB responses into four flat tables. No rows are dropped here."""
    movies, genres, people, trailers = [], [], [], []

    for raw in records:
        movie_id = raw.get("id")
        credits = raw.get("credits") or {}
        cast = sorted(
            (c for c in credits.get("cast") or [] if isinstance(c, dict) and c.get("name")),
            key=lambda c: c.get("order") if isinstance(c.get("order"), int) else 9999,
        )[: config.TOP_CAST_COUNT]
        directors = [
            c for c in credits.get("crew") or []
            if isinstance(c, dict) and c.get("job") == "Director" and c.get("name")
        ]
        trailer = pick_trailer((raw.get("videos") or {}).get("results"))

        movies.append({
            "movie_id": movie_id,
            "title": raw.get("title"),
            "original_title": raw.get("original_title"),
            "overview": raw.get("overview"),
            "tagline": raw.get("tagline"),
            "release_date": raw.get("release_date") or None,  # TMDB sends "" when unknown
            "vote_average": raw.get("vote_average"),
            "vote_count": raw.get("vote_count"),
            "popularity": raw.get("popularity"),
            "runtime": raw.get("runtime"),
            "original_language": raw.get("original_language"),
            "status": raw.get("status"),
            "adult": raw.get("adult"),
            "imdb_id": raw.get("imdb_id") or None,
            "poster_path": raw.get("poster_path") or None,
            "genres": _names(raw.get("genres")),
            "keywords": _names((raw.get("keywords") or {}).get("keywords")),
            "cast": _names(cast),
            "director": ", ".join(_names(directors)) or None,
            "trailer_key": trailer.get("key") if trailer else None,
            "extracted_at": extracted_at,
        })
        for g in raw.get("genres") or []:
            if isinstance(g, dict) and g.get("name"):
                genres.append({"movie_id": movie_id, "genre_id": g.get("id"), "genre_name": clean_text(g["name"])})
        for c in cast:
            people.append({
                "movie_id": movie_id, "person_id": c.get("id"), "name": clean_text(c["name"]),
                "role": "cast", "character": clean_text(c.get("character")), "cast_order": c.get("order"),
            })
        for d in directors:
            people.append({
                "movie_id": movie_id, "person_id": d.get("id"), "name": clean_text(d["name"]),
                "role": "director", "character": None, "cast_order": None,
            })
        if trailer:
            trailers.append({
                "movie_id": movie_id, "youtube_key": trailer.get("key"),
                "name": clean_text(trailer.get("name")), "type": trailer.get("type"),
                "official": bool(trailer.get("official")), "published_at": trailer.get("published_at"),
            })

    movies_df = pd.DataFrame(movies, columns=MOVIE_COLUMNS)
    # Force types now, so anything unexpected becomes a null the validator can catch.
    movies_df["movie_id"] = pd.to_numeric(movies_df["movie_id"], errors="coerce").astype("Int64")
    for col in ("vote_average", "popularity"):
        movies_df[col] = pd.to_numeric(movies_df[col], errors="coerce").astype("float64")
    for col in ("vote_count", "runtime"):
        movies_df[col] = pd.to_numeric(movies_df[col], errors="coerce").round().astype("Int64")
    for col in ("title", "original_title", "overview", "tagline"):
        movies_df[col] = movies_df[col].map(clean_text)

    return {
        "movies": movies_df,
        "genres": pd.DataFrame(genres, columns=GENRE_COLUMNS),
        "cast_crew": pd.DataFrame(people, columns=PEOPLE_COLUMNS),
        "trailers": pd.DataFrame(trailers, columns=TRAILER_COLUMNS),
    }


# --------------------------------------------------------------------------- #
# Clean + enrich (runs on validated rows only)
# --------------------------------------------------------------------------- #
def build_tags(row: pd.Series) -> str:
    """Keyword soup for TF-IDF: genres, keywords, cast, director, then the overview."""
    names = [*row["genres"], *row["keywords"], *row["cast"]]
    if isinstance(row["director"], str):  # missing values arrive as NaN, not None
        names += row["director"].split(", ")
    tokens = [t for t in (_token(n) for n in names) if t]
    overview = re.sub(r"[^a-z0-9 ]", " ", row["overview"].lower())
    return re.sub(r"\s+", " ", " ".join(tokens) + " " + overview).strip()


def build_embedding_text(row: pd.Series) -> str:
    """The same facts as readable sentences, for the sentence-embedding model."""
    parts = [f"{row['title']}."]
    if row["genres"]:
        parts.append(f"Genres: {', '.join(row['genres'])}.")
    if row["keywords"]:
        parts.append(f"Keywords: {', '.join(row['keywords'])}.")
    if isinstance(row["director"], str):
        parts.append(f"Directed by {row['director']}.")
    if row["cast"]:
        parts.append(f"Starring {', '.join(row['cast'])}.")
    parts.append(row["overview"])
    return " ".join(parts)


def enrich_movies(valid: pd.DataFrame) -> pd.DataFrame:
    """Standardise dates and add the derived columns the model and app need."""
    df = valid.copy()
    df["release_date"] = pd.to_datetime(df["release_date"], format="%Y-%m-%d", errors="coerce")
    df["release_year"] = df["release_date"].dt.year.astype("Int64")
    df["adult"] = df["adult"].fillna(False).astype(bool)
    df["genres_text"] = df["genres"].map(", ".join)
    df["poster_url"] = df["poster_path"].map(config.poster_url, na_action="ignore")
    df["trailer_url"] = df["trailer_key"].map(
        lambda key: config.YOUTUBE_WATCH_URL.format(key=key), na_action="ignore"
    )
    if df.empty:
        df["tags"] = pd.Series(dtype="object")
        df["embedding_text"] = pd.Series(dtype="object")
    else:
        df["tags"] = df.apply(build_tags, axis=1)
        df["embedding_text"] = df.apply(build_embedding_text, axis=1)
    return df


def _clean_child_tables(tables: dict[str, pd.DataFrame], valid_ids: pd.Series) -> dict[str, pd.DataFrame]:
    """Keep child rows for valid movies only, and enforce each table's primary key."""
    keys = {
        "genres": ["movie_id", "genre_id"],
        "cast_crew": ["movie_id", "person_id", "role"],
        "trailers": ["movie_id"],
    }
    cleaned = {}
    for name, key in keys.items():
        df = tables[name]
        df = df[df["movie_id"].isin(valid_ids)].dropna(subset=key).drop_duplicates(subset=key)
        for col in ("movie_id", "genre_id", "person_id", "cast_order"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")
        cleaned[name] = df.reset_index(drop=True)
    cleaned["trailers"]["url"] = cleaned["trailers"]["youtube_key"].map(
        lambda key: config.YOUTUBE_WATCH_URL.format(key=key)
    )
    return cleaned


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
@dataclass
class TransformResult:
    """Summary of one validate + transform run; the LOG stage records these counts."""

    run_date: str
    staged_dir: str
    rows_in: int = 0
    rows_valid: int = 0
    rows_rejected: int = 0
    unreadable_files: int = 0
    rejection_reasons: dict[str, int] = field(default_factory=dict)
    warnings: dict[str, int] = field(default_factory=dict)
    table_rows: dict[str, int] = field(default_factory=dict)


def latest_raw_date() -> str | None:
    """Most recent run-date folder in the raw layer, or None if there is none."""
    if not config.RAW_DIR.exists():
        return None
    dates = [p.name for p in config.RAW_DIR.iterdir() if (p / "movies").is_dir()]
    return max(dates) if dates else None


def run_transform(run_date: str | None = None) -> TransformResult:
    """Validate and transform one run date's raw files into the staged layer.

    Args:
        run_date: "YYYY-MM-DD" folder to process. Defaults to the latest raw folder.
    """
    run_date = run_date or latest_raw_date()
    if run_date is None:
        raise FileNotFoundError(f"No raw data found in {config.RAW_DIR}. Run the extract stage first.")

    raw_dir = config.RAW_DIR / run_date
    staged_dir = config.STAGED_DIR / run_date
    quarantine_file = config.QUARANTINE_DIR / run_date / "rejected_movies.parquet"
    result = TransformResult(run_date=run_date, staged_dir=str(staged_dir))

    records, unreadable = load_raw_movies(raw_dir)
    result.rows_in = len(records) + len(unreadable)
    result.unreadable_files = len(unreadable)
    log.info("Transform started for %s: %d raw movies", run_date, result.rows_in)

    tables = flatten(records, extracted_at=run_date)
    checked = validate_movies(tables["movies"])
    movies = enrich_movies(checked.valid)
    children = _clean_child_tables(tables, movies["movie_id"])

    # Quarantine: rejected rows + a line for each unreadable file.
    rejected = checked.rejected
    if unreadable:
        broken = pd.DataFrame({"title": unreadable, "reasons": "unreadable raw file"})
        rejected = pd.concat([rejected, broken], ignore_index=True)
    quarantine_file.unlink(missing_ok=True)  # clear a previous same-day result
    if not rejected.empty:
        quarantine_file.parent.mkdir(parents=True, exist_ok=True)
        rejected.assign(run_date=run_date).to_parquet(quarantine_file, index=False)

    # Staged layer
    staged_dir.mkdir(parents=True, exist_ok=True)
    for name, df in {"movies": movies, **children}.items():
        df.to_parquet(staged_dir / f"{name}.parquet", index=False)
        result.table_rows[name] = len(df)

    result.rows_valid = len(movies)
    result.rows_rejected = len(rejected)
    result.rejection_reasons = checked.reason_counts
    if unreadable:
        result.rejection_reasons["unreadable raw file"] = len(unreadable)
    result.warnings = checked.warnings
    (staged_dir / "_manifest.json").write_text(json.dumps(asdict(result), indent=2), encoding="utf-8")

    log.info(
        "Transform finished: %d valid, %d rejected -> %s",
        result.rows_valid, result.rows_rejected, staged_dir,
    )
    return result


if __name__ == "__main__":
    # Stand-alone entry point: python -m pipeline.transform [--run-date 2026-11-20]
    parser = argparse.ArgumentParser(description="Validate and transform raw TMDB data.")
    parser.add_argument("--run-date", default=None, help="raw folder to process (default: latest)")
    print(json.dumps(asdict(run_transform(parser.parse_args().run_date)), indent=2))
