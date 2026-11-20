"""Validate + transform tests, run on small hand-made raw records (no network)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from pipeline import config, transform
from pipeline.validate import validate_movies

RUN_DATE = "2026-11-20"


def raw_movie(movie_id: int = 1, **overrides) -> dict:
    """A complete, valid TMDB detail response; override fields to break it."""
    movie = {
        "id": movie_id,
        "title": "Inception",
        "original_title": "Inception",
        "overview": "A thief who steals secrets through dream-sharing   technology &amp; more.",
        "tagline": "Your mind is the scene of the crime.",
        "release_date": "2010-07-15",
        "vote_average": 8.4,
        "vote_count": 36000,
        "popularity": 90.5,
        "runtime": 148,
        "original_language": "en",
        "status": "Released",
        "adult": False,
        "imdb_id": "tt1375666",
        "poster_path": "/abc123.jpg",
        "genres": [{"id": 28, "name": "Action"}, {"id": 878, "name": "Science Fiction"}],
        "keywords": {"keywords": [{"id": 1, "name": "dream"}, {"id": 2, "name": "time travel"}]},
        "credits": {
            "cast": [
                {"id": 11, "name": "Second Actor", "character": "B", "order": 1},
                {"id": 10, "name": "Leonardo DiCaprio", "character": "Cobb", "order": 0},
            ],
            "crew": [
                {"id": 20, "name": "Christopher Nolan", "job": "Director"},
                {"id": 21, "name": "Some Producer", "job": "Producer"},
            ],
        },
        "videos": {"results": [{"site": "YouTube", "type": "Trailer", "key": "YoHD9XEInc0", "official": True, "name": "Trailer", "published_at": "2010-05-10"}]},
    }
    movie.update(overrides)
    return movie


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    for name in ("RAW_DIR", "STAGED_DIR", "QUARANTINE_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())


def write_raw(movies: list[dict], extra_files: dict[str, str] | None = None) -> None:
    folder = config.RAW_DIR / RUN_DATE / "movies"
    folder.mkdir(parents=True, exist_ok=True)
    for i, movie in enumerate(movies):
        (folder / f"{movie.get('id', 'x')}_{i}.json").write_text(json.dumps(movie), encoding="utf-8")
    for name, text in (extra_files or {}).items():
        (folder / name).write_text(text, encoding="utf-8")


def reasons_for(**overrides) -> str:
    """Validate one broken movie and return why it was rejected ('' if it passed)."""
    flat = transform.flatten([raw_movie(**overrides)], RUN_DATE)["movies"]
    rejected = validate_movies(flat).rejected
    return rejected["reasons"].iloc[0] if len(rejected) else ""


# ------------------------------- trailers ---------------------------------- #
def test_pick_trailer_prefers_official_youtube_trailer():
    videos = [
        {"site": "Vimeo", "type": "Trailer", "key": "vimeo", "official": True},
        {"site": "YouTube", "type": "Featurette", "key": "feat", "official": True},
        {"site": "YouTube", "type": "Trailer", "key": "fan", "official": False, "published_at": "2024"},
        {"site": "YouTube", "type": "Trailer", "key": "official", "official": True, "published_at": "2020"},
    ]
    assert transform.pick_trailer(videos)["key"] == "official"


def test_pick_trailer_falls_back_to_teaser_then_none():
    teaser = [{"site": "YouTube", "type": "Teaser", "key": "tease"}]
    assert transform.pick_trailer(teaser)["key"] == "tease"
    assert transform.pick_trailer([{"site": "YouTube", "type": "Clip", "key": "c"}]) is None
    assert transform.pick_trailer(None) is None


# ------------------------------ validation --------------------------------- #
def test_valid_movie_passes():
    assert reasons_for() == ""


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"title": None}, "missing title"),
        ({"title": "   "}, "missing title"),
        ({"vote_average": 11.5}, "rating outside 0-10"),
        ({"vote_average": -1}, "rating outside 0-10"),
        ({"vote_average": "n/a"}, "missing vote_average"),
        ({"overview": ""}, "missing overview"),
        ({"overview": "Too short."}, "overview too short"),
        ({"release_date": "15/07/2010"}, "unparseable release_date"),
        ({"adult": True}, "adult title"),
        ({"id": None}, "missing movie_id"),
    ],
)
def test_bad_rows_are_rejected_with_a_reason(overrides, expected):
    assert expected in reasons_for(**overrides)


def test_missing_release_date_is_allowed():
    assert reasons_for(release_date="") == ""


def test_row_failing_two_rules_lists_both_reasons():
    reasons = reasons_for(title=None, vote_average=99)
    assert "missing title" in reasons and "rating outside 0-10" in reasons


def test_duplicate_ids_keep_first_and_reject_the_rest():
    flat = transform.flatten([raw_movie(1), raw_movie(1, title="Copy"), raw_movie(2)], RUN_DATE)["movies"]
    result = validate_movies(flat)

    assert result.valid["movie_id"].tolist() == [1, 2]
    assert result.valid["title"].tolist()[0] == "Inception"
    assert result.rejected["reasons"].tolist() == ["duplicate movie_id"]


def test_broken_poster_and_trailer_are_repaired_not_rejected():
    videos = {"results": [{"site": "YouTube", "type": "Trailer", "key": "bad key!"}]}
    flat = transform.flatten([raw_movie(poster_path="not-a-path", videos=videos)], RUN_DATE)["movies"]
    result = validate_movies(flat)

    assert len(result.valid) == 1 and result.rejected.empty
    assert pd.isna(result.valid.loc[0, "poster_path"])
    assert pd.isna(result.valid.loc[0, "trailer_key"])
    assert result.warnings["malformed_poster_path"] == 1
    assert result.warnings["malformed_trailer_key"] == 1


# ------------------------------- transform --------------------------------- #
def test_staged_movie_has_clean_fields_and_tags():
    write_raw([raw_movie()])
    transform.run_transform(RUN_DATE)
    row = pd.read_parquet(config.STAGED_DIR / RUN_DATE / "movies.parquet").iloc[0]

    assert row["overview"] == "A thief who steals secrets through dream-sharing technology & more."
    assert row["release_year"] == 2010
    assert list(row["cast"]) == ["Leonardo DiCaprio", "Second Actor"]  # sorted by billing order
    assert row["director"] == "Christopher Nolan"
    assert row["poster_url"] == "https://image.tmdb.org/t/p/w500/abc123.jpg"
    assert row["trailer_url"] == "https://www.youtube.com/watch?v=YoHD9XEInc0"
    for token in ("sciencefiction", "timetravel", "leonardodicaprio", "christophernolan", "thief"):
        assert token in row["tags"].split()
    assert "Directed by Christopher Nolan." in row["embedding_text"]


def test_movie_with_no_credits_keywords_or_videos_still_transforms():
    write_raw([raw_movie(credits=None, keywords=None, videos=None, genres=[], poster_path=None)])
    result = transform.run_transform(RUN_DATE)
    row = pd.read_parquet(config.STAGED_DIR / RUN_DATE / "movies.parquet").iloc[0]

    assert result.rows_valid == 1
    assert pd.isna(row["director"]) and pd.isna(row["trailer_url"]) and pd.isna(row["poster_url"])
    assert "thief" in row["tags"]


def test_run_quarantines_bad_rows_and_child_tables_only_hold_valid_movies():
    write_raw(
        [raw_movie(1), raw_movie(2, title=None), raw_movie(3, vote_average=50)],
        extra_files={"broken.json": "{not json"},
    )
    result = transform.run_transform(RUN_DATE)
    staged = config.STAGED_DIR / RUN_DATE
    quarantine = pd.read_parquet(config.QUARANTINE_DIR / RUN_DATE / "rejected_movies.parquet")

    assert (result.rows_in, result.rows_valid, result.rows_rejected) == (4, 1, 3)
    assert set(quarantine["reasons"]) == {"missing title", "rating outside 0-10", "unreadable raw file"}
    for table in config.STAGED_TABLES:
        assert set(pd.read_parquet(staged / f"{table}.parquet")["movie_id"]) == {1}
    assert len(pd.read_parquet(staged / "cast_crew.parquet")) == 3  # 2 cast + 1 director


def test_rerun_is_idempotent_and_clears_stale_quarantine():
    write_raw([raw_movie(1), raw_movie(2, title=None)])
    first = transform.run_transform(RUN_DATE)
    # The bad file is fixed upstream, then the stage is run again.
    (config.RAW_DIR / RUN_DATE / "movies" / "2_1.json").write_text(json.dumps(raw_movie(2)), "utf-8")
    second = transform.run_transform(RUN_DATE)

    assert (first.rows_valid, second.rows_valid, second.rows_rejected) == (1, 2, 0)
    assert not (config.QUARANTINE_DIR / RUN_DATE / "rejected_movies.parquet").exists()
    assert len(pd.read_parquet(config.STAGED_DIR / RUN_DATE / "movies.parquet")) == 2


def test_missing_raw_layer_gives_a_clear_error():
    with pytest.raises(FileNotFoundError, match="Run the extract stage first"):
        transform.run_transform()
