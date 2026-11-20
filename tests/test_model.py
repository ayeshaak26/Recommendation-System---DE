"""Recommender tests. A fake encoder replaces the sentence-transformers model and a fake
client replaces TMDB, so these run in seconds with no PyTorch and no network."""

from __future__ import annotations

import json

import numpy as np
import pytest

from pipeline import config, load, recommender, refresh_model
from test_load import stage
from test_transform import raw_movie

DAY1, DAY2 = "2026-11-19", "2026-11-20"
SPACE = "Astronauts travel through deep space on a rocket to a distant galaxy and planet."
LOVE = "A tender romance about love, a wedding and two hearts in a small seaside town."


def movie(movie_id: int, title: str, overview: str, genre: str) -> dict:
    return raw_movie(
        movie_id, title=title, overview=overview, vote_count=1000 - movie_id,
        genres=[{"id": 1, "name": genre}], keywords={"keywords": []}, credits={"cast": [], "crew": []},
    )


CATALOGUE = [
    movie(1, "Star Voyage", SPACE, "Science Fiction"),
    movie(2, "Galaxy Run", SPACE + " The rocket crew must survive.", "Science Fiction"),
    movie(3, "Seaside Hearts", LOVE, "Romance"),
    movie(4, "The Wedding Town", LOVE + " A wedding changes everything.", "Romance"),
]


class FakeEncoder:
    """Maps a text to [space-ness, love-ness] and remembers what it was asked to encode."""

    def __init__(self):
        self.encoded: list[str] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.encoded += texts
        return np.array([[t.lower().count("space") + 0.01, t.lower().count("love") + 0.01] for t in texts])


class FakeClient:
    """TMDB says: movie 1 <-> movie 2, movie 3 <-> movie 4 (plus one movie we do not hold)."""

    recs = {1: [2, 999], 2: [1], 3: [4], 4: [3]}

    def __init__(self):
        self.calls = 0

    def get(self, path: str, **params) -> dict:
        self.calls += 1
        movie_id = int(path.split("/")[2])
        return {"results": [{"id": i} for i in self.recs.get(movie_id, [])]}


@pytest.fixture(autouse=True)
def isolated_paths(tmp_path, monkeypatch):
    for name in ("RAW_DIR", "STAGED_DIR", "QUARANTINE_DIR"):
        monkeypatch.setattr(config, name, tmp_path / name.lower())
    curated = tmp_path / "curated"
    monkeypatch.setattr(config, "DB_PATH", curated / "movies.db")
    monkeypatch.setattr(config, "EMBEDDINGS_ARTIFACT", curated / "embeddings.joblib")
    monkeypatch.setattr(config, "NEIGHBOURS_ARTIFACT", curated / "neighbours.joblib")
    monkeypatch.setattr(config, "EVAL_REPORT", curated / "model_comparison.json")
    monkeypatch.setattr(config, "RECS_CACHE_DIR", tmp_path / "recs")
    monkeypatch.setattr(config, "TFIDF_PARAMS", {"stop_words": "english"})  # tiny corpus: no min_df
    monkeypatch.setattr(config, "EVAL_SPOT_CHECK_TITLES", ("Star Voyage", "Not In Catalogue"))


def build(day: str = DAY1, movies: list[dict] = CATALOGUE, **kwargs):
    stage(day, movies)
    load.run_load(day)
    kwargs.setdefault("encoder", FakeEncoder())
    kwargs.setdefault("client", FakeClient())
    return refresh_model.run_refresh(**kwargs)


# ------------------------------ similarity --------------------------------- #
def test_top_k_neighbours_orders_by_similarity_and_excludes_self():
    matrix = np.array([[1, 0], [0.9, 0.1], [0, 1], [0.1, 0.9]], dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)

    indices, scores = refresh_model.top_k_neighbours(matrix, k=2, chunk_size=3)  # forces 2 chunks

    assert indices[:, 0].tolist() == [1, 0, 3, 2]
    assert all(i not in row for i, row in enumerate(indices.tolist()))
    assert (scores[:, 0] >= scores[:, 1]).all()


def test_top_k_neighbours_handles_a_single_movie():
    indices, scores = refresh_model.top_k_neighbours(np.ones((1, 3), dtype=np.float32), k=10)

    assert indices.shape == scores.shape == (1, 0)


def test_overlap_metrics():
    assert refresh_model.overlap_metrics([1, 2, 3, 4], {2, 4, 9}) == (0.5, 2 / 3, 1.0)
    assert refresh_model.overlap_metrics([1, 2], {7}) == (0.0, 0.0, 0.0)


# ------------------------------ full refresh ------------------------------- #
def test_both_methods_pair_up_similar_movies():
    build()
    artifact = recommender.load_neighbours()

    for method in recommender.METHODS:
        best = {m: recommender.recommend(artifact, m, method, n=1)[0][0] for m in (1, 2, 3, 4)}
        assert best == {1: 2, 2: 1, 3: 4, 4: 3}, method


def test_report_contains_metrics_winner_and_spot_checks():
    result = build()
    report = json.loads(config.EVAL_REPORT.read_text("utf-8"))

    assert result.refreshed and result.winner in recommender.METHODS
    assert report["evaluated_movies"] == 4
    assert set(report["metrics"]) == {"tfidf", "embeddings"}
    assert report["metrics"]["tfidf"]["hit_rate"] == 1.0
    assert report["spot_checks"]["Star Voyage"]["tfidf"][0] == "Galaxy Run"
    assert report["spot_checks"]["Star Voyage"]["tmdb"] == ["Galaxy Run"]  # 999 is not in our catalogue
    assert "Not In Catalogue" not in report["spot_checks"]
    assert recommender.load_neighbours()["default_method"] == report["winner"]


def test_refresh_is_skipped_when_no_movies_were_added():
    build()
    encoder = FakeEncoder()
    result = refresh_model.run_refresh(encoder=encoder, client=FakeClient())

    assert (result.refreshed, result.reason) == (False, "no new movies since the last build")
    assert encoder.encoded == []


def test_new_movie_triggers_refresh_and_only_it_is_encoded():
    build()
    encoder, client = FakeEncoder(), FakeClient()
    result = build(DAY2, [movie(5, "Moon Base", SPACE + " A base on the moon.", "Science Fiction")],
                   encoder=encoder, client=client)

    assert result.refreshed and result.n_movies == 5 and result.n_encoded == 1
    assert len(encoder.encoded) == 1 and encoder.encoded[0].startswith("Moon Base.")
    assert len(recommender.load_neighbours()["movie_ids"]) == 5
    assert client.calls == 1  # recommendations for movies 1-4 came from the cache


def test_force_rebuilds_but_reuses_unchanged_embeddings():
    build()
    encoder = FakeEncoder()
    result = refresh_model.run_refresh(force=True, encoder=encoder, client=FakeClient())

    assert result.refreshed and result.n_encoded == 0 and encoder.encoded == []


def test_refresh_without_api_key_still_builds_the_model():
    result = build(client=None, run_evaluation=False)
    report = json.loads(config.EVAL_REPORT.read_text("utf-8"))

    assert result.refreshed and result.winner == "embeddings"
    assert report["metrics"] == {} and "skipped" in report["note"]


def test_empty_database_does_not_build_anything():
    result = refresh_model.run_refresh(encoder=FakeEncoder(), client=FakeClient())

    assert (result.refreshed, result.reason) == (False, "database has no movies")
    assert not config.NEIGHBOURS_ARTIFACT.exists()


# -------------------------------- lookup ----------------------------------- #
def test_recommend_handles_unknown_movie_and_bad_method():
    build()
    artifact = recommender.load_neighbours()

    assert recommender.recommend(artifact, 424242) == []
    assert len(recommender.recommend(artifact, 1, n=10)) == 3  # only 3 other movies exist
    with pytest.raises(ValueError, match="Unknown method"):
        recommender.recommend(artifact, 1, method="magic")


def test_load_neighbours_returns_none_before_the_first_build():
    assert recommender.load_neighbours() is None
