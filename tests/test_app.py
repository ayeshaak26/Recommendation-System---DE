"""App tests using Streamlit's headless AppTest runner against a small temporary catalogue."""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline import config, load, refresh_model
from test_load import stage
from test_model import CATALOGUE, FakeClient, FakeEncoder, movie

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest

APP = str(Path(__file__).resolve().parents[1] / "app.py")
DAY = "2026-11-20"


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
    monkeypatch.setattr(config, "TFIDF_PARAMS", {"stop_words": "english"})


def build_catalogue(with_model: bool = True) -> None:
    # Movie 5 has no poster, no trailer and no votes: the awkward case for the UI.
    bare = movie(5, "Bare <Movie>", "A quiet film about love with nothing attached to it at all.", "Romance")
    bare.update(poster_path=None, videos={"results": []}, vote_count=0, vote_average=0)
    stage(DAY, [*CATALOGUE, bare])
    load.run_load(DAY)
    if with_model:
        refresh_model.run_refresh(encoder=FakeEncoder(), client=FakeClient())


def run_app() -> AppTest:
    app = AppTest.from_file(APP, default_timeout=30).run()
    assert not app.exception, app.exception
    return app


def page_text(app: AppTest) -> str:
    return " ".join(str(el.value) for el in [*app.markdown, *app.caption, *app.subheader, *app.info, *app.warning])


def test_landing_page_shows_popular_movies_and_the_tmdb_attribution():
    build_catalogue()
    app = run_app()

    assert app.subheader[0].value == "Popular right now"
    assert "Star Voyage" in page_text(app)
    assert config.TMDB_ATTRIBUTION in page_text(app)


def test_selecting_a_movie_shows_its_recommendations():
    build_catalogue()
    app = run_app()
    app.selectbox[0].select(1).run()

    assert not app.exception
    assert [s.value for s in app.subheader] == ["Star Voyage", "You might also like"]
    assert "Galaxy Run" in page_text(app)

    next(b for b in app.button if b.key.startswith("details-rec-1-") and b.key.endswith("-2")).click().run()  # open the pop-up
    assert not app.exception
    assert "Similarity score:" in page_text(app)


def test_switching_method_and_missing_poster_trailer_rating_do_not_break_the_page():
    build_catalogue()
    app = run_app()
    app.selectbox[0].select(5).run()          # the movie with nothing attached
    app.radio[0].set_value("embeddings").run()

    text = page_text(app)
    assert not app.exception
    assert "Not yet rated" in text and "No trailer available for this movie." in text
    assert app.subheader[0].value == "Bare <Movie>"


def test_find_similar_button_selects_that_movie():
    build_catalogue()
    app = run_app()
    next(b for b in app.button if b.key == "details-popular-3").click().run()   # open the pop-up
    assert any(b.key == "dialog-similar" for b in app.button)
    # AppTest cannot click inside a dialog the way a browser does (it reruns the whole page,
    # which closes the dialog first), so the button's effect is applied directly.
    app.session_state["pending_movie"] = 3
    app.run()

    assert app.selectbox[0].value == 3
    assert app.subheader[0].value == "Seaside Hearts"


def test_missing_database_and_missing_model_show_friendly_messages():
    app = run_app()
    assert "has not been built yet" in app.error[0].value

    build_catalogue(with_model=False)
    app = run_app()
    app.selectbox[0].select(1).run()
    assert not app.exception
    assert "recommender has not been built yet" in app.warning[0].value
