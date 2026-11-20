"""Streamlit app: pick a movie, get ten similar ones with posters and trailers.

    streamlit run app.py

The app only READS the curated layer (data/curated/movies.db + neighbours.joblib) that the
pipeline produces. It never calls the TMDB API, so it needs no API key, starts fast and
keeps working when TMDB is unreachable. Poster images and trailers are loaded by the
visitor's browser straight from TMDB's image CDN and YouTube.
"""

from __future__ import annotations

import html
import json
import sqlite3
from contextlib import closing
from typing import Any

import pandas as pd
import streamlit as st

from pipeline import config, recommender

GRID_COLUMNS = 5
METHOD_LABELS = {"tfidf": "TF-IDF (keywords, cast, crew)", "embeddings": "Sentence embeddings (meaning)"}
MOVIE_COLUMNS = (
    "movie_id, title, release_year, vote_average, vote_count, popularity, runtime, genres_text, "
    "overview, tagline, director, top_cast, poster_url, trailer_url"
)

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")

# A little CSS on top of the dark theme in .streamlit/config.toml.
st.markdown(
    """
    <style>
      .block-container {padding-top: 2.2rem; max-width: 1250px;}
      [data-testid="stImage"] img {border-radius: 8px;}
      .movie-title {font-weight: 600; font-size: 0.98rem; line-height: 1.25; margin: 0.35rem 0 0.1rem;}
      .movie-meta {color: #9aa4af; font-size: 0.83rem; line-height: 1.35;}
      .movie-meta .star {color: #f5c518;}
      .footer {color: #8b949e; font-size: 0.8rem; text-align: center; padding: 1.5rem 0 0.5rem;}
      .footer a {color: #01b4e4; text-decoration: none;}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Data access (cached)
# --------------------------------------------------------------------------- #
def _mtime(path) -> float:
    """File modification time, used as a cache key so a pipeline run invalidates the cache."""
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_data(show_spinner="Loading the movie catalogue...")
def load_movies(db_mtime: float) -> pd.DataFrame:
    """All movies from the curated database, indexed by movie_id, most popular first."""
    with closing(sqlite3.connect(config.DB_PATH)) as conn:
        movies = pd.read_sql(f"SELECT {MOVIE_COLUMNS} FROM movies ORDER BY vote_count DESC", conn)
    return movies.set_index("movie_id", drop=False)


@st.cache_data
def load_pipeline_status(db_mtime: float) -> dict[str, Any]:
    """When the data was last refreshed, for the sidebar."""
    with closing(sqlite3.connect(config.DB_PATH)) as conn:
        row = conn.execute(
            "SELECT finished_at FROM pipeline_runs WHERE status = 'success' ORDER BY run_id DESC LIMIT 1"
        ).fetchone()
    return {"last_success": row[0][:16].replace("T", " ") if row and row[0] else None}


@st.cache_resource(show_spinner="Loading the recommender...")
def load_model(artifact_mtime: float) -> dict[str, Any] | None:
    """Precomputed neighbours for both methods (a few MB; shared by all sessions)."""
    return recommender.load_neighbours()


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def has_text(value: Any) -> bool:
    """True for a non-empty string (the database returns None/NaN for missing values)."""
    return isinstance(value, str) and bool(value.strip())


def option_label(movie: pd.Series) -> str:
    """Text shown in the search box, e.g. "Inception (2010)"."""
    year = movie["release_year"]
    return f"{movie['title']} ({int(year)})" if pd.notna(year) else movie["title"]


def rating_text(movie: pd.Series) -> str:
    """Rating with its vote count, or a clear label for movies nobody has rated yet."""
    votes = movie["vote_count"]
    if pd.isna(votes) or votes == 0:
        return "Not yet rated"
    return f"<span class='star'>★</span> {movie['vote_average']:.1f} ({int(votes):,} votes)"


def meta_line(movie: pd.Series) -> str:
    """Year and rating on one line."""
    year = str(int(movie["release_year"])) if pd.notna(movie["release_year"]) else "Year unknown"
    return f"{year} · {rating_text(movie)}"


def poster(movie: pd.Series) -> str:
    """Poster URL, or the local placeholder image when TMDB has none."""
    return movie["poster_url"] if has_text(movie["poster_url"]) else str(config.PLACEHOLDER_POSTER)


def cast_list(movie: pd.Series) -> list[str]:
    """Top-billed cast (stored as a JSON array in the database)."""
    try:
        return json.loads(movie["top_cast"]) if has_text(movie["top_cast"]) else []
    except json.JSONDecodeError:
        return []


# --------------------------------------------------------------------------- #
# Page sections
# --------------------------------------------------------------------------- #
@st.dialog("Movie details", width="large")
def show_details(movie: pd.Series, score: float | None = None) -> None:
    """Pop-up with the overview and a full-size trailer for one card.

    Being a dialog, the YouTube player is only created when a visitor asks for it,
    so a page of ten cards never loads ten videos.
    """
    left, right = st.columns([1, 4], gap="medium")  # small poster, so the trailer stays in view
    with left:
        st.image(poster(movie), width="stretch")
    with right:
        st.subheader(movie["title"])
        st.markdown(f"<div class='movie-meta'>{meta_line(movie)}</div>", unsafe_allow_html=True)
        if has_text(movie["genres_text"]):
            st.caption(movie["genres_text"])
        st.write(movie["overview"])
        if score is not None:
            # Cosine similarity: useful for comparing results, not a percentage "match".
            st.caption(f"Similarity score: {score:.2f} (0 = unrelated, 1 = identical)")
    if has_text(movie["trailer_url"]):
        st.video(movie["trailer_url"])
    else:
        st.info("No trailer available for this movie.")
    # A dialog is a fragment: a click inside it only reruns the dialog. So the choice is
    # parked in session state and st.rerun() reruns the whole page, which also closes the dialog.
    if st.button("Find similar movies", key="dialog-similar", type="primary", width="stretch"):
        st.session_state["pending_movie"] = int(movie["movie_id"])
        st.rerun()


def render_card(movie: pd.Series, key: str, score: float | None = None) -> None:
    """One poster card: image, title, year, rating, genres, and a button that opens the details."""
    st.image(poster(movie), width="stretch")
    genres = movie["genres_text"] if has_text(movie["genres_text"]) else "Genre unknown"
    st.markdown(
        # Titles come from an external API, so they are escaped before going into HTML.
        f"<div class='movie-title'>{html.escape(movie['title'])}</div>"
        f"<div class='movie-meta'>{meta_line(movie)}<br>{html.escape(genres)}</div>",
        unsafe_allow_html=True,
    )
    # Short label: it has to fit a narrow column on small screens and inside an embed frame.
    if st.button("▶ Trailer", key=f"details-{key}", width="stretch"):
        show_details(movie, score)


def render_grid(items: list[tuple[pd.Series, float | None]], key_prefix: str) -> None:
    """Lay cards out in rows of GRID_COLUMNS so each row lines up at the top."""
    for start in range(0, len(items), GRID_COLUMNS):
        columns = st.columns(GRID_COLUMNS, gap="medium")
        for column, (movie, score) in zip(columns, items[start:start + GRID_COLUMNS]):
            with column:
                render_card(movie, key=f"{key_prefix}-{movie['movie_id']}", score=score)
        st.write("")


def render_selected(movie: pd.Series) -> None:
    """Details of the movie the visitor picked."""
    left, right = st.columns([1, 3], gap="large")
    with left:
        st.image(poster(movie), width="stretch")
    with right:
        st.subheader(movie["title"])
        st.markdown(f"<div class='movie-meta'>{meta_line(movie)}</div>", unsafe_allow_html=True)
        facts = []
        if has_text(movie["genres_text"]):
            facts.append(movie["genres_text"])
        if pd.notna(movie["runtime"]) and movie["runtime"] > 0:
            facts.append(f"{int(movie['runtime'])} min")
        if facts:
            st.caption(" · ".join(facts))
        if has_text(movie["tagline"]):
            st.markdown(f"*{movie['tagline']}*")
        st.write(movie["overview"])
        people = []
        if has_text(movie["director"]):
            people.append(f"**Director:** {movie['director']}")
        if cast_list(movie):
            people.append(f"**Cast:** {', '.join(cast_list(movie))}")
        if people:
            st.markdown("  \n".join(people))
        if has_text(movie["trailer_url"]):
            with st.expander("Watch the trailer"):
                st.video(movie["trailer_url"])
        else:
            st.caption("No trailer available for this movie.")


def render_footer() -> None:
    """TMDB attribution (required by the TMDB API terms of use)."""
    st.divider()
    st.markdown(
        f"<div class='footer'>{config.TMDB_ATTRIBUTION}<br>"
        "Movie data and images from <a href='https://www.themoviedb.org/' target='_blank'>TMDB</a>. "
        "Trailers are played from YouTube.</div>",
        unsafe_allow_html=True,
    )


# --------------------------------------------------------------------------- #
# Page
# --------------------------------------------------------------------------- #
def main() -> None:
    st.title("🎬 Movie Recommender")
    st.caption("Pick a movie you like and get similar ones, with posters and trailers.")

    if not config.DB_PATH.exists():
        st.error("The movie database has not been built yet. Run `python run_pipeline.py --stage all` first.")
        render_footer()
        return
    movies = load_movies(_mtime(config.DB_PATH))
    if movies.empty:
        st.error("The movie database is empty. Run `python run_pipeline.py --stage all` to fill it.")
        render_footer()
        return
    model = load_model(_mtime(config.NEIGHBOURS_ARTIFACT))

    # ------------------------------- sidebar -------------------------------- #
    with st.sidebar:
        st.header("Settings")
        methods = list(model["methods"]) if model else list(recommender.METHODS)
        default = methods.index(model["default_method"]) if model else 0
        method = st.radio(
            "Recommendation method", methods, index=default, format_func=METHOD_LABELS.get,
            help="TF-IDF matches shared words such as genres, keywords, cast and director. "
                 "Embeddings compare the meaning of the plot summaries. "
                 "The default is the one that agreed most with TMDB's own recommendations.",
        )
        count = st.slider("Number of recommendations", 5, 20, config.TOP_N_RECOMMENDATIONS, step=5)
        st.divider()
        status = load_pipeline_status(_mtime(config.DB_PATH))
        st.metric("Movies in the catalogue", f"{len(movies):,}")
        st.caption(f"Data last refreshed: {status['last_success'] or 'unknown'}")
        st.caption("An automated pipeline pulls new and changed movies from TMDB and rebuilds the model.")

    # -------------------------------- search -------------------------------- #
    # A movie chosen with "Find similar movies" becomes the selection. This has to happen
    # here, before the search box is created, because a widget's value cannot be changed afterwards.
    if "pending_movie" in st.session_state:
        st.session_state["selected_movie"] = st.session_state.pop("pending_movie")
    # A stale selection (movie removed since the session started) is cleared instead of crashing.
    if st.session_state.get("selected_movie") not in movies.index:
        st.session_state["selected_movie"] = None
    labels = {movie_id: option_label(row) for movie_id, row in movies.iterrows()}
    selected_id = st.selectbox(
        "Search for a movie", movies.index, key="selected_movie", index=None,
        format_func=labels.get, placeholder="Type a title, e.g. Inception",
        help="Start typing to filter. The list holds every movie in the catalogue, most-voted first.",
    )

    if selected_id is None:
        st.subheader("Popular right now")
        st.caption("Not sure where to start? Open a movie and choose *Find similar movies*.")
        popular = movies.sort_values("popularity", ascending=False).head(GRID_COLUMNS * 2)
        render_grid([(row, None) for _, row in popular.iterrows()], key_prefix="popular")
        render_footer()
        return

    # --------------------------- selected movie ----------------------------- #
    render_selected(movies.loc[selected_id])
    st.divider()

    # --------------------------- recommendations ---------------------------- #
    st.subheader("You might also like")
    if model is None:
        st.warning("The recommender has not been built yet. Run `python run_pipeline.py --stage refresh`.")
    else:
        pairs = recommender.recommend(model, int(selected_id), method=method, n=count)
        # Skip any neighbour that is no longer in the database.
        items = [(movies.loc[movie_id], score) for movie_id, score in pairs if movie_id in movies.index]
        if items:
            st.caption(f"Method: {METHOD_LABELS[method]}")
            render_grid(items, key_prefix=f"rec-{selected_id}-{method}")
        else:
            st.info("This movie was added after the last model build, so it has no recommendations yet. "
                    "They will appear after the next pipeline run.")

    render_footer()


main()
