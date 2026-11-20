"""Central configuration for the pipeline and the app.

Every path and tunable setting lives here so no other module hardcodes them.
Secrets come from the .env file (local) or environment variables (Task Scheduler,
Streamlit Cloud) - never from source code.

Importing this module has no side effects beyond reading .env: it does not create
folders or require the API key. Call `ensure_directories()` / `require_api_key()`
explicitly from the code that needs them.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
# Resolved from this file's location, NOT the current working directory, so the
# pipeline behaves the same from a terminal, a notebook, or Task Scheduler
# (which often starts in C:\Windows\System32).
PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_DIR: Path = DATA_DIR / "raw"                # untouched API JSON, one folder per run date
STAGED_DIR: Path = DATA_DIR / "staged"          # cleaned + validated Parquet
CURATED_DIR: Path = DATA_DIR / "curated"        # SQLite DB + model artifacts, read by the app
QUARANTINE_DIR: Path = DATA_DIR / "quarantine"  # rows that failed validation, kept for review
LOG_DIR: Path = PROJECT_ROOT / "logs"
NOTEBOOK_DIR: Path = PROJECT_ROOT / "notebooks"
REPORT_DIR: Path = NOTEBOOK_DIR / "reports"     # dated papermill output notebooks

DB_PATH: Path = CURATED_DIR / "movies.db"
LOG_FILE: Path = LOG_DIR / "pipeline.log"

# Model artifacts (written by refresh_model.py)
EMBEDDINGS_ARTIFACT: Path = CURATED_DIR / "embeddings.joblib"   # vectors, reused between refreshes
NEIGHBOURS_ARTIFACT: Path = CURATED_DIR / "neighbours.joblib"   # top-N per movie, loaded by app.py
EVAL_REPORT: Path = CURATED_DIR / "model_comparison.json"
# TMDB's own recommendations, cached so the evaluation does not re-fetch them every refresh
RECS_CACHE_DIR: Path = RAW_DIR / "recommendations"

# --------------------------------------------------------------------------- #
# TMDB API
# --------------------------------------------------------------------------- #
TMDB_API_KEY: str | None = os.getenv("TMDB_API_KEY")
TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL: str = "https://image.tmdb.org/t/p/w500"
YOUTUBE_WATCH_URL: str = "https://www.youtube.com/watch?v={key}"
TMDB_ATTRIBUTION: str = (
    "This product uses the TMDB API but is not endorsed or certified by TMDB."
)

# List endpoints used to discover movie IDs. /discover/movie fills the dataset
# up to TARGET_MOVIE_COUNT once the three curated lists run out.
LIST_ENDPOINTS: tuple[str, ...] = (
    "/movie/popular",
    "/movie/now_playing",
    "/movie/upcoming",
    "/discover/movie",
)
DETAIL_APPEND: str = "credits,keywords,videos"  # one detail call returns everything we need
TMDB_MAX_PAGE: int = 500                        # TMDB hard limit on paginated endpoints (20 rows/page)
TMDB_CHANGES_MAX_DAYS: int = 14                 # /movie/changes only looks back this far

# How the ~5,000 IDs are sourced on each run:
#   1. now_playing + upcoming  -> fresh releases (a few pages each)
#   2. discover by vote count  -> well-known titles, so "Inception" is always in the set
#   3. popular                 -> fills whatever is left up to TARGET_MOVIE_COUNT
FRESH_LIST_MAX_PAGES: int = 5
DISCOVER_SHARE: float = 0.6                     # share of the target filled by top-voted movies
DISCOVER_PARAMS: dict[str, str] = {
    "sort_by": "vote_count.desc",
    "include_adult": "false",
    "include_video": "false",
}

# --------------------------------------------------------------------------- #
# Extract settings
# --------------------------------------------------------------------------- #
TARGET_MOVIE_COUNT: int = int(os.getenv("TARGET_MOVIE_COUNT", "5000"))
REQUEST_TIMEOUT_SECONDS: int = 15
REQUESTS_PER_SECOND: float = 20.0               # stay well under TMDB's ~40-50 req/s ceiling
EXTRACT_WORKERS: int = 8                        # threads for detail calls (rate limit still applies)
MAX_RETRIES: int = 5                            # for 429s, 5xx and network errors
BACKOFF_MIN_SECONDS: float = 1.0
BACKOFF_MAX_SECONDS: float = 60.0
# Incremental runs re-fetch a known movie only if it is older than this, so
# ratings and trailers drift back into sync without a full refresh.
REFRESH_STALE_AFTER_DAYS: int = 30

# --------------------------------------------------------------------------- #
# Validate / transform settings
# --------------------------------------------------------------------------- #
RATING_MIN: float = 0.0
RATING_MAX: float = 10.0
MIN_OVERVIEW_CHARS: int = 20                    # shorter overviews carry no signal for the model
TOP_CAST_COUNT: int = 5                         # cast members that go into the "tags" field
POSTER_PATH_REGEX: str = r"^/[\w\-]+\.(jpg|jpeg|png|webp|svg)$"
YOUTUBE_KEY_REGEX: str = r"^[A-Za-z0-9_-]{11}$" # YouTube video IDs are always 11 characters
TRAILER_TYPE_PREFERENCE: tuple[str, ...] = ("Trailer", "Teaser")  # teaser only if no trailer exists
STAGED_TABLES: tuple[str, ...] = ("movies", "genres", "cast_crew", "trailers")

# --------------------------------------------------------------------------- #
# Model settings
# --------------------------------------------------------------------------- #
EMBEDDING_MODEL_NAME: str = "all-MiniLM-L6-v2"
TOP_N_RECOMMENDATIONS: int = 10
NEIGHBOURS_STORED: int = 30                     # stored per movie; app shows the top 10
EVAL_SPOT_CHECK_TITLES: tuple[str, ...] = ("Inception", "Toy Story", "The Godfather", "Parasite")
EVAL_SAMPLE_SIZE: int = 300                     # most-voted movies compared against TMDB's recommendations
EMBEDDING_BATCH_SIZE: int = 64
TFIDF_PARAMS: dict = {
    "stop_words": "english",
    "ngram_range": (1, 2),      # single words and two-word phrases
    "min_df": 2,                # ignore terms that appear in only one movie
    "max_df": 0.8,              # ignore terms that appear in almost every movie
    "sublinear_tf": True,       # dampen words repeated many times in one overview
    "max_features": 50_000,
}

# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
LOG_LEVEL: str = os.getenv("LOG_LEVEL", "INFO").upper()
LOG_MAX_BYTES: int = 5 * 1024 * 1024            # rotate at 5 MB
LOG_BACKUP_COUNT: int = 5

# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
PLACEHOLDER_POSTER: Path = PROJECT_ROOT / "assets" / "poster_placeholder.png"


def ensure_directories() -> None:
    """Create every data/log folder if missing. Safe to call on every run."""
    for folder in (RAW_DIR, STAGED_DIR, CURATED_DIR, QUARANTINE_DIR, LOG_DIR, REPORT_DIR):
        folder.mkdir(parents=True, exist_ok=True)


def require_api_key() -> str:
    """Return the TMDB key, or fail with a clear message if it is not set.

    Only the extract stage calls this - transform, load, the model refresh and
    the app all work offline from data already on disk.
    """
    if not TMDB_API_KEY or TMDB_API_KEY == "your_tmdb_v3_api_key_here":
        raise RuntimeError(
            "TMDB_API_KEY is not set. Copy .env.example to .env and add your key "
            f"(expected at {PROJECT_ROOT / '.env'})."
        )
    return TMDB_API_KEY


def poster_url(poster_path: str | None) -> str | None:
    """Build a full poster URL, or None so the caller can show the placeholder."""
    return f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None


if __name__ == "__main__":
    # Quick self-check: `python -m pipeline.config`
    ensure_directories()
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Database     : {DB_PATH}")
    print(f"Target movies: {TARGET_MOVIE_COUNT}")
    print(f"API key set  : {bool(TMDB_API_KEY) and TMDB_API_KEY != 'your_tmdb_v3_api_key_here'}")
