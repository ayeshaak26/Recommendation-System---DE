"""REFRESH stage: rebuild the recommender, but only when the catalogue has changed.

Two content-based methods are built and compared on every refresh:

  A) TF-IDF + cosine similarity          on the `tags` field (keyword soup)
  B) Sentence embeddings + cosine sim.   on the `embedding_text` field (readable sentences),
                                         using all-MiniLM-L6-v2

For each method we precompute every movie's top-N neighbours and save them, so the app
answers a request with a single array lookup instead of loading a model.

Refresh rules
-------------
- Skip entirely if the set of movie IDs in the database matches the saved model.
- When rebuilding, embeddings are only computed for movies whose text is new or changed;
  the rest are reused from the previous artifact. A daily run with 20 new movies encodes
  20 texts, not 5,000.

Evaluation
----------
- Overlap with TMDB's own /movie/{id}/recommendations for the most-voted movies:
  precision@10, recall@10 and hit rate. TMDB's list comes from user behaviour, not
  content, so absolute overlap is expected to be low; what matters is which method
  scores higher.
- Qualitative spot-checks: the top 10 for a few well-known titles, saved in the report.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd

from pipeline import config, recommender
from pipeline.logging_setup import get_logger

log = get_logger(__name__)

Encoder = Callable[[list[str]], np.ndarray]
K = config.TOP_N_RECOMMENDATIONS


# --------------------------------------------------------------------------- #
# Data
# --------------------------------------------------------------------------- #
def load_catalogue() -> pd.DataFrame:
    """Read the model inputs from the curated database, in a stable order."""
    if not config.DB_PATH.exists():
        return pd.DataFrame(columns=["movie_id", "title", "tags", "embedding_text", "vote_count"])
    with closing(sqlite3.connect(config.DB_PATH)) as conn:
        return pd.read_sql_query(
            "SELECT movie_id, title, tags, embedding_text, vote_count FROM movies ORDER BY movie_id", conn
        )


def _atomic_dump(obj: Any, path: Path) -> None:
    """Write to a temp file then swap, so the app never reads a half-written artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(obj, tmp, compress=3)
    tmp.replace(path)


# --------------------------------------------------------------------------- #
# Similarity
# --------------------------------------------------------------------------- #
def top_k_neighbours(matrix: Any, k: int, chunk_size: int = 512) -> tuple[np.ndarray, np.ndarray]:
    """Top-k most similar rows for every row of an L2-normalised matrix.

    With normalised rows, the dot product IS the cosine similarity. Working in chunks
    keeps memory flat: we never hold the full n x n similarity matrix (100 MB at 5,000
    movies), only chunk_size x n at a time.

    Returns (indices, scores), each of shape (n, k), best match first.
    """
    n = matrix.shape[0]
    k = max(0, min(k, n - 1))
    indices = np.zeros((n, k), dtype=np.int32)
    scores = np.zeros((n, k), dtype=np.float32)
    if k == 0:
        return indices, scores

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        sims = matrix[start:stop] @ matrix.T
        sims = np.asarray(sims.toarray() if hasattr(sims, "toarray") else sims, dtype=np.float32)
        sims[np.arange(stop - start), np.arange(start, stop)] = -np.inf  # a movie is not its own neighbour

        top = np.argpartition(-sims, k - 1, axis=1)[:, :k]               # the k best, unordered
        top_scores = np.take_along_axis(sims, top, axis=1)
        order = np.argsort(-top_scores, axis=1)                          # now sort just those k
        indices[start:stop] = np.take_along_axis(top, order, axis=1)
        scores[start:stop] = np.take_along_axis(top_scores, order, axis=1)
    return indices, scores


def build_tfidf_matrix(tags: list[str]) -> Any:
    """Method A: sparse TF-IDF matrix (rows are L2-normalised by default)."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return TfidfVectorizer(**config.TFIDF_PARAMS).fit_transform(tags)


def default_encoder(texts: list[str]) -> np.ndarray:
    """Method B: encode texts with the sentence-transformers model (CPU)."""
    from sentence_transformers import SentenceTransformer  # heavy import, only when needed

    model = SentenceTransformer(config.EMBEDDING_MODEL_NAME, device="cpu")
    return model.encode(
        texts,
        batch_size=config.EMBEDDING_BATCH_SIZE,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )


def _text_hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def build_embeddings(catalogue: pd.DataFrame, encoder: Encoder | None = None) -> tuple[np.ndarray, int]:
    """Embedding matrix aligned with `catalogue`, reusing saved vectors where the text is unchanged.

    Returns (matrix, number of texts actually encoded this time).
    """
    encoder = encoder or default_encoder
    texts = catalogue["embedding_text"].tolist()
    hashes = [_text_hash(t) for t in texts]

    cached: dict[tuple[int, str], np.ndarray] = {}
    if config.EMBEDDINGS_ARTIFACT.exists():
        old = joblib.load(config.EMBEDDINGS_ARTIFACT)
        if old.get("model_name") == config.EMBEDDING_MODEL_NAME:
            cached = {(int(m), h): v for m, h, v in zip(old["movie_ids"], old["text_hashes"], old["embeddings"])}

    keys = list(zip(catalogue["movie_id"].astype(int), hashes))
    todo = [i for i, key in enumerate(keys) if key not in cached]
    log.info("Embeddings: %d reused, %d to encode", len(keys) - len(todo), len(todo))

    fresh = np.asarray(encoder([texts[i] for i in todo]), dtype=np.float32) if todo else None
    fresh_at = {i: n for n, i in enumerate(todo)}
    matrix = np.vstack([fresh[fresh_at[i]] if i in fresh_at else cached[key] for i, key in enumerate(keys)])
    matrix = matrix.astype(np.float32)
    matrix /= np.clip(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-12, None)  # guarantee unit length

    _atomic_dump(
        {
            "model_name": config.EMBEDDING_MODEL_NAME,
            "movie_ids": catalogue["movie_id"].to_numpy(dtype=np.int64),
            "text_hashes": hashes,
            "embeddings": matrix,
        },
        config.EMBEDDINGS_ARTIFACT,
    )
    return matrix, len(todo)


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
def overlap_metrics(recommended: list[int], relevant: set[int]) -> tuple[float, float, float]:
    """(precision@k, recall@k, hit) of our list against TMDB's list."""
    hits = len(set(recommended) & relevant)
    return hits / max(len(recommended), 1), hits / max(len(relevant), 1), float(hits > 0)


def fetch_tmdb_recommendations(client: Any, movie_id: int) -> list[int]:
    """TMDB's recommended movie IDs (first page, up to 20), cached in the raw layer."""
    cache = config.RECS_CACHE_DIR / f"{movie_id}.json"
    if cache.exists():
        data = json.loads(cache.read_text(encoding="utf-8"))
    else:
        data = client.get(f"/movie/{movie_id}/recommendations")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return [r["id"] for r in data.get("results") or [] if isinstance(r.get("id"), int)]


def _make_client() -> Any | None:
    """A TMDB client, or None when there is no API key (evaluation is then skipped)."""
    from pipeline.extract import TMDBClient

    try:
        return TMDBClient(config.require_api_key())
    except RuntimeError:
        return None


def evaluate(artifact: dict[str, Any], catalogue: pd.DataFrame, client: Any | None) -> dict[str, Any]:
    """Score both methods against TMDB's recommendations and collect spot-checks."""
    from pipeline.extract import TMDBError

    titles = dict(zip(catalogue["movie_id"].astype(int), catalogue["title"]))
    in_catalogue = set(titles)
    report: dict[str, Any] = {"metrics": {}, "evaluated_movies": 0, "note": None}

    # --- overlap with TMDB ---------------------------------------------------
    tmdb_recs: dict[int, set[int]] = {}
    if client is None:
        report["note"] = "No TMDB API key: overlap evaluation skipped."
    else:
        sample = catalogue.nlargest(config.EVAL_SAMPLE_SIZE, "vote_count")["movie_id"].astype(int)
        for movie_id in sample:
            try:
                relevant = set(fetch_tmdb_recommendations(client, movie_id)) & in_catalogue
            except TMDBError as exc:
                log.warning("No TMDB recommendations for %s: %s", movie_id, exc)
                continue
            if relevant:  # we can only be judged on movies we actually hold
                tmdb_recs[movie_id] = relevant
        report["evaluated_movies"] = len(tmdb_recs)

    for method in recommender.METHODS:
        rows = [
            overlap_metrics([m for m, _ in recommender.recommend(artifact, movie_id, method, K)], relevant)
            for movie_id, relevant in tmdb_recs.items()
        ]
        if rows:
            precision, recall, hit = (round(float(x), 4) for x in np.mean(rows, axis=0))
            report["metrics"][method] = {f"precision@{K}": precision, f"recall@{K}": recall, "hit_rate": hit}

    # --- how much do the two methods agree with each other? -------------------
    a, b = (artifact["methods"][m]["indices"][:, :K] for m in recommender.METHODS)
    if a.shape[1]:
        shared = [len(set(x) & set(y)) for x, y in zip(a, b)]
        report["avg_shared_recommendations"] = round(float(np.mean(shared)), 2)

    # --- winner ----------------------------------------------------------------
    metrics = report["metrics"]
    if len(metrics) == 2:
        report["winner"] = max(metrics, key=lambda m: (metrics[m][f"recall@{K}"], m == "embeddings"))
    else:
        report["winner"] = "embeddings"  # no evidence either way: default to the semantic model

    # --- spot-checks -------------------------------------------------------------
    report["spot_checks"] = {}
    for title in config.EVAL_SPOT_CHECK_TITLES:
        matches = catalogue[catalogue["title"] == title].nlargest(1, "vote_count")
        if matches.empty:
            continue
        movie_id = int(matches["movie_id"].iloc[0])
        report["spot_checks"][title] = {
            method: [titles[m] for m, _ in recommender.recommend(artifact, movie_id, method, K)]
            for method in recommender.METHODS
        }
        if movie_id in tmdb_recs:
            report["spot_checks"][title]["tmdb"] = sorted(titles[m] for m in tmdb_recs[movie_id])
    return report


# --------------------------------------------------------------------------- #
# Stage entry point
# --------------------------------------------------------------------------- #
@dataclass
class RefreshResult:
    """Summary of one refresh; `refreshed` goes into pipeline_runs.model_refreshed."""

    refreshed: bool
    reason: str
    n_movies: int = 0
    n_encoded: int = 0
    winner: str | None = None
    metrics: dict[str, Any] = field(default_factory=dict)


def needs_refresh(catalogue: pd.DataFrame) -> tuple[bool, str]:
    """Decide whether the saved model still matches the database."""
    if catalogue.empty:
        return False, "database has no movies"
    artifact = recommender.load_neighbours()
    if artifact is None or not config.EMBEDDINGS_ARTIFACT.exists():
        return True, "no saved model yet"
    current, saved = set(catalogue["movie_id"].astype(int)), set(artifact["movie_ids"].tolist())
    if current != saved:
        return True, f"{len(current - saved)} new and {len(saved - current)} removed movies"
    return False, "no new movies since the last build"


def run_refresh(
    force: bool = False,
    encoder: Encoder | None = None,
    client: Any | None = None,
    run_evaluation: bool = True,
) -> RefreshResult:
    """Rebuild both models, compare them and save the artifacts - if anything changed.

    Args:
        force: rebuild even when no movies were added (e.g. after changing TFIDF_PARAMS).
        encoder, client: injectable for tests (no PyTorch, no network).
        run_evaluation: set False to skip the TMDB overlap comparison.
    """
    catalogue = load_catalogue()
    refresh, reason = needs_refresh(catalogue)
    if force and not catalogue.empty:
        refresh, reason = True, f"forced ({reason})"
    if not refresh:
        log.info("Model refresh skipped: %s", reason)
        return RefreshResult(refreshed=False, reason=reason, n_movies=len(catalogue))

    log.info("Model refresh started for %d movies: %s", len(catalogue), reason)
    stored = config.NEIGHBOURS_STORED

    tfidf_idx, tfidf_scores = top_k_neighbours(build_tfidf_matrix(catalogue["tags"].tolist()), stored)
    log.info("TF-IDF neighbours built")
    embeddings, n_encoded = build_embeddings(catalogue, encoder)
    emb_idx, emb_scores = top_k_neighbours(embeddings, stored)
    log.info("Embedding neighbours built")

    artifact = {
        "movie_ids": catalogue["movie_id"].to_numpy(dtype=np.int64),
        "methods": {
            "tfidf": {"indices": tfidf_idx, "scores": tfidf_scores},
            "embeddings": {"indices": emb_idx, "scores": emb_scores},
        },
        "default_method": "embeddings",
        "built_at": datetime.now().isoformat(timespec="seconds"),
    }

    if client is None and run_evaluation:
        client = _make_client()
    report = evaluate(artifact, catalogue, client if run_evaluation else None)
    report.update(built_at=artifact["built_at"], n_movies=len(catalogue))
    artifact["default_method"] = report["winner"]

    _atomic_dump(artifact, config.NEIGHBOURS_ARTIFACT)
    config.EVAL_REPORT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Model refresh finished. Winner: %s %s", report["winner"], report["metrics"])

    return RefreshResult(
        refreshed=True, reason=reason, n_movies=len(catalogue), n_encoded=n_encoded,
        winner=report["winner"], metrics=report["metrics"],
    )


if __name__ == "__main__":
    # Stand-alone entry point: python -m pipeline.refresh_model [--force]
    parser = argparse.ArgumentParser(description="Rebuild the recommender if the catalogue changed.")
    parser.add_argument("--force", action="store_true", help="rebuild even if no movies were added")
    print(json.dumps(asdict(run_refresh(force=parser.parse_args().force)), indent=2))
