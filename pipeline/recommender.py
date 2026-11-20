"""Read-side of the recommender: look up precomputed neighbours.

Deliberately light (numpy + joblib only) so the Streamlit app never has to import
scikit-learn or PyTorch. All the heavy work happens in refresh_model.py.

neighbours.joblib layout
------------------------
{
    "movie_ids":      int array, shape (n_movies,)          row i <-> movie_ids[i]
    "methods": {
        "tfidf":      {"indices": (n_movies, k) int, "scores": (n_movies, k) float},
        "embeddings": {"indices": ...,                "scores": ...},
    },
    "default_method": "tfidf" | "embeddings"               the evaluation winner
    "built_at":       ISO timestamp
}
`indices` hold row numbers (not movie IDs), best match first.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib

from pipeline import config

METHODS = ("tfidf", "embeddings")


def load_neighbours(path: Path | None = None) -> dict[str, Any] | None:
    """Load the neighbours artifact, or None if the model has not been built yet."""
    path = path or config.NEIGHBOURS_ARTIFACT
    return joblib.load(path) if path.exists() else None


def recommend(
    artifact: dict[str, Any],
    movie_id: int,
    method: str | None = None,
    n: int = config.TOP_N_RECOMMENDATIONS,
) -> list[tuple[int, float]]:
    """Return up to `n` (movie_id, similarity) pairs for a movie, best first.

    An unknown movie returns an empty list rather than raising, so the app can show
    a friendly "not found" message.
    """
    method = method or artifact["default_method"]
    if method not in artifact["methods"]:
        raise ValueError(f"Unknown method {method!r}; choose from {list(artifact['methods'])}")

    movie_ids = artifact["movie_ids"]
    rows = (movie_ids == movie_id).nonzero()[0]
    if len(rows) == 0:
        return []
    table = artifact["methods"][method]
    row = rows[0]
    return [
        (int(movie_ids[i]), float(s))
        for i, s in zip(table["indices"][row][:n], table["scores"][row][:n])
    ]
