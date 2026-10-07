"""
Recommender module.

Implements the ensemble recommendation logic described in the project
report:

1. Cosine-similarity recommendations.
2. KNN-based recommendations.
3. Counter-based voting to combine both lists.
4. Return top-N movies.

The SVM, Decision Tree, and Gradient Boosting models are trained and
saved as part of the project methodology but are NOT used in the
default recommendation function.  An optional *experimental* mode
that incorporates all models is provided separately.
"""

import logging
import sys
from collections import Counter
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import joblib
from difflib import get_close_matches

from src.utils import (
    COSINE_SIM_PATH,
    KNN_MODEL_PATH,
    PROCESSED_MOVIES_PATH,
    TFIDF_MATRIX_PATH,
    TFIDF_VECTORIZER_PATH,
    SVM_MODEL_PATH,
    DECISION_TREE_PATH,
    GRADIENT_BOOSTING_PATH,
)

logger = logging.getLogger("movie_recommender")


# ---------------------------------------------------------------------------
# Artifact loading
# ---------------------------------------------------------------------------

def _load_artifact(path, name: str):
    """Load a joblib artifact or exit with a clear message."""
    if not path.exists():
        print(
            f"\n[ERROR] Model artifact not found: {path}\n"
            f"Please run  python train.py  first to generate the models.\n"
        )
        sys.exit(1)
    try:
        return joblib.load(path)
    except Exception as exc:
        print(f"\n[ERROR] Could not load {name}: {exc}\n")
        sys.exit(1)


def load_recommendation_artifacts() -> dict:
    """Load all artifacts required by the recommender."""
    artifacts = {
        "movies": _load_artifact(PROCESSED_MOVIES_PATH, "processed movies"),
        "cosine_sim": _load_artifact(COSINE_SIM_PATH, "cosine similarity"),
        "knn_model": _load_artifact(KNN_MODEL_PATH, "KNN model"),
        "tfidf_matrix": _load_artifact(TFIDF_MATRIX_PATH, "TF-IDF matrix"),
    }
    return artifacts


# ---------------------------------------------------------------------------
# Movie title lookup
# ---------------------------------------------------------------------------

def find_movie_index(
    movie_title: str, movies: pd.DataFrame
) -> Optional[int]:
    """
    Find the DataFrame index of a movie by title (case-insensitive).

    Returns ``None`` if no match is found.
    """
    # Build a lowercase-title → index mapping
    lower_titles = movies["title"].str.lower()
    matches = movies.index[lower_titles == movie_title.lower()]
    if len(matches) == 0:
        return None
    # Deterministic: return the first match
    return int(matches[0])


def suggest_titles(
    movie_title: str, movies: pd.DataFrame, n: int = 5
) -> List[str]:
    """Return the closest matching movie titles."""
    all_titles = movies["title"].tolist()
    lower_titles = [t.lower() for t in all_titles]
    close = get_close_matches(movie_title.lower(), lower_titles, n=n, cutoff=0.4)
    # Map back to original casing
    suggestions = []
    for c in close:
        idx = lower_titles.index(c)
        suggestions.append(all_titles[idx])
    return suggestions


# ---------------------------------------------------------------------------
# Cosine-similarity recommendations
# ---------------------------------------------------------------------------

def _cosine_recommendations(
    idx: int, cosine_sim: np.ndarray, n: int
) -> List[int]:
    """
    Return the top-N most similar movie indices using cosine similarity.

    Ignores the input movie itself (cosine_sim[idx][idx] == 1.0).
    """
    # Enumerate similarity scores for the target movie
    sim_scores = list(enumerate(cosine_sim[idx]))
    # Sort by similarity descending
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    # Skip the first entry (the movie itself)
    sim_scores = sim_scores[1 : n + 1]
    return [i for i, _ in sim_scores]


# ---------------------------------------------------------------------------
# KNN recommendations
# ---------------------------------------------------------------------------

def _knn_recommendations(
    idx: int, knn_model, tfidf_matrix, n: int
) -> List[int]:
    """
    Return the top-N nearest-neighbour movie indices using the trained
    KNN model.

    Ignores the input movie itself.
    """
    n_neighbors = min(n + 1, knn_model.n_samples_fit_)
    distances, indices = knn_model.kneighbors(
        tfidf_matrix[idx], n_neighbors=n_neighbors
    )
    # Flatten and remove the input movie
    neighbor_indices = indices.flatten().tolist()
    neighbor_indices = [i for i in neighbor_indices if i != idx]
    return neighbor_indices[:n]


# ---------------------------------------------------------------------------
# Ensemble recommendation (default / original)
# ---------------------------------------------------------------------------

def get_ensemble_recommendations(
    movie_title: str,
    n_recommendations: int = 10,
    artifacts: Optional[dict] = None,
) -> Tuple[str, List[str]]:
    """
    Generate movie recommendations using the ensemble method from the
    original project report.

    Algorithm
    ---------
    1. Find the movie index.
    2. Get cosine-similarity recommendations.
    3. Get KNN recommendations.
    4. Combine both lists.
    5. Vote using ``Counter``.
    6. Return the top-N most-voted movie titles.

    Parameters
    ----------
    movie_title : str
        Title of the movie to base recommendations on.
    n_recommendations : int
        Number of movies to recommend.
    artifacts : dict, optional
        Pre-loaded artifacts (avoids reloading on every call).

    Returns
    -------
    matched_title : str
        The exact title as stored in the dataset.
    recommendations : list of str
        Recommended movie titles (up to ``n_recommendations``).
    """
    # --- Load artifacts if not provided -----------------------------------
    if artifacts is None:
        artifacts = load_recommendation_artifacts()

    movies = artifacts["movies"]
    cosine_sim = artifacts["cosine_sim"]
    knn_model = artifacts["knn_model"]
    tfidf_matrix = artifacts["tfidf_matrix"]

    # --- Step 1: find the movie -------------------------------------------
    idx = find_movie_index(movie_title, movies)
    if idx is None:
        suggestions = suggest_titles(movie_title, movies)
        msg = f"Movie '{movie_title}' was not found in the dataset."
        if suggestions:
            msg += "\n\nDid you mean one of these?\n"
            for s in suggestions:
                msg += f"  - {s}\n"
        return movie_title, [msg]

    matched_title = movies.loc[idx, "title"]

    # --- Step 2: cosine-similarity recommendations ------------------------
    cosine_indices = _cosine_recommendations(
        idx, cosine_sim, n_recommendations
    )

    # --- Step 3: KNN recommendations --------------------------------------
    knn_indices = _knn_recommendations(
        idx, knn_model, tfidf_matrix, n_recommendations
    )

    # --- Step 4: combine both lists ---------------------------------------
    combined_indices = cosine_indices + knn_indices

    # --- Step 5: vote using Counter ---------------------------------------
    vote_counter = Counter(combined_indices)

    # --- Step 6: most common → top N --------------------------------------
    # Remove the input movie from the counter (safety)
    vote_counter.pop(idx, None)
    top_indices = [i for i, _ in vote_counter.most_common(n_recommendations)]

    # --- Step 7: map indices to titles ------------------------------------
    recommendations = movies.loc[top_indices, "title"].tolist()

    return matched_title, recommendations
