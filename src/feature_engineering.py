"""
Feature engineering module.

Transforms the textual ``tags`` column into a TF-IDF matrix and
computes the movie-to-movie cosine similarity matrix.
"""

import logging

import numpy as np
from scipy.sparse import issparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger("movie_recommender")


def build_tfidf_matrix(tags: "pd.Series"):
    """
    Fit a TF-IDF vectorizer on the tags and return both the
    vectorizer and the sparse TF-IDF matrix.

    Parameters
    ----------
    tags : pd.Series
        Series of normalised tag strings (one per movie).

    Returns
    -------
    tfidf : TfidfVectorizer
        Fitted vectorizer (can be persisted for later use).
    tfidf_matrix : sparse matrix
        Sparse TF-IDF matrix of shape (n_movies, n_features).
    """
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(tags)

    print(f"\nTF-IDF Matrix")
    print(f"  Shape     : {tfidf_matrix.shape}")
    print(f"  Movies    : {tfidf_matrix.shape[0]}")
    print(f"  Features  : {tfidf_matrix.shape[1]}")
    print(f"  Sparse    : {issparse(tfidf_matrix)}\n")

    logger.info(
        "TF-IDF matrix built: %d movies × %d features (sparse=%s)",
        tfidf_matrix.shape[0],
        tfidf_matrix.shape[1],
        issparse(tfidf_matrix),
    )
    return tfidf, tfidf_matrix


def compute_cosine_similarity(tfidf_matrix) -> np.ndarray:
    """
    Compute the full movie-to-movie cosine similarity matrix.

    Each cell ``cosine_sim[i][j]`` represents the textual similarity
    between movie *i* and movie *j*:
      - 1.0 → identical tag content
      - 0.0 → no overlap

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        TF-IDF matrix of shape (n_movies, n_features).

    Returns
    -------
    np.ndarray
        Dense cosine similarity matrix of shape (n_movies, n_movies).
    """
    logger.info("Computing cosine similarity matrix ...")
    cosine_sim = cosine_similarity(tfidf_matrix)
    logger.info("Cosine similarity matrix shape: %s", cosine_sim.shape)
    return cosine_sim
