"""
Utility functions and project configuration.

Centralizes all path definitions and shared helper functions
so that no other module needs to hard-code paths.
"""

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Project paths (relative to the project root)
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "outputs"

MOVIES_FILE = DATA_DIR / "tmdb_5000_movies.csv"
CREDITS_FILE = DATA_DIR / "tmdb_5000_credits.csv"

# Model artifact filenames
TFIDF_VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"
COSINE_SIM_PATH = MODEL_DIR / "cosine_similarity.pkl"
KNN_MODEL_PATH = MODEL_DIR / "knn_model.pkl"
SVM_MODEL_PATH = MODEL_DIR / "svm_model.pkl"
DECISION_TREE_PATH = MODEL_DIR / "decision_tree_model.pkl"
GRADIENT_BOOSTING_PATH = MODEL_DIR / "gradient_boosting_model.pkl"
PROCESSED_MOVIES_PATH = MODEL_DIR / "processed_movies.pkl"
TFIDF_MATRIX_PATH = MODEL_DIR / "tfidf_matrix.pkl"

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project-wide logger."""
    logger = logging.getLogger("movie_recommender")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "[%(levelname)s] %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def ensure_directory(path: Path) -> None:
    """Create a directory (and parents) if it does not exist."""
    path.mkdir(parents=True, exist_ok=True)
