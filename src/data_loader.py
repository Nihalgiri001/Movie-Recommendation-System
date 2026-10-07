"""
Data loading module.

Handles loading, validating, and merging the TMDB 5000 Movies
and Credits CSV datasets.
"""

import sys
import logging
from typing import Tuple

import pandas as pd

from src.utils import MOVIES_FILE, CREDITS_FILE

logger = logging.getLogger("movie_recommender")

# Minimum columns expected in each dataset
EXPECTED_MOVIE_COLS = {"title", "overview", "genres", "keywords"}
EXPECTED_CREDIT_COLS = {"title", "cast"}


def _validate_file(path) -> None:
    """Raise a clear error when a required dataset file is missing."""
    if not path.exists():
        print(
            f"\n[ERROR] Required dataset file not found:\n"
            f"  {path}\n\n"
            f"Please download the TMDB 5000 Movies Dataset and place the\n"
            f"following files in the 'data/' directory:\n"
            f"  - tmdb_5000_movies.csv\n"
            f"  - tmdb_5000_credits.csv\n"
        )
        sys.exit(1)


def load_movies() -> pd.DataFrame:
    """Load and return the movies CSV as a DataFrame."""
    _validate_file(MOVIES_FILE)
    logger.info("Loading movies dataset from %s", MOVIES_FILE.name)
    movies = pd.read_csv(MOVIES_FILE)
    # Strip whitespace from column names
    movies.columns = movies.columns.str.strip()
    return movies


def load_credits() -> pd.DataFrame:
    """Load and return the credits CSV as a DataFrame."""
    _validate_file(CREDITS_FILE)
    logger.info("Loading credits dataset from %s", CREDITS_FILE.name)
    credits = pd.read_csv(CREDITS_FILE)
    # Strip whitespace from column names
    credits.columns = credits.columns.str.strip()
    return credits


def merge_datasets(
    movies: pd.DataFrame, credits: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge movies and credits on ``movie_id``.

    Steps
    -----
    1. Rename ``id`` → ``movie_id`` in the movies DataFrame if needed.
    2. Ensure the credits DataFrame has ``movie_id``.
    3. Merge on ``movie_id``.
    4. Print summary statistics.
    """
    # --- Standardise the merge key in the movies DataFrame ----------------
    if "id" in movies.columns and "movie_id" not in movies.columns:
        movies = movies.rename(columns={"id": "movie_id"})
        logger.info("Renamed 'id' → 'movie_id' in movies dataset.")
    elif "movie_id" not in movies.columns:
        logger.error(
            "Movies dataset has neither 'id' nor 'movie_id' column."
        )
        sys.exit(1)

    # --- Standardise the merge key in the credits DataFrame ---------------
    if "movie_id" not in credits.columns:
        # Some versions of the dataset use just 'id'
        if "id" in credits.columns:
            credits = credits.rename(columns={"id": "movie_id"})
            logger.info("Renamed 'id' → 'movie_id' in credits dataset.")
        else:
            logger.error(
                "Credits dataset has neither 'movie_id' nor 'id' column."
            )
            sys.exit(1)

    # --- Validate expected columns ----------------------------------------
    missing_movie = EXPECTED_MOVIE_COLS - set(movies.columns)
    if missing_movie:
        logger.warning(
            "Movies dataset is missing expected columns: %s", missing_movie
        )

    missing_credit = EXPECTED_CREDIT_COLS - set(credits.columns)
    if missing_credit:
        logger.warning(
            "Credits dataset is missing expected columns: %s", missing_credit
        )

    # --- Merge ------------------------------------------------------------
    # Avoid duplicate 'title' columns: drop title from credits before merge
    credits_to_merge = credits.drop(columns=["title"], errors="ignore")
    merged = pd.merge(movies, credits_to_merge, on="movie_id", how="inner")

    logger.info("Merged dataset: %d rows × %d columns", *merged.shape)

    # Print summary
    print(f"\n{'='*50}")
    print("DATASET SUMMARY")
    print(f"{'='*50}")
    print(f"  Rows   : {merged.shape[0]}")
    print(f"  Columns: {merged.shape[1]}")
    print(f"\nSample records (first 3):")
    print(merged[["movie_id", "title", "genres", "overview"]].head(3).to_string(index=False))

    # Missing values for key columns
    key_cols = ["overview", "genres", "keywords", "cast"]
    available = [c for c in key_cols if c in merged.columns]
    print(f"\nMissing values in key columns:")
    for col in available:
        n_missing = merged[col].isna().sum()
        print(f"  {col:12s}: {n_missing}")
    print(f"{'='*50}\n")

    return merged


def load_and_merge() -> pd.DataFrame:
    """Convenience function: load both files and return the merged DataFrame."""
    movies = load_movies()
    credits = load_credits()
    return merge_datasets(movies, credits)
