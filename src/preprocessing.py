"""
Data preprocessing module.

Parses JSON-like metadata columns (genres, keywords, cast) and
creates the unified ``tags`` column used for TF-IDF vectorization.
"""

import ast
import logging
from typing import List

import pandas as pd

logger = logging.getLogger("movie_recommender")


# ---------------------------------------------------------------------------
# Safe metadata parsers
# ---------------------------------------------------------------------------

def _safe_literal_eval(value) -> list:
    """
    Safely parse a string that looks like a Python list of dicts.

    Returns an empty list on any failure (NaN, empty string, malformed).
    """
    if pd.isna(value):
        return []
    try:
        result = ast.literal_eval(str(value))
        if isinstance(result, list):
            return result
        return []
    except (ValueError, SyntaxError):
        logger.debug("Could not parse value: %s", str(value)[:80])
        return []


def extract_names(value) -> str:
    """
    Extract the ``name`` field from a list-of-dicts string.

    Used for genres, keywords, and similar columns.

    Example input  : '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
    Example output : 'Action Adventure'
    """
    items = _safe_literal_eval(value)
    names = [
        item["name"]
        for item in items
        if isinstance(item, dict) and "name" in item
    ]
    return " ".join(names)


def extract_cast_names(value, max_actors: int = 5) -> str:
    """
    Extract actor names from the cast column.

    Only the first ``max_actors`` entries are used to keep the
    representation focused on the primary/relevant cast.

    Example output: 'Robert Downey Jr. Chris Evans Scarlett Johansson'
    """
    items = _safe_literal_eval(value)
    names = [
        item["name"]
        for item in items[:max_actors]
        if isinstance(item, dict) and "name" in item
    ]
    return " ".join(names)


# ---------------------------------------------------------------------------
# Tags creation
# ---------------------------------------------------------------------------

def create_tags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a ``tags`` column by combining overview, genres, keywords, and cast.

    Steps
    -----
    1. Parse genres  → space-separated names.
    2. Parse keywords → space-separated names.
    3. Parse cast    → space-separated names (top actors).
    4. Fill missing overview with empty string.
    5. Concatenate all four fields into a single ``tags`` string.
    6. Normalise to lowercase and strip extra whitespace.

    Parameters
    ----------
    df : pd.DataFrame
        Merged movies-credits DataFrame.

    Returns
    -------
    pd.DataFrame
        The same DataFrame with new columns:
        ``parsed_genres``, ``parsed_keywords``, ``parsed_cast``, ``tags``.
    """
    df = df.copy()

    logger.info("Parsing genres ...")
    df["parsed_genres"] = df["genres"].apply(extract_names)

    logger.info("Parsing keywords ...")
    df["parsed_keywords"] = df["keywords"].apply(extract_names)

    logger.info("Parsing cast ...")
    df["parsed_cast"] = df["cast"].apply(extract_cast_names)

    # Fill missing overview with empty string
    df["overview"] = df["overview"].fillna("")

    # Combine into a single tags column
    logger.info("Creating unified 'tags' column ...")
    df["tags"] = (
        df["overview"].astype(str) + " " +
        df["parsed_genres"].astype(str) + " " +
        df["parsed_keywords"].astype(str) + " " +
        df["parsed_cast"].astype(str)
    )

    # --- Text normalisation -----------------------------------------------
    df["tags"] = df["tags"].str.lower().str.strip()
    # Collapse multiple whitespace into single space
    df["tags"] = df["tags"].str.replace(r"\s+", " ", regex=True)
    # Final safety net: replace any remaining NaN with empty string
    df["tags"] = df["tags"].fillna("")

    n_empty = (df["tags"].str.len() == 0).sum()
    if n_empty:
        logger.warning("%d movies have empty tags.", n_empty)

    logger.info("Tags created for %d movies.", len(df))
    return df
