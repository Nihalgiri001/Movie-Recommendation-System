#!/usr/bin/env python3
"""
train.py — Complete training pipeline for the TMDB Movie Recommender.

Usage
-----
    python train.py

This script:
  1. Loads and merges the TMDB 5000 Movies and Credits datasets.
  2. Parses metadata (genres, keywords, cast).
  3. Creates a unified ``tags`` column.
  4. Builds a TF-IDF matrix from the tags.
  5. Trains SVM, Decision Tree, KNN and Gradient Boosting classifiers.
  6. Computes the cosine similarity matrix.
  7. Saves all artifacts to the ``models/`` directory.
"""

import sys
import time
from pathlib import Path

import joblib

# Ensure the project root is on sys.path so ``src`` can be imported
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import (
    MODEL_DIR,
    TFIDF_VECTORIZER_PATH,
    COSINE_SIM_PATH,
    KNN_MODEL_PATH,
    SVM_MODEL_PATH,
    DECISION_TREE_PATH,
    GRADIENT_BOOSTING_PATH,
    PROCESSED_MOVIES_PATH,
    TFIDF_MATRIX_PATH,
    setup_logging,
    ensure_directory,
)
from src.data_loader import load_and_merge
from src.preprocessing import create_tags
from src.feature_engineering import build_tfidf_matrix, compute_cosine_similarity
from src.model_training import train_all_models


def main() -> None:
    logger = setup_logging()

    print("=" * 50)
    print("TMDB MOVIE RECOMMENDER — TRAINING PIPELINE")
    print("=" * 50)

    start = time.time()

    # ------------------------------------------------------------------
    # 1. Load & merge datasets
    # ------------------------------------------------------------------
    print("\n[1/7] Loading and merging datasets ...")
    df = load_and_merge()

    # ------------------------------------------------------------------
    # 2. Parse metadata and create tags
    # ------------------------------------------------------------------
    print("[2/7] Parsing metadata and creating tags ...")
    df = create_tags(df)

    # ------------------------------------------------------------------
    # 3. Build TF-IDF matrix
    # ------------------------------------------------------------------
    print("[3/7] Building TF-IDF matrix ...")
    tfidf_vectorizer, tfidf_matrix = build_tfidf_matrix(df["tags"])

    # ------------------------------------------------------------------
    # 4. Train auxiliary ML models
    # ------------------------------------------------------------------
    print("[4/7] Training ML models ...")
    labels = df["movie_id"]
    models = train_all_models(tfidf_matrix, labels)

    # ------------------------------------------------------------------
    # 5. Compute cosine similarity
    # ------------------------------------------------------------------
    print("[5/7] Computing cosine similarity matrix ...")
    cosine_sim = compute_cosine_similarity(tfidf_matrix)

    # ------------------------------------------------------------------
    # 6. Save all artifacts
    # ------------------------------------------------------------------
    print("[6/7] Saving model artifacts ...")
    ensure_directory(MODEL_DIR)

    # Prepare a slim DataFrame with just the columns needed at inference
    movies_slim = df[["movie_id", "title", "tags"]].reset_index(drop=True)

    artifact_map = {
        PROCESSED_MOVIES_PATH: ("processed_movies.pkl", movies_slim),
        TFIDF_VECTORIZER_PATH: ("tfidf_vectorizer.pkl", tfidf_vectorizer),
        TFIDF_MATRIX_PATH: ("tfidf_matrix.pkl", tfidf_matrix),
        COSINE_SIM_PATH: ("cosine_similarity.pkl", cosine_sim),
        KNN_MODEL_PATH: ("knn_model.pkl", models["knn"]),
        SVM_MODEL_PATH: ("svm_model.pkl", models["svm"]),
        DECISION_TREE_PATH: ("decision_tree_model.pkl", models["decision_tree"]),
        GRADIENT_BOOSTING_PATH: (
            "gradient_boosting_model.pkl",
            models["gradient_boosting"],
        ),
    }

    for path, (name, obj) in artifact_map.items():
        joblib.dump(obj, path)
        logger.info("Saved %s", path)

    elapsed = time.time() - start

    # ------------------------------------------------------------------
    # 7. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"\n  Movies processed : {len(df)}")
    print(f"  TF-IDF features  : {tfidf_matrix.shape[1]}")
    print(f"  Training time    : {elapsed:.1f}s")
    print(f"\n  Saved artifacts:")
    for _, (name, _) in artifact_map.items():
        print(f"    - {name}")
    print(f"\nYou can now run:\n")
    print(f'  python recommend.py --movie "Batman"')
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
