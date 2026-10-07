"""
Model training module.

Trains the four auxiliary classifiers described in the project report
(SVM, Decision Tree, KNN, Gradient Boosting) and provides evaluation
utilities.

IMPORTANT TECHNICAL NOTE
-------------------------
The original project treats ``movie_id`` as the supervised target for
the auxiliary classifiers.  This is a project-specific formulation and
should NOT be interpreted as conventional recommendation-system ground
truth.  Movie IDs are arbitrary identifiers, not meaningful class
labels.  The classifiers are trained here to reproduce the original
report's methodology; the actual recommendation logic relies on cosine
similarity + KNN neighbours combined via Counter-based voting
(see ``recommender.py``).
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score

logger = logging.getLogger("movie_recommender")


def prepare_train_test(
    tfidf_matrix,
    labels,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Split the TF-IDF matrix and target labels into train / test sets.

    Uses 80 / 20 split as described in the original report.

    Parameters
    ----------
    tfidf_matrix : sparse matrix
        TF-IDF features.
    labels : array-like
        Target labels (movie_id in the original formulation).
    test_size : float
        Fraction reserved for testing.
    random_state : int
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    # NOTE: The original project treats movie_id as the supervised
    # target for the auxiliary classifiers.  This is a project-specific
    # formulation and should not be interpreted as conventional
    # recommendation-system ground truth.
    X_train, X_test, y_train, y_test = train_test_split(
        tfidf_matrix, labels, test_size=test_size, random_state=random_state
    )
    logger.info(
        "Train/test split: train=%d, test=%d", X_train.shape[0], X_test.shape[0]
    )
    return X_train, X_test, y_train, y_test


def train_svm(X_train, y_train) -> SVC:
    """Train a Support Vector Machine classifier."""
    logger.info("Training SVM (kernel=linear) ...")
    model = SVC(kernel="linear", probability=True)
    model.fit(X_train, y_train)
    logger.info("SVM training complete.")
    return model


def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """Train a Decision Tree classifier."""
    logger.info("Training Decision Tree ...")
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)
    logger.info("Decision Tree training complete.")
    return model


def train_knn(X_train, y_train, n_neighbors: int = 5) -> KNeighborsClassifier:
    """Train a K-Nearest Neighbors classifier."""
    # Ensure n_neighbors does not exceed training samples
    actual_k = min(n_neighbors, X_train.shape[0])
    if actual_k < n_neighbors:
        logger.warning(
            "Reducing KNN n_neighbors from %d to %d (training set size).",
            n_neighbors, actual_k,
        )
    logger.info("Training KNN (n_neighbors=%d) ...", actual_k)
    model = KNeighborsClassifier(n_neighbors=actual_k)
    model.fit(X_train, y_train)
    logger.info("KNN training complete.")
    return model


def train_gradient_boosting(
    X_train, y_train, n_estimators: int = 100
) -> GradientBoostingClassifier:
    """Train a Gradient Boosting classifier."""
    logger.info("Training Gradient Boosting (n_estimators=%d) ...", n_estimators)
    model = GradientBoostingClassifier(
        n_estimators=n_estimators, random_state=42
    )
    model.fit(X_train, y_train)
    logger.info("Gradient Boosting training complete.")
    return model


def evaluate_model(
    model, X_test, y_test, model_name: str = "Model"
) -> Optional[float]:
    """
    Evaluate a classifier using accuracy score.

    Returns the accuracy or ``None`` if evaluation is not meaningful.
    """
    try:
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"  {model_name:25s} accuracy: {acc:.4f}")
        logger.info("%s accuracy: %.4f", model_name, acc)
        return acc
    except Exception as exc:
        logger.warning("Could not evaluate %s: %s", model_name, exc)
        print(f"  {model_name:25s} accuracy: N/A ({exc})")
        return None


def train_all_models(
    tfidf_matrix, labels
) -> Dict[str, Any]:
    """
    Run the full training pipeline for all four classifiers.

    Returns a dict mapping model name → fitted model.
    """
    X_train, X_test, y_train, y_test = prepare_train_test(
        tfidf_matrix, labels
    )

    # ---- Train models ----------------------------------------------------
    svm_model = train_svm(X_train, y_train)
    dt_model = train_decision_tree(X_train, y_train)
    knn_model = train_knn(X_train, y_train, n_neighbors=5)
    gb_model = train_gradient_boosting(X_train, y_train, n_estimators=100)

    # ---- Evaluate --------------------------------------------------------
    print(f"\n{'='*50}")
    print("MODEL EVALUATION  (auxiliary classifiers)")
    print("NOTE: movie_id is the target — see technical note.")
    print(f"{'='*50}")

    evaluate_model(svm_model, X_test, y_test, "SVM (linear)")
    evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    evaluate_model(knn_model, X_test, y_test, "KNN (k=5)")
    evaluate_model(gb_model, X_test, y_test, "Gradient Boosting")

    print(f"{'='*50}\n")

    return {
        "svm": svm_model,
        "decision_tree": dt_model,
        "knn": knn_model,
        "gradient_boosting": gb_model,
    }
