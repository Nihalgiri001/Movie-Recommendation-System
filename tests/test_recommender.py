"""
Unit tests for the recommender module.

Uses small synthetic fixtures so the full TMDB dataset is NOT required.
"""

import unittest
import sys
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.recommender import (
    find_movie_index,
    suggest_titles,
    _cosine_recommendations,
    _knn_recommendations,
    get_ensemble_recommendations,
)
from src.feature_engineering import compute_cosine_similarity


class TestFindMovieIndex(unittest.TestCase):
    """Tests for movie-title lookup."""

    def setUp(self):
        self.movies = pd.DataFrame({
            "title": ["Batman", "Superman", "Spider-Man", "Iron Man"],
        })

    def test_exact_match(self):
        self.assertEqual(find_movie_index("Batman", self.movies), 0)

    def test_case_insensitive(self):
        self.assertEqual(find_movie_index("batman", self.movies), 0)
        self.assertEqual(find_movie_index("BATMAN", self.movies), 0)
        self.assertEqual(find_movie_index("BaTmAn", self.movies), 0)

    def test_not_found(self):
        self.assertIsNone(find_movie_index("Nonexistent", self.movies))


class TestSuggestTitles(unittest.TestCase):
    """Tests for fuzzy title suggestions."""

    def setUp(self):
        self.movies = pd.DataFrame({
            "title": ["Batman", "Batman Begins", "Batman Returns",
                       "Superman", "Spider-Man"],
        })

    def test_suggestions_returned(self):
        suggestions = suggest_titles("Batmn", self.movies)
        self.assertGreater(len(suggestions), 0)
        # Should suggest at least "Batman"
        lower_suggestions = [s.lower() for s in suggestions]
        self.assertIn("batman", lower_suggestions)

    def test_no_crash_on_unknown(self):
        suggestions = suggest_titles("zzzzzzzzz", self.movies)
        self.assertIsInstance(suggestions, list)


class TestCosineRecommendations(unittest.TestCase):
    """Tests for cosine-similarity based recommendations."""

    def setUp(self):
        # Create a small synthetic TF-IDF matrix
        tags = [
            "action hero superhero",
            "action hero villain",
            "romance love story",
            "action hero cape",
            "romance love dance",
        ]
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(tags)
        self.cosine_sim = compute_cosine_similarity(tfidf_matrix)

    def test_returns_correct_count(self):
        result = _cosine_recommendations(0, self.cosine_sim, n=2)
        self.assertEqual(len(result), 2)

    def test_does_not_include_self(self):
        result = _cosine_recommendations(0, self.cosine_sim, n=4)
        self.assertNotIn(0, result)


class TestKNNRecommendations(unittest.TestCase):
    """Tests for KNN-based recommendations."""

    def setUp(self):
        tags = [
            "action hero superhero",
            "action hero villain",
            "romance love story",
            "action hero cape",
            "romance love dance",
        ]
        vectorizer = TfidfVectorizer(stop_words="english")
        self.tfidf_matrix = vectorizer.fit_transform(tags)
        # Train a KNN model
        self.knn = KNeighborsClassifier(n_neighbors=3)
        self.knn.fit(self.tfidf_matrix, list(range(5)))

    def test_returns_results(self):
        result = _knn_recommendations(0, self.knn, self.tfidf_matrix, n=2)
        self.assertGreater(len(result), 0)

    def test_does_not_include_self(self):
        result = _knn_recommendations(0, self.knn, self.tfidf_matrix, n=4)
        self.assertNotIn(0, result)


class TestEnsembleRecommendations(unittest.TestCase):
    """Integration test with synthetic data."""

    def _build_artifacts(self):
        """Build minimal artifacts from synthetic data."""
        movies = pd.DataFrame({
            "movie_id": [1, 2, 3, 4, 5],
            "title": ["Batman", "Batman Begins", "Superman", "Iron Man", "Romance Film"],
            "tags": [
                "dark hero gotham batman",
                "dark hero origin batman begins",
                "superman hero krypton",
                "iron man hero stark technology",
                "romance love dance",
            ],
        })
        vectorizer = TfidfVectorizer(stop_words="english")
        tfidf_matrix = vectorizer.fit_transform(movies["tags"])
        cosine_sim = compute_cosine_similarity(tfidf_matrix)

        knn = KNeighborsClassifier(n_neighbors=min(3, len(movies)))
        knn.fit(tfidf_matrix, movies["movie_id"])

        return {
            "movies": movies,
            "tfidf_matrix": tfidf_matrix,
            "cosine_sim": cosine_sim,
            "knn_model": knn,
        }

    def test_batman_returns_results(self):
        artifacts = self._build_artifacts()
        matched, recs = get_ensemble_recommendations(
            "Batman", n_recommendations=3, artifacts=artifacts
        )
        self.assertEqual(matched, "Batman")
        self.assertGreater(len(recs), 0)
        self.assertLessEqual(len(recs), 3)

    def test_input_not_in_recommendations(self):
        artifacts = self._build_artifacts()
        _, recs = get_ensemble_recommendations(
            "Batman", n_recommendations=4, artifacts=artifacts
        )
        self.assertNotIn("Batman", recs)

    def test_invalid_movie(self):
        artifacts = self._build_artifacts()
        _, recs = get_ensemble_recommendations(
            "Nonexistent Movie", n_recommendations=3, artifacts=artifacts
        )
        self.assertTrue(recs[0].startswith("Movie '"))

    def test_case_insensitive_query(self):
        artifacts = self._build_artifacts()
        matched, recs = get_ensemble_recommendations(
            "batman", n_recommendations=3, artifacts=artifacts
        )
        self.assertEqual(matched, "Batman")


class TestCounterVoting(unittest.TestCase):
    """Verify that Counter-based voting works correctly."""

    def test_counter_most_common(self):
        indices = [1, 2, 3, 1, 2, 1]
        counter = Counter(indices)
        top2 = [i for i, _ in counter.most_common(2)]
        self.assertEqual(top2, [1, 2])


if __name__ == "__main__":
    unittest.main()
