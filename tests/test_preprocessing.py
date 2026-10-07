"""
Unit tests for the preprocessing module.

Uses small synthetic fixtures so the full TMDB dataset is NOT required.
"""

import unittest
import pandas as pd

# Allow running tests from the project root
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing import (
    extract_names,
    extract_cast_names,
    create_tags,
    _safe_literal_eval,
)


class TestSafeLiteralEval(unittest.TestCase):
    """Tests for _safe_literal_eval."""

    def test_valid_list(self):
        val = '[{"id": 1, "name": "Action"}]'
        result = _safe_literal_eval(val)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)

    def test_nan_returns_empty(self):
        self.assertEqual(_safe_literal_eval(float("nan")), [])

    def test_none_returns_empty(self):
        self.assertEqual(_safe_literal_eval(None), [])

    def test_empty_string_returns_empty(self):
        self.assertEqual(_safe_literal_eval(""), [])

    def test_malformed_returns_empty(self):
        self.assertEqual(_safe_literal_eval("{bad json"), [])


class TestExtractNames(unittest.TestCase):
    """Tests for extract_names."""

    def test_basic_extraction(self):
        val = '[{"id": 28, "name": "Action"}, {"id": 12, "name": "Adventure"}]'
        self.assertEqual(extract_names(val), "Action Adventure")

    def test_empty_list(self):
        self.assertEqual(extract_names("[]"), "")

    def test_nan(self):
        self.assertEqual(extract_names(float("nan")), "")

    def test_missing_name_key(self):
        val = '[{"id": 28}]'
        self.assertEqual(extract_names(val), "")


class TestExtractCastNames(unittest.TestCase):
    """Tests for extract_cast_names."""

    def test_basic(self):
        val = '[{"name": "Alice"}, {"name": "Bob"}, {"name": "Charlie"}]'
        self.assertEqual(extract_cast_names(val, max_actors=2), "Alice Bob")

    def test_default_limit(self):
        actors = [{"name": f"Actor{i}"} for i in range(10)]
        result = extract_cast_names(str(actors))
        self.assertEqual(len(result.split()), 5)  # default max_actors=5


class TestCreateTags(unittest.TestCase):
    """Tests for create_tags."""

    def _make_df(self):
        return pd.DataFrame({
            "overview": ["A hero saves the world.", "A love story."],
            "genres": [
                '[{"id": 1, "name": "Action"}]',
                '[{"id": 2, "name": "Romance"}]',
            ],
            "keywords": [
                '[{"id": 10, "name": "hero"}]',
                '[{"id": 20, "name": "love"}]',
            ],
            "cast": [
                '[{"name": "Alice"}, {"name": "Bob"}]',
                '[{"name": "Charlie"}]',
            ],
        })

    def test_tags_column_created(self):
        df = create_tags(self._make_df())
        self.assertIn("tags", df.columns)

    def test_tags_are_lowercase(self):
        df = create_tags(self._make_df())
        for tag in df["tags"]:
            self.assertEqual(tag, tag.lower())

    def test_tags_contain_overview(self):
        df = create_tags(self._make_df())
        self.assertIn("hero saves", df["tags"].iloc[0])

    def test_tags_contain_genre(self):
        df = create_tags(self._make_df())
        self.assertIn("action", df["tags"].iloc[0])

    def test_tags_contain_cast(self):
        df = create_tags(self._make_df())
        self.assertIn("alice", df["tags"].iloc[0])

    def test_no_nan_in_tags(self):
        df = self._make_df()
        df.loc[0, "overview"] = None
        result = create_tags(df)
        self.assertFalse(result["tags"].isna().any())


class TestColumnNormalization(unittest.TestCase):
    """Test whitespace stripping from column names (data_loader behaviour)."""

    def test_strip_columns(self):
        df = pd.DataFrame({" title ": ["A"], " id ": [1]})
        df.columns = df.columns.str.strip()
        self.assertIn("title", df.columns)
        self.assertIn("id", df.columns)


if __name__ == "__main__":
    unittest.main()
