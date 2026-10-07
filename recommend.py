#!/usr/bin/env python3
"""
recommend.py — CLI for the TMDB Movie Recommender.

Usage
-----
Interactive mode:
    python recommend.py

Command-line mode:
    python recommend.py --movie "Batman"
    python recommend.py --movie "Batman" --top 10
    python recommend.py --movie "Batman" --mode experimental
"""

import argparse
import sys
from pathlib import Path

# Ensure the project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import setup_logging
from src.recommender import get_ensemble_recommendations, load_recommendation_artifacts


def print_recommendations(matched_title: str, recommendations: list) -> None:
    """Pretty-print the recommendation results."""
    # Check if the first item is an error message
    if recommendations and recommendations[0].startswith("Movie '"):
        print(f"\n{recommendations[0]}")
        return

    print(f"\n{'='*50}")
    print(f"Recommended Movies similar to '{matched_title}':")
    print(f"{'='*50}\n")
    for i, title in enumerate(recommendations, 1):
        print(f"  {i:2d}. {title}")
    print()


def interactive_mode(artifacts: dict) -> None:
    """Run the recommender in interactive mode."""
    print(f"\n{'='*50}")
    print("  TMDB MOVIE RECOMMENDATION SYSTEM")
    print(f"{'='*50}\n")
    print("Type a movie title and press Enter.")
    print("Type 'quit' or 'exit' to stop.\n")

    while True:
        try:
            movie = input("Enter a movie title: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not movie:
            continue
        if movie.lower() in ("quit", "exit", "q"):
            print("Goodbye!")
            break

        matched, recs = get_ensemble_recommendations(
            movie, n_recommendations=10, artifacts=artifacts
        )
        print_recommendations(matched, recs)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TMDB Movie Recommendation System"
    )
    parser.add_argument(
        "--movie", "-m",
        type=str,
        default=None,
        help="Movie title to get recommendations for.",
    )
    parser.add_argument(
        "--top", "-n",
        type=int,
        default=10,
        help="Number of recommendations to return (default: 10).",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["original", "experimental"],
        default="original",
        help="Recommendation mode (default: original).",
    )
    args = parser.parse_args()

    setup_logging()

    # Load artifacts once
    artifacts = load_recommendation_artifacts()

    if args.movie:
        # Command-line mode
        matched, recs = get_ensemble_recommendations(
            args.movie, n_recommendations=args.top, artifacts=artifacts
        )
        print_recommendations(matched, recs)
    else:
        # Interactive mode
        interactive_mode(artifacts)


if __name__ == "__main__":
    main()
