# Implementation of Machine Learning Ensemble Model for Movie Recommendation

A content-based movie recommendation system built using the **TMDB 5000 Movies Dataset**. The system combines **TF-IDF vectorization**, **cosine similarity**, **K-Nearest Neighbors**, and **Counter-based ensemble voting** to recommend movies similar to a user-provided title.

> **Note:** This is a content-based recommender system. It does NOT use collaborative filtering, user-item interaction data, or deep learning.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Problem Statement](#problem-statement)
3. [Objectives](#objectives)
4. [Dataset](#dataset)
5. [Features Used](#features-used)
6. [Methodology](#methodology)
7. [Architecture / Workflow](#architecture--workflow)
8. [Technologies Used](#technologies-used)
9. [Installation](#installation)
10. [Dataset Setup](#dataset-setup)
11. [Training Instructions](#training-instructions)
12. [Recommendation Instructions](#recommendation-instructions)
13. [Example Usage](#example-usage)
14. [Expected Output](#expected-output)
15. [Project Structure](#project-structure)
16. [Explanation of TF-IDF](#explanation-of-tf-idf)
17. [Explanation of Cosine Similarity](#explanation-of-cosine-similarity)
18. [Explanation of KNN](#explanation-of-knn)
19. [Explanation of Ensemble Voting](#explanation-of-ensemble-voting)
20. [Limitations](#limitations)
21. [Future Enhancements](#future-enhancements)

---

## Project Overview

This project implements a movie recommendation engine that suggests movies based on content similarity. Given a movie title (e.g., "Batman"), the system analyses the movie's textual metadata — its overview, genres, keywords, and cast — and finds other movies with the most similar content profiles.

The recommendation pipeline:
1. Represents each movie as a combined text ("tags") from its metadata.
2. Converts tags into numerical TF-IDF vectors.
3. Uses cosine similarity and KNN to find related movies.
4. Combines both recommendation lists via Counter-based voting.

Four auxiliary machine-learning classifiers (SVM, Decision Tree, KNN, Gradient Boosting) are trained as part of the methodology to learn relationships in the vectorised movie representations.

---

## Problem Statement

With thousands of movies available, finding relevant content is challenging. This project addresses the problem by building an automated recommendation system that analyses movie metadata and suggests similar films, removing the need for manual browsing.

---

## Objectives

- Load and clean the TMDB 5000 Movies Dataset.
- Extract and combine relevant movie metadata into unified textual tags.
- Convert text into numerical features using TF-IDF.
- Calculate movie-to-movie similarity using cosine similarity.
- Train SVM, Decision Tree, KNN, and Gradient Boosting classifiers.
- Generate recommendations using cosine similarity + KNN + ensemble voting.
- Provide a simple CLI for end users.

---

## Dataset

**TMDB 5000 Movies Dataset** consisting of two files:

| File | Description |
|------|-------------|
| `tmdb_5000_movies.csv` | Movie metadata (title, overview, genres, keywords, etc.) |
| `tmdb_5000_credits.csv` | Cast and crew information |

The dataset contains approximately **4,803 movies** after merging.

---

## Features Used

| Feature | Source | Description |
|---------|--------|-------------|
| `overview` | Movies CSV | Plot summary of the movie |
| `genres` | Movies CSV | List of genre labels |
| `keywords` | Movies CSV | Thematic keywords |
| `cast` | Credits CSV | Top 5 actors |

These four features are combined into a single `tags` column for each movie.

---

## Methodology

```
Input Movie Title
       ↓
Find Movie Index
       ↓
   ┌───┴───┐
   ↓       ↓
Cosine    KNN
Similarity Neighbours
   ↓       ↓
   └───┬───┘
       ↓
Counter Voting
       ↓
Top N Recommendations
```

1. **Tag Creation**: Combine overview + genres + keywords + cast into a single text string per movie.
2. **TF-IDF Vectorization**: Convert tags into sparse numerical vectors using `TfidfVectorizer(stop_words="english")`.
3. **Cosine Similarity**: Compute pairwise similarity between all movie vectors.
4. **KNN**: Find nearest neighbours in the TF-IDF feature space.
5. **Ensemble Voting**: Merge cosine and KNN recommendation lists, count occurrences with `Counter`, and return the most frequently recommended movies.

---

## Architecture / Workflow

```
TMDB Movies CSV  +  TMDB Credits CSV
              ↓
        Data Cleaning
              ↓
        Dataset Merge
              ↓
       Metadata Parsing
              ↓
     Feature Engineering
              ↓
       Combined `tags`
              ↓
      Text Normalization
              ↓
     TF-IDF Vectorization
              ↓
        TF-IDF Matrix
        ├──────────────┐
        ↓              ↓
 Cosine Similarity   ML Models
                     ├── SVM
                     ├── Decision Tree
                     ├── KNN
                     └── Gradient Boosting
        ↓
Cosine Recommendations
        +
 KNN Recommendations
        ↓
  Counter-Based Voting
        ↓
    Top N Movies
        ↓
       User
```

---

## Technologies Used

| Technology | Purpose |
|-----------|---------|
| Python 3.8+ | Programming language |
| pandas | Data manipulation |
| NumPy | Numerical computing |
| scikit-learn | ML models, TF-IDF, cosine similarity |
| SciPy | Sparse matrix operations |
| joblib | Model persistence |
| collections.Counter | Ensemble voting |
| ast.literal_eval | Safe JSON-like string parsing |
| argparse | CLI interface |
| pathlib | Cross-platform path handling |

---

## Installation

```bash
# Clone or download the project
cd movie-recommender

# (Optional) Create a virtual environment
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
# venv\Scripts\activate    # Windows

# Install dependencies
pip install -r requirements.txt
```

---

## Dataset Setup

1. Download the **TMDB 5000 Movies Dataset**.
2. Place the two CSV files in the `data/` directory:

```
movie-recommender/
  data/
    tmdb_5000_movies.csv
    tmdb_5000_credits.csv
```

> The application will display a clear error if the files are missing.

---

## Training Instructions

```bash
python3 train.py
```

This runs the complete pipeline:
1. Loads and merges datasets.
2. Parses metadata and creates tags.
3. Builds the TF-IDF matrix.
4. Trains SVM, Decision Tree, KNN, and Gradient Boosting.
5. Computes cosine similarity.
6. Saves all model artifacts to `models/`.

Training takes approximately 5–15 minutes depending on hardware.

---

## Recommendation Instructions

```bash
# Command-line mode
python3 recommend.py --movie "Batman"
python3 recommend.py --movie "Batman" --top 10

# Interactive mode
python3 recommend.py
```

> `recommend.py` loads pre-trained artifacts — it does NOT retrain models.

---

## Example Usage

```bash
$ python3 recommend.py --movie "Batman"

==================================================
Recommended Movies similar to 'Batman':
==================================================

   1. Batman Returns
   2. Batman Forever
   3. Batman & Robin
   4. Batman Begins
   5. The Dark Knight
   6. The Dark Knight Rises
   7. Superman
   8. Man of Steel
   9. Superman III
  10. Superman IV: The Quest for Peace
```

*(Actual results depend on the dataset and TF-IDF representation.)*

---

## Expected Output

For the query **"Batman"**, the system recommends movies with similar content profiles — primarily other Batman films, DC superhero films, and thematically related titles.

---

## Project Structure

```
movie-recommender/
│
├── data/                          # Dataset files (not committed)
│   ├── tmdb_5000_movies.csv
│   └── tmdb_5000_credits.csv
│
├── models/                        # Trained model artifacts
│   ├── tfidf_vectorizer.pkl
│   ├── cosine_similarity.pkl
│   ├── tfidf_matrix.pkl
│   ├── knn_model.pkl
│   ├── svm_model.pkl
│   ├── decision_tree_model.pkl
│   ├── gradient_boosting_model.pkl
│   └── processed_movies.pkl
│
├── src/                           # Source modules
│   ├── __init__.py
│   ├── data_loader.py             # Dataset loading and merging
│   ├── preprocessing.py           # Metadata parsing and tag creation
│   ├── feature_engineering.py     # TF-IDF and cosine similarity
│   ├── model_training.py          # ML model training and evaluation
│   ├── recommender.py             # Recommendation logic
│   └── utils.py                   # Configuration and utilities
│
├── tests/                         # Unit tests
│   ├── test_preprocessing.py
│   └── test_recommender.py
│
├── notebooks/                     # Jupyter notebook demo
│   └── movie_recommender_demo.ipynb
│
├── outputs/                       # Output directory
│   └── .gitkeep
│
├── train.py                       # Training pipeline
├── recommend.py                   # CLI recommendation script
├── requirements.txt               # Python dependencies
├── README.md                      # This file
└── .gitignore                     # Git ignore rules
```

---

## Explanation of TF-IDF

**Term Frequency–Inverse Document Frequency (TF-IDF)** converts text into numerical vectors that capture word importance.

- **TF (Term Frequency):** How often a word appears in a document (movie tags).
- **IDF (Inverse Document Frequency):** Down-weights words that appear in many documents (common words contribute less).
- **TF-IDF = TF × IDF:** Words that are frequent in a specific movie but rare overall get high scores.

English stop words (the, is, and, etc.) are automatically removed.

---

## Explanation of Cosine Similarity

**Cosine similarity** measures the angle between two TF-IDF vectors:

```
cosine_similarity(A, B) = (A · B) / (||A|| × ||B||)
```

- **1.0** → Identical content profiles.
- **0.0** → No overlap.

The system computes a full movie-to-movie similarity matrix during training and retrieves pre-computed scores during recommendation.

---

## Explanation of KNN

**K-Nearest Neighbors** finds the K movies closest to the input movie in the TF-IDF feature space.

- Uses `KNeighborsClassifier(n_neighbors=5)` from scikit-learn.
- At recommendation time, the `kneighbors()` method returns the indices of the nearest movies.
- KNN captures neighbourhood-based similarity that may differ slightly from cosine similarity rankings.

---

## Explanation of Ensemble Voting

The system combines cosine similarity and KNN recommendations using **Counter-based voting**:

1. Get top-N candidates from cosine similarity.
2. Get top-N candidates from KNN.
3. Merge both lists.
4. Count how often each movie appears using `collections.Counter`.
5. Return movies with the highest vote counts.

Movies recommended by **both** methods are ranked higher, creating a more robust recommendation.

> **Note:** The SVM, Decision Tree, and Gradient Boosting classifiers are trained as part of the project methodology but are not used in the default recommendation function. The actual ensemble combines only cosine similarity and KNN via Counter voting.

---

## Limitations

- **Content-based only:** Recommendations are based on movie metadata, not user preferences or watch history.
- **No collaborative filtering:** The system cannot capture "users who liked X also liked Y" patterns.
- **Auxiliary classifiers:** SVM, Decision Tree, and Gradient Boosting use `movie_id` as a target label, which is a project-specific formulation (movie IDs are identifiers, not meaningful classes).
- **Cold start:** New movies without metadata cannot be recommended.
- **English only:** TF-IDF uses English stop words; non-English content may not be handled optimally.
- **Static dataset:** Recommendations don't update unless the dataset is refreshed and models are retrained.

---

## Future Enhancements

- **Collaborative filtering:** Incorporate user ratings for hybrid recommendations.
- **Deep learning:** Use word embeddings (Word2Vec, BERT) for richer text representations.
- **Web interface:** Build a Flask/Streamlit UI for interactive exploration.
- **Real-time updates:** Connect to the TMDB API for live movie data.
- **Crew information:** Include director and writer metadata in tags.
- **Poster display:** Show movie posters alongside recommendations.
- **Full ensemble:** Use predictions from all four classifiers in the voting mechanism.
- **Evaluation metrics:** Implement precision@K, recall@K, and NDCG for recommendation quality.

---

## License

This project is for educational purposes as part of a B.Tech Computer Science project.
