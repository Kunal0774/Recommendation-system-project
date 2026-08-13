# Movie Recommendation System — Collaborative Filtering (MovieLens 100K)

A user-based collaborative filtering movie recommender built with Python, SQL, and Flask, trained and evaluated on the real MovieLens 100K dataset from GroupLens Research.

## Problem Statement
Recommend movies to a user based on the rating patterns of other users with similar taste — the same core idea behind "recommended for you" on Netflix, Amazon Prime Video, etc.

## Dataset
[MovieLens 100K](https://grouplens.org/datasets/movielens/) — 100,000 ratings from 943 users on 1,682 movies (1–5 star scale).
- `u.data` — ratings (user_id, item_id, rating, timestamp)
- `u.item` — movie titles

## Tech Stack
- **Python** — Pandas, NumPy
- **SQL (SQLite)** — CTE + JOIN for data cleaning
- **Scikit-learn** — Cosine Similarity
- **Flask** — REST API for real-time recommendations

## How It Works
1. **Load** — Raw ratings and movie titles are read from the MovieLens files
2. **SQL Cleaning** — A CTE computes each movie's rating count and average; a JOIN filters out movies with fewer than 5 ratings, so low-signal/obscure titles don't distort similarity
3. **User-Item Matrix** — Cleaned ratings are pivoted into a matrix of users × movies × ratings
4. **Similarity Computation** — Cosine similarity is computed between every pair of users, based on the angle between their rating vectors (so it's not thrown off by users who rate everything higher or lower on average)
5. **Recommendation Generation** — For a target user, the 20 most similar users are found; their ratings on movies the target user hasn't seen are combined, weighted by similarity, into a predicted relevance score; the top 5 unseen movies are returned with real titles
6. **Evaluation** — Precision@5 is measured by hiding a portion of each test user's known "liked" movies (rating ≥ 4) and checking how many of the top-5 recommendations correctly land on hidden movies
7. **API Layer** — A Flask endpoint serves live recommendations as JSON

## Run It

```bash
pip install pandas numpy scikit-learn flask

# Build the model, run validation + precision@5 evaluation
python recommendation_system.py --build

# Get recommendations for a specific real user
python recommendation_system.py --recommend 196

# Start the REST API
python recommendation_system.py --serve
```

### API Usage
```
GET /recommend/<user_id>?top_n=5
```
Example response:
```json
{
  "user_id": 196,
  "recommendations": [
    {"item_id": 100, "title": "Fargo (1996)", "predicted_score": 16.908},
    {"item_id": 204, "title": "Back to the Future (1985)", "predicted_score": 14.282}
  ]
}
```

## Results / Findings
- Trained on **99,287 ratings** (after SQL filtering removed movies with under 5 ratings) across 943 users and ~1,650 movies
- **Precision@5 ≈ 0.33** on a 100-user held-out test split — meaning roughly 1 in 3 of the top-5 recommended movies were movies the user genuinely rated highly but which were hidden from training. This is a solid, honest baseline for a simple user-based CF model on this dataset size
- Recommendations were clearly personalized — e.g. one test user's top recommendations were dominated by classic sci-fi/action (Terminator, Jurassic Park), another's by 90s dramas (English Patient, L.A. Confidential) — confirming the model is capturing distinct taste profiles, not just recommending globally popular movies to everyone
- The `predicted_score` is a similarity-weighted relevance score used for **ranking**, not a literal predicted star rating

## Limitations & Future Improvements
- **Cold-start problem**: a brand-new user with no ratings gets no recommendations — could be handled by falling back to "most popular" or genre-based recommendations
- Only user-based CF is implemented; item-based CF or a hybrid (content + collaborative) model using the genre flags in `u.item` could be compared against this baseline
- Precision@5 could be improved with matrix factorization (SVD) instead of raw cosine similarity — a natural next step

## Files in This Repo
- `recommendation_system.py` — full pipeline: data loading, SQL cleaning, similarity model, evaluation, Flask API
- `u.data` — MovieLens ratings data
- `u.item` — MovieLens movie titles
- `README.md` — this file

## Author
Kunal Chindarkar
