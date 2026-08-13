# ==========================================================
# Movie Recommendation System — User-Based Collaborative Filtering
# Author: Kunal Chindarkar
# Dataset: MovieLens 100K (GroupLens Research)
# Stack: Python, Pandas, SQLite (SQL), Scikit-learn (Cosine Similarity), Flask
# ==========================================================
#
# Files required in the same folder:
#   u.data  -> ratings: user_id, item_id, rating, timestamp (tab-separated)
#   u.item  -> movie info: movie_id, title, release_date, ... genre flags (| separated)
#
# Run:
#   python recommendation_system.py --build              -> builds model, prints evaluation + sample recs
#   python recommendation_system.py --recommend 42        -> CLI recommendations for user_id 42 (real titles)
#   python recommendation_system.py --serve                -> starts Flask API on localhost:5000
# ==========================================================

import argparse
import sqlite3
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

DATA_PATH = "u.data"
ITEM_PATH = "u.item"
DB_PATH = "movielens.db"
TOP_N = 5
MIN_RATINGS_PER_MOVIE = 5   # filters out obscure movies with too little signal


# ----------------------------------------------------------
# STEP 1: Load raw MovieLens files
# ----------------------------------------------------------
def load_raw_data():
    ratings = pd.read_csv(DATA_PATH, sep="\t",
                           names=["user_id", "item_id", "rating", "timestamp"])

    movies = pd.read_csv(ITEM_PATH, sep="|", encoding="latin-1", header=None,
                          usecols=[0, 1], names=["item_id", "title"])

    print(f"Loaded {len(ratings)} ratings from {ratings['user_id'].nunique()} users "
          f"on {ratings['item_id'].nunique()} movies")
    return ratings, movies


# ----------------------------------------------------------
# STEP 2: Load into SQLite, clean & aggregate with SQL (CTE + JOIN)
# ----------------------------------------------------------
def clean_with_sql(ratings, db_path=DB_PATH):
    conn = sqlite3.connect(db_path)
    ratings.to_sql("ratings", conn, if_exists="replace", index=False)

    # CTE computes rating count/avg per movie, JOIN filters out low-signal movies
    query = f"""
    WITH movie_stats AS (
        SELECT item_id, COUNT(*) AS n_ratings, AVG(rating) AS avg_rating
        FROM ratings
        GROUP BY item_id
    )
    SELECT r.user_id, r.item_id, r.rating
    FROM ratings r
    JOIN movie_stats s ON r.item_id = s.item_id
    WHERE s.n_ratings >= {MIN_RATINGS_PER_MOVIE}
    """
    clean = pd.read_sql_query(query, conn)
    conn.close()
    print(f"SQL cleaning: {len(ratings)} raw ratings -> {len(clean)} after filtering "
          f"movies with fewer than {MIN_RATINGS_PER_MOVIE} ratings")
    return clean


# ----------------------------------------------------------
# STEP 3: Build user-item matrix + user-user cosine similarity
# ----------------------------------------------------------
def build_similarity_matrix(clean_ratings):
    user_item_matrix = clean_ratings.pivot_table(index="user_id", columns="item_id",
                                                   values="rating", fill_value=0)
    similarity = cosine_similarity(user_item_matrix)
    similarity_df = pd.DataFrame(similarity, index=user_item_matrix.index,
                                  columns=user_item_matrix.index)
    return user_item_matrix, similarity_df


# ----------------------------------------------------------
# STEP 4: Generate top-N recommendations (with real movie titles)
# ----------------------------------------------------------
def recommend_for_user(user_id, user_item_matrix, similarity_df, movies, top_n=TOP_N):
    if user_id not in similarity_df.index:
        return []

    similar_users = similarity_df[user_id].drop(user_id).sort_values(ascending=False)
    top_similar_users = similar_users.head(20).index

    already_rated = set(user_item_matrix.loc[user_id][user_item_matrix.loc[user_id] > 0].index)

    scores = pd.Series(dtype=float)
    for neighbor in top_similar_users:
        weight = similarity_df.loc[user_id, neighbor]
        if weight <= 0:
            continue
        neighbor_ratings = user_item_matrix.loc[neighbor]
        scores = scores.add(neighbor_ratings * weight, fill_value=0)

    scores = scores.drop(labels=[i for i in already_rated if i in scores.index], errors="ignore")
    top_items = scores.sort_values(ascending=False).head(top_n)

    results = []
    for item_id, score in top_items.items():
        title_row = movies[movies["item_id"] == item_id]
        title = title_row["title"].values[0] if len(title_row) else f"Movie #{item_id}"
        results.append({"item_id": int(item_id), "title": title, "predicted_score": round(float(score), 3)})
    return results


# ----------------------------------------------------------
# STEP 5: Evaluation — Precision@K on a held-out test split
# ----------------------------------------------------------
def evaluate_precision_at_k(ratings, movies, k=5, like_threshold=4, n_test_users=100, seed=42):
    """
    For each test user, hide their highest ratings, see how many of our
    top-K recommendations land on movies they actually rated highly (>= like_threshold).
    """
    rng = np.random.default_rng(seed)
    train_rows = []
    test_truth = {}

    for user_id, group in ratings.groupby("user_id"):
        if len(group) < 10:
            train_rows.append(group)
            continue
        liked = group[group["rating"] >= like_threshold]
        if len(liked) == 0:
            train_rows.append(group)
            continue
        # hold out ~20% of this user's liked movies as "ground truth"
        n_hide = max(1, int(len(liked) * 0.2))
        hidden = liked.sample(n=n_hide, random_state=seed)
        test_truth[user_id] = set(hidden["item_id"])
        train_rows.append(group.drop(hidden.index))

    train_ratings = pd.concat(train_rows)
    clean_train = clean_with_sql(train_ratings, db_path="eval_temp.db")
    matrix, sim = build_similarity_matrix(clean_train)

    sample_users = rng.choice(list(test_truth.keys()),
                               size=min(n_test_users, len(test_truth)), replace=False)

    precisions = []
    for uid in sample_users:
        if uid not in matrix.index:
            continue
        recs = recommend_for_user(uid, matrix, sim, movies, top_n=k)
        rec_ids = {r["item_id"] for r in recs}
        hits = len(rec_ids & test_truth[uid])
        precisions.append(hits / k)

    avg_precision = np.mean(precisions) if precisions else 0.0
    print(f"\nPrecision@{k} over {len(precisions)} test users: {avg_precision:.3f}")
    return avg_precision


# ----------------------------------------------------------
# STEP 6: Validate across sample users (sanity check, real titles)
# ----------------------------------------------------------
def validate(user_item_matrix, similarity_df, movies, n_samples=5):
    sample_users = user_item_matrix.index[:n_samples]
    print("\n--- Sample recommendations ---")
    for uid in sample_users:
        recs = recommend_for_user(uid, user_item_matrix, similarity_df, movies)
        titles = [r["title"] for r in recs]
        print(f"User {uid}: {titles}")


# ----------------------------------------------------------
# STEP 7: Flask REST API
# ----------------------------------------------------------
def create_app(user_item_matrix, similarity_df, movies):
    from flask import Flask, jsonify, request

    app = Flask(__name__)

    @app.route("/recommend/<int:user_id>", methods=["GET"])
    def recommend(user_id):
        top_n = int(request.args.get("top_n", TOP_N))
        recs = recommend_for_user(user_id, user_item_matrix, similarity_df, movies, top_n)
        if not recs:
            return jsonify({"user_id": user_id, "error": "user not found"}), 404
        return jsonify({"user_id": user_id, "recommendations": recs})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok", "users": len(user_item_matrix.index),
                         "movies": len(user_item_matrix.columns)})

    return app


# ----------------------------------------------------------
# Main / CLI
# ----------------------------------------------------------
def build_pipeline():
    ratings, movies = load_raw_data()
    clean = clean_with_sql(ratings)
    matrix, sim = build_similarity_matrix(clean)
    validate(matrix, sim, movies)
    evaluate_precision_at_k(ratings, movies, k=TOP_N)
    return matrix, sim, movies


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--recommend", type=int)
    args = parser.parse_args()

    matrix, sim, movies = build_pipeline()

    if args.recommend:
        print(f"\nRecommendations for user {args.recommend}:")
        for r in recommend_for_user(args.recommend, matrix, sim, movies):
            print(f"  {r['title']}  (score: {r['predicted_score']})")
    elif args.serve:
        app = create_app(matrix, sim, movies)
        app.run(debug=True, port=5000)
