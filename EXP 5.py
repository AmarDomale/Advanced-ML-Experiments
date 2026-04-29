# ==========================================
# RECOMMENDATION SYSTEM USING
# 1. CONTENT-BASED FILTERING
# 2. COLLABORATIVE FILTERING
# Reference Concept:
# https://github.com/Elmar999/Recommendation-systems
# ==========================================

# -----------------------------
# 1. IMPORT LIBRARIES
# -----------------------------
import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix

# -----------------------------
# 2. LOAD DATASET
# -----------------------------
# Make sure movies.csv and ratings.csv are in the same folder as this notebook

movies = pd.read_csv("AML\\ml-latest-small\\movies.csv")     # Columns: movieId, title, genres
ratings = pd.read_csv("AML\\ml-latest-small\\ratings.csv")   # Columns: userId, movieId, rating, timestamp

print("Movies Dataset Shape:", movies.shape)
print("Ratings Dataset Shape:", ratings.shape)

print("\nSample Movies Data:")
print(movies.head())

print("\nSample Ratings Data:")
print(ratings.head())

# =========================================================
# PART A: CONTENT-BASED FILTERING
# =========================================================

print("\n" + "="*60)
print("PART A: CONTENT-BASED FILTERING")
print("="*60)

# -----------------------------
# 3. PREPROCESS GENRES
# -----------------------------
# Replace | with space for vectorization
movies["genres_clean"] = movies["genres"].str.replace("|", " ", regex=False)

# -----------------------------
# 4. CONVERT GENRES INTO FEATURES
# -----------------------------
vectorizer = CountVectorizer(token_pattern=r"[^ ]+")
genre_matrix = vectorizer.fit_transform(movies["genres_clean"])

print("\nGenre Matrix Shape:", genre_matrix.shape)

# -----------------------------
# 5. COMPUTE COSINE SIMILARITY
# -----------------------------
cosine_sim = cosine_similarity(genre_matrix)

print("Cosine Similarity Matrix Shape:", cosine_sim.shape)

# -----------------------------
# 6. CREATE TITLE INDEX MAPPING
# -----------------------------
title_to_index = pd.Series(movies.index, index=movies["title"]).to_dict()

# -----------------------------
# 7. CONTENT-BASED RECOMMENDATION FUNCTION
# -----------------------------
def recommend_similar_movies(movie_title, top_n=5):
    if movie_title not in title_to_index:
        return f"Movie '{movie_title}' not found in dataset."

    idx = title_to_index[movie_title]

    # Get similarity scores
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort by similarity descending
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Skip the same movie
    sim_scores = sim_scores[1:top_n+1]

    movie_indices = [i[0] for i in sim_scores]

    return movies[["title", "genres"]].iloc[movie_indices]

# -----------------------------
# 8. KMEANS CLUSTERING (Repo-style concept)
# -----------------------------
# This follows the idea from the referenced repository:
# content filtering using genres + clustering

kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
movies["cluster"] = kmeans.fit_predict(genre_matrix)

print("\nMovie Clusters Assigned Successfully")
print(movies[["title", "genres", "cluster"]].head(10))

# -----------------------------
# 9. MERGE RATINGS WITH MOVIES
# -----------------------------
ratings_movies = ratings.merge(movies, on="movieId")

# -----------------------------
# 10. USER-BASED CONTENT RECOMMENDATION USING CLUSTERS
# -----------------------------
def recommend_by_user_cluster(user_id, top_n=5):
    user_data = ratings_movies[ratings_movies["userId"] == user_id]

    if user_data.empty:
        return f"No ratings found for user {user_id}"

    # Movies rated 4 or above are treated as liked
    liked_movies = user_data[user_data["rating"] >= 4]

    # If no highly rated movies, take top rated ones
    if liked_movies.empty:
        liked_movies = user_data.sort_values(by="rating", ascending=False).head(5)

    # Find favorite clusters
    favorite_clusters = liked_movies["cluster"].value_counts().index.tolist()

    # Movies already watched
    watched_movies = set(user_data["movieId"])

    # Candidate movies from favorite clusters that user has not watched
    candidates = movies[
        (movies["cluster"].isin(favorite_clusters)) &
        (~movies["movieId"].isin(watched_movies))
    ].copy()

    # Rank candidates using average rating and rating count
    movie_stats = ratings.groupby("movieId").agg(
        avg_rating=("rating", "mean"),
        rating_count=("rating", "count")
    ).reset_index()

    candidates = candidates.merge(movie_stats, on="movieId", how="left")
    candidates["avg_rating"] = candidates["avg_rating"].fillna(0)
    candidates["rating_count"] = candidates["rating_count"].fillna(0)

    candidates = candidates.sort_values(
        by=["avg_rating", "rating_count"], ascending=False
    )

    return candidates[["title", "genres", "cluster", "avg_rating", "rating_count"]].head(top_n)

# -----------------------------
# 11. TEST CONTENT-BASED FILTERING
# -----------------------------
print("\n" + "-"*60)
print("Content-Based Recommendation: Similar Movies to 'Toy Story (1995)'")
print("-"*60)
print(recommend_similar_movies("Toy Story (1995)", top_n=5))

print("\n" + "-"*60)
print("Content-Based Recommendation for User 1 (using favorite clusters)")
print("-"*60)
print(recommend_by_user_cluster(user_id=1, top_n=5))

# =========================================================
# PART B: COLLABORATIVE FILTERING
# =========================================================

print("\n" + "="*60)
print("PART B: COLLABORATIVE FILTERING")
print("="*60)

# -----------------------------
# 12. CREATE USER-ITEM MATRIX
# -----------------------------
user_item_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

print("\nUser-Item Matrix Shape:", user_item_matrix.shape)

# -----------------------------
# 13. CONVERT TO SPARSE MATRIX
# -----------------------------
sparse_matrix = csr_matrix(user_item_matrix.values)

# -----------------------------
# 14. APPLY MATRIX FACTORIZATION USING SVD
# -----------------------------
# This is the collaborative filtering part
# It predicts unseen ratings using latent features

svd = TruncatedSVD(n_components=20, random_state=42)

user_factors = svd.fit_transform(sparse_matrix)
item_factors = svd.components_

print("User Factors Shape:", user_factors.shape)
print("Item Factors Shape:", item_factors.shape)

# -----------------------------
# 15. RECONSTRUCT PREDICTED RATINGS
# -----------------------------
predicted_ratings = np.dot(user_factors, item_factors)

print("Predicted Ratings Matrix Shape:", predicted_ratings.shape)

# -----------------------------
# 16. CREATE USER AND MOVIE MAPPINGS
# -----------------------------
user_ids = user_item_matrix.index.tolist()
movie_ids = user_item_matrix.columns.tolist()

user_id_to_index = {uid: idx for idx, uid in enumerate(user_ids)}
movie_id_to_index = {mid: idx for idx, mid in enumerate(movie_ids)}

# -----------------------------
# 17. COLLABORATIVE FILTERING RECOMMENDATION FUNCTION
# -----------------------------
def recommend_collaborative(user_id, top_n=5):
    if user_id not in user_id_to_index:
        return f"User {user_id} not found."

    user_idx = user_id_to_index[user_id]
    user_pred_scores = predicted_ratings[user_idx]

    # Movies already rated by the user
    rated_movies = set(ratings[ratings["userId"] == user_id]["movieId"])

    recommendations = []
    for movie_id, score in zip(movie_ids, user_pred_scores):
        if movie_id not in rated_movies:
            recommendations.append((movie_id, score))

    # Sort by predicted score descending
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)[:top_n]

    rec_movie_ids = [r[0] for r in recommendations]
    rec_scores = [r[1] for r in recommendations]

    result = movies[movies["movieId"].isin(rec_movie_ids)].copy()
    result["predicted_score"] = result["movieId"].map(dict(zip(rec_movie_ids, rec_scores)))

    return result[["title", "genres", "predicted_score"]].sort_values(
        by="predicted_score", ascending=False
    )

# -----------------------------
# 18. TEST COLLABORATIVE FILTERING
# -----------------------------
print("\n" + "-"*60)
print("Collaborative Filtering Recommendation for User 1")
print("-"*60)
print(recommend_collaborative(user_id=1, top_n=5))

# =========================================================
# PART C: FINAL OUTPUT
# =========================================================

print("\n" + "="*60)
print("FINAL RESULTS")
print("="*60)

print("\n1. Similar Movies (Content-Based):")
print(recommend_similar_movies("Toy Story (1995)", top_n=5))

print("\n2. User-Based Content Recommendation:")
print(recommend_by_user_cluster(user_id=1, top_n=5))

print("\n3. Collaborative Filtering Recommendation:")
print(recommend_collaborative(user_id=1, top_n=5))