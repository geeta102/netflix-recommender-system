from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

INPUT_FILE = Path("outputs/ratings_with_titles.csv")
OUTPUT_METRICS = Path("outputs/svd_metrics.csv")
OUTPUT_RECOMMENDATIONS = Path("outputs/svd_recommendations.csv")

TOP_N = 10
N_COMPONENTS = 20


def calculate_rmse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return np.sqrt(mse)


def create_mappings(train_data):
    user_ids = sorted(train_data["user_id"].unique())
    movie_ids = sorted(train_data["movie_id"].unique())

    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}

    index_to_movie = {index: movie_id for movie_id, index in movie_to_index.items()}

    return user_to_index, movie_to_index, index_to_movie


def build_sparse_matrix(train_data, user_to_index, movie_to_index, global_mean):
    row_indices = train_data["user_id"].map(user_to_index)
    column_indices = train_data["movie_id"].map(movie_to_index)

    values = train_data["rating"] - global_mean

    matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(user_to_index), len(movie_to_index)),
    )

    return matrix


def train_svd_model(train_data):
    global_mean = train_data["rating"].mean()

    user_to_index, movie_to_index, index_to_movie = create_mappings(train_data)

    rating_matrix = build_sparse_matrix(
        train_data,
        user_to_index,
        movie_to_index,
        global_mean,
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=42,
    )

    user_factors = svd.fit_transform(rating_matrix)
    movie_factors = svd.components_.T

    return global_mean, user_to_index, movie_to_index, index_to_movie, user_factors, movie_factors


def predict_rating(user_id, movie_id, global_mean, user_to_index, movie_to_index, user_factors, movie_factors):
    if user_id not in user_to_index:
        return global_mean

    if movie_id not in movie_to_index:
        return global_mean

    user_index = user_to_index[user_id]
    movie_index = movie_to_index[movie_id]

    prediction = global_mean + np.dot(
        user_factors[user_index],
        movie_factors[movie_index],
    )

    prediction = min(5, max(1, prediction))
    return prediction


def generate_user_recommendations(
    user_id,
    train_data,
    movies,
    global_mean,
    user_to_index,
    movie_to_index,
    user_factors,
    movie_factors,
):
    watched_movies = set(train_data[train_data["user_id"] == user_id]["movie_id"])
    all_movies = set(movies["movie_id"])

    unseen_movies = list(all_movies - watched_movies)

    recommendations = []

    for movie_id in unseen_movies:
        predicted_rating = predict_rating(
            user_id,
            movie_id,
            global_mean,
            user_to_index,
            movie_to_index,
            user_factors,
            movie_factors,
        )

        movie_row = movies[movies["movie_id"] == movie_id].iloc[0]

        recommendations.append(
            {
                "user_id": int(user_id),
                "movie_id": movie_id,
                "title": movie_row["title"],
                "year": movie_row["year"],
                "predicted_rating": round(predicted_rating, 2),
            }
        )

    recommendations_df = pd.DataFrame(recommendations)

    return recommendations_df.sort_values(
        by="predicted_rating",
        ascending=False,
    ).head(TOP_N)


def main():
    ratings = pd.read_csv(INPUT_FILE)

    active_users = ratings["user_id"].value_counts()
    active_users = active_users[active_users >= 5].index

    ratings = ratings[ratings["user_id"].isin(active_users)]

    train_data, test_data = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42,
    )

    (
        global_mean,
        user_to_index,
        movie_to_index,
        index_to_movie,
        user_factors,
        movie_factors,
    ) = train_svd_model(train_data)

    predictions = []

    for _, row in test_data.iterrows():
        predicted_rating = predict_rating(
            row["user_id"],
            row["movie_id"],
            global_mean,
            user_to_index,
            movie_to_index,
            user_factors,
            movie_factors,
        )
        predictions.append(predicted_rating)

    rmse = calculate_rmse(test_data["rating"], predictions)

    print("Matrix Factorization SVD Model")
    print("=" * 40)
    print("Global average rating:", round(global_mean, 2))
    print("Users used:", len(user_to_index))
    print("Movies used:", len(movie_to_index))
    print("SVD components:", N_COMPONENTS)
    print("RMSE:", round(rmse, 4))

    metrics = pd.DataFrame(
        [
            {
                "model": "Matrix Factorization SVD",
                "rmse": round(rmse, 4),
            }
        ]
    )
    metrics.to_csv(OUTPUT_METRICS, index=False)

    movies = ratings[["movie_id", "title", "year"]].drop_duplicates()

    user_summary = (
        train_data.groupby("user_id")["rating"]
        .agg(["count", "mean"])
        .reset_index()
    )

    good_users = user_summary[
        (user_summary["count"] >= 10) & (user_summary["mean"] >= 3.5)
    ]

    sample_user = int(
        good_users.sort_values(
            by=["count", "mean"],
            ascending=False,
        ).iloc[0]["user_id"]
    )

    recommendations = generate_user_recommendations(
        sample_user,
        train_data,
        movies,
        global_mean,
        user_to_index,
        movie_to_index,
        user_factors,
        movie_factors,
    )

    print("\nSample user:", sample_user)
    print("\nTop 10 recommendations:")
    print(recommendations)

    recommendations.to_csv(OUTPUT_RECOMMENDATIONS, index=False)
    print("\nSaved metrics to:", OUTPUT_METRICS)
    print("Saved recommendations to:", OUTPUT_RECOMMENDATIONS)


if __name__ == "__main__":
    main()