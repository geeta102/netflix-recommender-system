from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split

INPUT_FILE = Path("outputs/ratings_with_titles.csv")
OUTPUT_FILE = Path("outputs/map_at_10_results.csv")

TOP_K = 10
RELEVANCE_THRESHOLD = 3.5
N_COMPONENTS = 20


def train_bias_baseline(train_data):
    global_mean = train_data["rating"].mean()

    user_mean = train_data.groupby("user_id")["rating"].mean()
    movie_mean = train_data.groupby("movie_id")["rating"].mean()

    user_bias = user_mean - global_mean
    movie_bias = movie_mean - global_mean

    return global_mean, user_bias, movie_bias


def predict_bias_rating(user_id, movie_id, global_mean, user_bias, movie_bias):
    prediction = global_mean

    if user_id in user_bias:
        prediction += user_bias[user_id]

    if movie_id in movie_bias:
        prediction += movie_bias[movie_id]

    prediction = min(5, max(1, prediction))
    return prediction


def train_svd_model(train_data):
    global_mean = train_data["rating"].mean()

    user_ids = sorted(train_data["user_id"].unique())
    movie_ids = sorted(train_data["movie_id"].unique())

    user_to_index = {user_id: index for index, user_id in enumerate(user_ids)}
    movie_to_index = {movie_id: index for index, movie_id in enumerate(movie_ids)}

    row_indices = train_data["user_id"].map(user_to_index)
    column_indices = train_data["movie_id"].map(movie_to_index)

    values = train_data["rating"] - global_mean

    rating_matrix = csr_matrix(
        (values, (row_indices, column_indices)),
        shape=(len(user_to_index), len(movie_to_index)),
    )

    svd = TruncatedSVD(
        n_components=N_COMPONENTS,
        random_state=42,
    )

    user_factors = svd.fit_transform(rating_matrix)
    movie_factors = svd.components_.T

    return global_mean, user_to_index, movie_to_index, user_factors, movie_factors


def predict_svd_rating(
    user_id,
    movie_id,
    global_mean,
    user_to_index,
    movie_to_index,
    user_factors,
    movie_factors,
):
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


def average_precision_at_k(actual_relevant_movies, predicted_movies, k):
    if len(actual_relevant_movies) == 0:
        return None

    score = 0.0
    hits = 0

    for index, movie_id in enumerate(predicted_movies[:k], start=1):
        if movie_id in actual_relevant_movies:
            hits += 1
            precision_at_index = hits / index
            score += precision_at_index

    return score / min(len(actual_relevant_movies), k)


def calculate_map_at_10(test_data, prediction_function):
    average_precision_scores = []

    for user_id, user_test_data in test_data.groupby("user_id"):
        actual_relevant_movies = set(
            user_test_data[
                user_test_data["rating"] >= RELEVANCE_THRESHOLD
            ]["movie_id"]
        )

        if len(actual_relevant_movies) == 0:
            continue

        candidate_movies = user_test_data["movie_id"].unique()

        predictions = []

        for movie_id in candidate_movies:
            predicted_rating = prediction_function(user_id, movie_id)

            predictions.append(
                {
                    "movie_id": movie_id,
                    "predicted_rating": predicted_rating,
                }
            )

        predictions_df = pd.DataFrame(predictions)

        predicted_movies = (
            predictions_df.sort_values(
                by="predicted_rating",
                ascending=False,
            )["movie_id"]
            .tolist()
        )

        ap_score = average_precision_at_k(
            actual_relevant_movies,
            predicted_movies,
            TOP_K,
        )

        if ap_score is not None:
            average_precision_scores.append(ap_score)

    return np.mean(average_precision_scores)


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

    bias_global_mean, user_bias, movie_bias = train_bias_baseline(train_data)

    bias_map_at_10 = calculate_map_at_10(
        test_data,
        lambda user_id, movie_id: predict_bias_rating(
            user_id,
            movie_id,
            bias_global_mean,
            user_bias,
            movie_bias,
        ),
    )

    (
        svd_global_mean,
        user_to_index,
        movie_to_index,
        user_factors,
        movie_factors,
    ) = train_svd_model(train_data)

    svd_map_at_10 = calculate_map_at_10(
        test_data,
        lambda user_id, movie_id: predict_svd_rating(
            user_id,
            movie_id,
            svd_global_mean,
            user_to_index,
            movie_to_index,
            user_factors,
            movie_factors,
        ),
    )

    results = pd.DataFrame(
        [
            {
                "model": "Bias Baseline",
                "map_at_10": round(bias_map_at_10, 4),
                "relevance_definition": "actual rating >= 3.5",
            },
            {
                "model": "Matrix Factorization SVD",
                "map_at_10": round(svd_map_at_10, 4),
                "relevance_definition": "actual rating >= 3.5",
            },
        ]
    )

    print("MAP@10 Evaluation")
    print("=" * 40)
    print(results)

    results.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved MAP@10 results to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()