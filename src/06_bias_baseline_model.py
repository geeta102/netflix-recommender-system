from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

INPUT_FILE = Path("outputs/ratings_with_titles.csv")
OUTPUT_METRICS = Path("outputs/bias_baseline_metrics.csv")
OUTPUT_RECOMMENDATIONS = Path("outputs/bias_baseline_recommendations.csv")

TOP_N = 10


def calculate_rmse(actual, predicted):
    mse = mean_squared_error(actual, predicted)
    return np.sqrt(mse)


def train_bias_baseline(train_data):
    global_mean = train_data["rating"].mean()

    user_mean = train_data.groupby("user_id")["rating"].mean()
    movie_mean = train_data.groupby("movie_id")["rating"].mean()

    user_bias = user_mean - global_mean
    movie_bias = movie_mean - global_mean

    return global_mean, user_bias, movie_bias


def predict_rating(user_id, movie_id, global_mean, user_bias, movie_bias):
    prediction = global_mean

    if user_id in user_bias:
        prediction += user_bias[user_id]

    if movie_id in movie_bias:
        prediction += movie_bias[movie_id]

    prediction = min(5, max(1, prediction))
    return prediction


def generate_user_recommendations(
    user_id,
    train_data,
    movies,
    global_mean,
    user_bias,
    movie_bias,
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
            user_bias,
            movie_bias,
        )

        movie_row = movies[movies["movie_id"] == movie_id].iloc[0]

        recommendations.append(
            {
                "user_id": user_id,
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

    train_data, test_data = train_test_split(
        ratings,
        test_size=0.2,
        random_state=42,
    )

    global_mean, user_bias, movie_bias = train_bias_baseline(train_data)

    predictions = []

    for _, row in test_data.iterrows():
        predicted_rating = predict_rating(
            row["user_id"],
            row["movie_id"],
            global_mean,
            user_bias,
            movie_bias,
        )
        predictions.append(predicted_rating)

    rmse = calculate_rmse(test_data["rating"], predictions)

    print("Bias Baseline Model")
    print("=" * 40)
    print("Global average rating:", round(global_mean, 2))
    print("RMSE:", round(rmse, 4))

    metrics = pd.DataFrame(
        [
            {
                "model": "Bias Baseline",
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

    sample_user = good_users.sort_values(
        by=["count", "mean"],
        ascending=False,
    ).iloc[0]["user_id"]

    recommendations = generate_user_recommendations(
        sample_user,
        train_data,
        movies,
        global_mean,
        user_bias,
        movie_bias,
    )

    print("\nSample user:", sample_user)
    print("\nTop 10 recommendations:")
    print(recommendations)

    recommendations.to_csv(OUTPUT_RECOMMENDATIONS, index=False)
    print("\nSaved metrics to:", OUTPUT_METRICS)
    print("Saved recommendations to:", OUTPUT_RECOMMENDATIONS)


if __name__ == "__main__":
    main()