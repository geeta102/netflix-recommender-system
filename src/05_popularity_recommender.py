from pathlib import Path

import pandas as pd

INPUT_FILE = Path("outputs/ratings_with_titles.csv")
OUTPUT_FILE = Path("outputs/popularity_recommendations.csv")

MIN_RATINGS = 100
TOP_N = 10


def main():
    ratings = pd.read_csv(INPUT_FILE)

    movie_stats = (
        ratings.groupby(["movie_id", "title", "year"])
        .agg(
            rating_count=("rating", "count"),
            average_rating=("rating", "mean"),
        )
        .reset_index()
    )

    popular_movies = movie_stats[movie_stats["rating_count"] >= MIN_RATINGS]

    recommendations = popular_movies.sort_values(
        by=["average_rating", "rating_count"],
        ascending=False,
    ).head(TOP_N)

    recommendations["average_rating"] = recommendations["average_rating"].round(2)

    print("Top Popular Recommendations")
    print("=" * 40)
    print(recommendations)

    recommendations.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved recommendations to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()