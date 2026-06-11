from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

INPUT_FILE = Path("outputs/ratings_sample.csv")
OUTPUT_DIR = Path("outputs")


def main():
    ratings = pd.read_csv(INPUT_FILE, parse_dates=["date"])

    print("EDA Summary")
    print("=" * 40)

    total_ratings = len(ratings)
    total_users = ratings["user_id"].nunique()
    total_movies = ratings["movie_id"].nunique()
    average_rating = ratings["rating"].mean()

    print("Total ratings:", total_ratings)
    print("Total users:", total_users)
    print("Total movies:", total_movies)
    print("Average rating:", round(average_rating, 2))

    possible_ratings = total_users * total_movies
    sparsity = 1 - (total_ratings / possible_ratings)
    print("Sparsity:", round(sparsity * 100, 2), "%")

    print("\nRating distribution:")
    print(ratings["rating"].value_counts().sort_index())

    print("\nTop 10 most rated movies:")
    print(ratings["movie_id"].value_counts().head(10))

    print("\nTop 10 most active users:")
    print(ratings["user_id"].value_counts().head(10))

    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(7, 4))
    sns.countplot(data=ratings, x="rating")
    plt.title("Rating Distribution")
    plt.xlabel("Rating")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "rating_distribution.png")
    plt.close()

    movie_counts = ratings["movie_id"].value_counts().head(10)

    plt.figure(figsize=(8, 4))
    movie_counts.plot(kind="bar")
    plt.title("Top 10 Most Rated Movies")
    plt.xlabel("Movie ID")
    plt.ylabel("Number of Ratings")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "top_movies.png")
    plt.close()

    print("\nCharts saved:")
    print("- outputs/rating_distribution.png")
    print("- outputs/top_movies.png")


if __name__ == "__main__":
    main()