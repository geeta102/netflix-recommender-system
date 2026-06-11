import csv
import zipfile
from pathlib import Path

import pandas as pd

DATA_ZIP = Path("data/archive.zip")
RATINGS_FILE = Path("outputs/ratings_sample.csv")
OUTPUT_FILE = Path("outputs/ratings_with_titles.csv")


def load_movie_titles():
    rows = []

    with zipfile.ZipFile(DATA_ZIP, "r") as zip_file:
        with zip_file.open("movie_titles.csv") as file:
            reader = csv.reader(line.decode("latin1") for line in file)

            for row in reader:
                if len(row) >= 3:
                    movie_id = int(row[0])
                    year = row[1]
                    title = ",".join(row[2:])

                    rows.append(
                        {
                            "movie_id": movie_id,
                            "year": year,
                            "title": title,
                        }
                    )

    return pd.DataFrame(rows)


def main():
    ratings = pd.read_csv(RATINGS_FILE, parse_dates=["date"])
    movies = load_movie_titles()

    ratings_with_titles = ratings.merge(movies, on="movie_id", how="left")

    print("Ratings with titles created.")
    print("Rows:", len(ratings_with_titles))
    print("\nFirst 5 rows:")
    print(ratings_with_titles.head())

    print("\nTop 10 most rated movies with titles:")
    top_movies = (
        ratings_with_titles.groupby(["movie_id", "title"])
        .size()
        .sort_values(ascending=False)
        .head(10)
    )
    print(top_movies)

    ratings_with_titles.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved file to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()