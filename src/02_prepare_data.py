import zipfile
from pathlib import Path

import pandas as pd

DATA_ZIP = Path("data/archive.zip")
OUTPUT_DIR = Path("outputs")
OUTPUT_FILE = OUTPUT_DIR / "ratings_sample.csv"

MAX_RATINGS = 250_000
MAX_MOVIES = 1500


def load_ratings_sample():
    rows = []
    current_movie_id = None

    with zipfile.ZipFile(DATA_ZIP, "r") as zip_file:
        with zip_file.open("combined_data_1.txt") as file:
            for raw_line in file:
                line = raw_line.decode("utf-8").strip()

                if line.endswith(":"):
                    current_movie_id = int(line.replace(":", ""))

                    if current_movie_id > MAX_MOVIES:
                        break

                else:
                    user_id, rating, date = line.split(",")

                    rows.append(
                        {
                            "user_id": int(user_id),
                            "movie_id": current_movie_id,
                            "rating": int(rating),
                            "date": date,
                        }
                    )

                    if len(rows) >= MAX_RATINGS:
                        break

    ratings = pd.DataFrame(rows)
    ratings["date"] = pd.to_datetime(ratings["date"])

    return ratings


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)

    print("Reading ratings from Netflix dataset...")
    ratings = load_ratings_sample()

    print("Sample created successfully.")
    print("Rows:", len(ratings))
    print("Users:", ratings["user_id"].nunique())
    print("Movies:", ratings["movie_id"].nunique())
    print("\nFirst 5 rows:")
    print(ratings.head())

    ratings.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved clean ratings file to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()