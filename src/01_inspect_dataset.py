import zipfile
from pathlib import Path

DATA_ZIP = Path("data/archive.zip")


def main():
    if not DATA_ZIP.exists():
        print("Dataset zip not found at data/archive.zip")
        return

    print("Dataset found:", DATA_ZIP)
    print("Zip size in MB:", round(DATA_ZIP.stat().st_size / (1024 * 1024), 2))
    print("\nFiles inside zip:")

    with zipfile.ZipFile(DATA_ZIP, "r") as zip_file:
        for file_info in zip_file.infolist():
            size_mb = file_info.file_size / (1024 * 1024)
            print(f"- {file_info.filename}: {size_mb:.2f} MB")

        print("\nFirst 10 movie titles:")
        with zip_file.open("movie_titles.csv") as file:
            for i, line in enumerate(file):
                if i == 10:
                    break
                print(line.decode("latin1").strip())

        print("\nFirst 20 lines of combined_data_1.txt:")
        with zip_file.open("combined_data_1.txt") as file:
            for i, line in enumerate(file):
                if i == 20:
                    break
                print(line.decode("utf-8").strip())


if __name__ == "__main__":
    main()