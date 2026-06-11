from pathlib import Path

import pandas as pd

MODEL_COMPARISON_FILE = Path("outputs/model_comparison.csv")
MAP_FILE = Path("outputs/map_at_10_results.csv")
OUTPUT_FILE = Path("outputs/final_results_summary.csv")


def main():
    model_comparison = pd.read_csv(MODEL_COMPARISON_FILE)
    map_results = pd.read_csv(MAP_FILE)

    final_results = model_comparison.merge(
        map_results[["model", "map_at_10"]],
        on="model",
        how="left",
    )

    print("Final Results Summary")
    print("=" * 40)
    print(final_results)

    final_results.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved final results to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()