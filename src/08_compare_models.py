from pathlib import Path

import pandas as pd

BIAS_METRICS = Path("outputs/bias_baseline_metrics.csv")
SVD_METRICS = Path("outputs/svd_metrics.csv")
OUTPUT_FILE = Path("outputs/model_comparison.csv")


def main():
    bias_metrics = pd.read_csv(BIAS_METRICS)
    svd_metrics = pd.read_csv(SVD_METRICS)

    comparison = pd.concat(
        [bias_metrics, svd_metrics],
        ignore_index=True,
    )

    comparison = comparison.sort_values(
        by="rmse",
        ascending=True,
    )

    best_model = comparison.iloc[0]

    print("Model Comparison")
    print("=" * 40)
    print(comparison)

    print("\nBest model based on RMSE:")
    print(best_model["model"], "with RMSE =", best_model["rmse"])

    comparison.to_csv(OUTPUT_FILE, index=False)
    print("\nSaved comparison to:", OUTPUT_FILE)


if __name__ == "__main__":
    main()