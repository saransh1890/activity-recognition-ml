
import json
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"


def main():
    # Load the three-model comparison
    comparison_path = RESULTS_DIR / "model_comparison.csv"
    models_df = pd.read_csv(comparison_path)

    # Load the 30-feature experiment
    feature_path = RESULTS_DIR / "feature_selection_results.json"
    with open(feature_path, "r", encoding="utf-8") as file:
        feature_results = json.load(file)

    # Build a common summary table
    rows = []

    for _, row in models_df.iterrows():
        rows.append({
            "Model": row["Model"],
            "Features Used": 561,
            "Test Accuracy (%)": row["Test Accuracy (%)"],
            "Weighted F1 (%)": row["Weighted F1 (%)"],
            "Training Time (seconds)": row["Training Time (seconds)"]
        })

    rows.append({
        "Model": "SVM (30 selected features)",
        "Features Used": feature_results["selected_feature_count"],
        "Test Accuracy (%)": feature_results["test_accuracy_percent"],
        "Weighted F1 (%)": feature_results["weighted_f1_percent"],
        "Training Time (seconds)": feature_results["training_time_seconds"]
    })

    summary_df = pd.DataFrame(rows).sort_values(
        "Test Accuracy (%)",
        ascending=False
    )

    # Save the combined table
    csv_path = RESULTS_DIR / "final_model_comparison.csv"
    summary_df.to_csv(csv_path, index=False)

    # Save an accuracy graph
    plt.figure(figsize=(10, 6))
    bars = plt.bar(
        summary_df["Model"],
        summary_df["Test Accuracy (%)"]
    )

    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Model")
    plt.title("Activity Recognition: Model Comparison")
    plt.ylim(0, 100)
    plt.xticks(rotation=20, ha="right")

    for bar, accuracy in zip(bars, summary_df["Test Accuracy (%)"]):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{accuracy:.2f}%",
            ha="center",
            va="bottom"
        )

    plt.tight_layout()

    graph_path = RESULTS_DIR / "final_model_comparison.png"
    plt.savefig(graph_path, dpi=200)
    plt.close()

    print("\n===== FINAL MODEL COMPARISON =====")
    print(summary_df.to_string(index=False))
    print("\nSaved table:", csv_path)
    print("Saved graph:", graph_path)


if __name__ == "__main__":
    main()
