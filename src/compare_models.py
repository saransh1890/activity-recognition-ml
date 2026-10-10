
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_data


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    X_train, X_test, y_train, y_test = load_data()

    # All models use the same official training and test sets.
    models = {
        "SVM (RBF)": SVC(kernel="rbf", C=10, gamma="scale"),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),
        "KNN (k=5)": KNeighborsClassifier(
            n_neighbors=5,
            n_jobs=-1
        )
    }

    results = []

    for name, model in models.items():
        print(f"\nTraining {name}...")
        start = time.time()

        model.fit(X_train, y_train)
        training_time = time.time() - start

        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        weighted_f1 = f1_score(
            y_test,
            predictions,
            average="weighted"
        )

        results.append({
            "Model": name,
            "Test Accuracy (%)": round(accuracy * 100, 2),
            "Weighted F1 (%)": round(weighted_f1 * 100, 2),
            "Training Time (seconds)": round(training_time, 2)
        })

        print(f"Test accuracy: {accuracy * 100:.2f}%")
        print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
        print(f"Training time: {training_time:.2f} seconds")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(
        by="Test Accuracy (%)",
        ascending=False
    )

    # Save comparison table
    csv_path = RESULTS_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)

    # Plot accuracy comparison
    plt.figure(figsize=(9, 5))
    plt.bar(
        results_df["Model"],
        results_df["Test Accuracy (%)"]
    )
    plt.ylabel("Test Accuracy (%)")
    plt.xlabel("Machine Learning Model")
    plt.title("Activity Recognition Model Comparison")
    plt.ylim(0, 100)
    plt.tight_layout()

    plot_path = RESULTS_DIR / "model_comparison.png"
    plt.savefig(plot_path, dpi=200)
    plt.close()

    print("\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))
    print("\nSaved table:", csv_path)
    print("Saved graph:", plot_path)


if __name__ == "__main__":
    main()
