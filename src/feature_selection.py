
import json
import time
from pathlib import Path

import joblib
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

from data_loader import load_data


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    X_train, X_test, y_train, y_test = load_data()

    # Select the 30 features with the strongest ANOVA F-statistics.
    # Feature selection is fitted using training data only.
    model = Pipeline([
        ("feature_selection", SelectKBest(score_func=f_classif, k=30)),
        ("scaler", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=10, gamma="scale"))
    ])

    print("Training SVM with 30 selected features...")
    start = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    weighted_f1 = f1_score(
        y_test, predictions, average="weighted"
    )

    selector = model.named_steps["feature_selection"]
    selected_names = X_train.columns[selector.get_support()].tolist()

    results = {
        "model": "SVM with SelectKBest (30 features)",
        "selected_feature_count": len(selected_names),
        "test_accuracy_percent": round(accuracy * 100, 2),
        "weighted_f1_percent": round(weighted_f1 * 100, 2),
        "training_time_seconds": round(training_seconds, 2),
        "selected_features": selected_names
    }

    print("\n===== FEATURE SELECTION RESULTS =====")
    print(f"Selected features: {len(selected_names)} of {X_train.shape[1]}")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Weighted F1: {weighted_f1 * 100:.2f}%")
    print(f"Training time: {training_seconds:.2f} seconds")

    output_path = RESULTS_DIR / "feature_selection_results.json"
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    joblib.dump(model, RESULTS_DIR / "svm_30_features.joblib")

    pd.DataFrame({"Selected Feature": selected_names}).to_csv(
        RESULTS_DIR / "selected_features.csv",
        index=False
    )

    print("\nSaved feature-selection results:", output_path)


if __name__ == "__main__":
    main()
