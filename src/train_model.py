
import time
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from data_loader import load_data


# Set up project paths
PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    # Load data
    print("Loading dataset...")
    X_train, X_test, y_train, y_test = load_data()

    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)

    # Train SVM
    print("\nTraining SVM model...")
    model = SVC(kernel="rbf", C=10, gamma="scale")

    start_time = time.time()
    model.fit(X_train, y_train)
    training_seconds = time.time() - start_time

    # Predict test labels
    print("Predicting test activities...")
    y_pred = model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)

    print("\n===== SVM RESULTS =====")
    print(f"Test accuracy: {accuracy * 100:.2f}%")
    print(f"Training time: {training_seconds:.2f} seconds")

    # Detailed classification metrics
    activity_names = [
        "WALKING",
        "WALKING_UPSTAIRS",
        "WALKING_DOWNSTAIRS",
        "SITTING",
        "STANDING",
        "LAYING"
    ]

    print("\n===== CLASSIFICATION REPORT =====")
    print(
        classification_report(
            y_test,
            y_pred,
            labels=[1, 2, 3, 4, 5, 6],
            target_names=activity_names,
            zero_division=0
        )
    )

    # Save the trained model
    model_path = RESULTS_DIR / "svm_model.joblib"
    joblib.dump(model, model_path)

    # Save metrics as JSON
    results = {
        "model": "SVM (RBF kernel)",
        "test_accuracy_percent": round(accuracy * 100, 2),
        "training_time_seconds": round(training_seconds, 2)
    }

    results_path = RESULTS_DIR / "svm_results.json"
    with open(results_path, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=4)

    # Save confusion matrix
    matrix = confusion_matrix(y_test, y_pred, labels=[1, 2, 3, 4, 5, 6])

    matrix_path = RESULTS_DIR / "svm_confusion_matrix.csv"
    np.savetxt(matrix_path, matrix, delimiter=",", fmt="%d")

    print("Saved model:", model_path)
    print("Saved metrics:", results_path)
    print("Saved confusion matrix:", matrix_path)


if __name__ == "__main__":
    main()
