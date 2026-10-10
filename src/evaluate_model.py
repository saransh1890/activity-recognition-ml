
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

from data_loader import load_data


PROJECT_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_DIR / "results"

ACTIVITY_NAMES = [
    "WALKING",
    "WALKING_UPSTAIRS",
    "WALKING_DOWNSTAIRS",
    "SITTING",
    "STANDING",
    "LAYING"
]

LABELS = [1, 2, 3, 4, 5, 6]


def main():
    # Load the test data and saved model
    _, X_test, _, y_test = load_data()
    model = joblib.load(RESULTS_DIR / "svm_model.joblib")

    # Generate predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Verified test accuracy: {accuracy * 100:.2f}%")

    # Calculate confusion matrix
    matrix = confusion_matrix(
        y_test,
        y_pred,
        labels=LABELS
    )

    # Save the matrix as a readable CSV
    matrix_df = pd.DataFrame(
        matrix,
        index=ACTIVITY_NAMES,
        columns=ACTIVITY_NAMES
    )
    matrix_df.to_csv(RESULTS_DIR / "svm_confusion_matrix.csv")

    # Plot the confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        matrix_df,
        annot=True,
        fmt="d",
        cmap="Blues"
    )
    plt.title("SVM Confusion Matrix")
    plt.xlabel("Predicted Activity")
    plt.ylabel("Actual Activity")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()

    output_path = RESULTS_DIR / "svm_confusion_matrix.png"
    plt.savefig(output_path, dpi=200)
    plt.close()

    print("Confusion matrix saved to:", output_path)


if __name__ == "__main__":
    main()
