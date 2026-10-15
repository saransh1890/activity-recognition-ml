
from pathlib import Path
import joblib
import pandas as pd

# Find the main project directory
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "UCI HAR Dataset"
MODEL_PATH = ROOT / "results" / "svm_model.joblib"

# Check whether the trained model exists
if not MODEL_PATH.exists():
    print("Model not found. Run: python src/train_model.py")
    raise SystemExit(1)

# Load the saved SVM model
model = joblib.load(MODEL_PATH)

# Load actual test sensor data and labels
X_test = pd.read_csv(
    DATA / "test" / "X_test.txt",
    sep=r"\s+",
    header=None
)

y_test = pd.read_csv(
    DATA / "test" / "y_test.txt",
    sep=r"\s+",
    header=None
).iloc[:, 0]

# Load activity names
activity_df = pd.read_csv(
    DATA / "activity_labels.txt",
    sep=r"\s+",
    header=None,
    names=["id", "activity"]
)

activity_names = dict(
    zip(activity_df["id"], activity_df["activity"])
)

print("\n" + "=" * 45)
print("   HUMAN ACTIVITY RECOGNITION DEMO")
print("=" * 45)
print(f"Loaded test samples: {len(X_test)}")
print(f"Features per sample: {X_test.shape[1]}")

while True:
    choice = input(
        "\nEnter a sample number (1-2947), "
        "or Q to quit: "
    ).strip()

    if choice.lower() == "q":
        print("Demo ended.")
        break

    if not choice.isdigit():
        print("Please enter a valid number.")
        continue

    sample_number = int(choice)

    if not 1 <= sample_number <= len(X_test):
        print("Choose a number between 1 and 2947.")
        continue

    # Select one test sample
    sample = X_test.iloc[[sample_number - 1]]

    # Predict its activity
    prediction = int(model.predict(sample)[0])
    actual = int(y_test.iloc[sample_number - 1])

    predicted_activity = activity_names[prediction]
    actual_activity = activity_names[actual]

    print("\n--- PREDICTION RESULT ---")
    print(f"Sample number:       {sample_number}")
    print(f"Input features:      {sample.shape[1]}")
    print(f"Actual activity:     {actual_activity}")
    print(f"Predicted activity:  {predicted_activity}")

    if prediction == actual:
        print("Result:              CORRECT PREDICTION")
    else:
        print("Result:              INCORRECT PREDICTION")

    print("-" * 30)
