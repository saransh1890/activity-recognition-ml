
from pathlib import Path
import pandas as pd

# Locate the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"


def make_unique_names(names):
    """Make duplicate feature names unique without changing their order."""
    unique_names = []
    counts = {}

    for name in names:
        count = counts.get(name, 0) + 1
        counts[name] = count

        if count == 1:
            unique_names.append(name)
        else:
            unique_names.append(f"{name}__duplicate_{count}")

    return unique_names


def load_data():
    """Load the official UCI HAR training and testing datasets."""

    # Load sensor feature values
    X_train = pd.read_csv(
        DATA_DIR / "train" / "X_train.txt",
        sep=r"\s+",
        header=None
    )

    X_test = pd.read_csv(
        DATA_DIR / "test" / "X_test.txt",
        sep=r"\s+",
        header=None
    )

    # Load activity labels
    y_train = pd.read_csv(
        DATA_DIR / "train" / "y_train.txt",
        sep=r"\s+",
        header=None
    ).iloc[:, 0]

    y_test = pd.read_csv(
        DATA_DIR / "test" / "y_test.txt",
        sep=r"\s+",
        header=None
    ).iloc[:, 0]

    # Read feature names
    features = pd.read_csv(
        DATA_DIR / "features.txt",
        sep=r"\s+",
        header=None,
        usecols=[1]
    ).iloc[:, 0].tolist()

    # Give duplicate names unique suffixes
    features = make_unique_names(features)

    # Assign feature names to both datasets
    X_train.columns = features
    X_test.columns = features

    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = load_data()

    activity_names = {
        1: "WALKING",
        2: "WALKING_UPSTAIRS",
        3: "WALKING_DOWNSTAIRS",
        4: "SITTING",
        5: "STANDING",
        6: "LAYING"
    }

    print("\n===== DATASET SUMMARY =====")
    print("Training features:", X_train.shape)
    print("Testing features:", X_test.shape)
    print("Training labels:", y_train.shape)
    print("Testing labels:", y_test.shape)

    print("\nTraining activity distribution:")
    print(
        y_train.map(activity_names)
        .value_counts()
        .to_string()
    )

    print("\nDuplicate column names:", X_train.columns.duplicated().sum())
    print("Missing values in training data:", X_train.isnull().sum().sum())
    print("Missing values in testing data:", X_test.isnull().sum().sum())
