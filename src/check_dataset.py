
from pathlib import Path

# Locate the project and dataset folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data" / "UCI HAR Dataset"

print("Project folder:", PROJECT_DIR)
print("Dataset folder:", DATA_DIR)
print("Dataset folder exists:", DATA_DIR.exists())

# Check the important dataset files
required_files = [
    "activity_labels.txt",
    "features.txt",
    "train/X_train.txt",
    "train/y_train.txt",
    "train/subject_train.txt",
    "test/X_test.txt",
    "test/y_test.txt",
    "test/subject_test.txt",
]

print("\nChecking required files:")
for file_name in required_files:
    file_path = DATA_DIR / file_name
    print(f"{file_name}: {'FOUND' if file_path.exists() else 'MISSING'}")
