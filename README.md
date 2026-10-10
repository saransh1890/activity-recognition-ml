# Activity Recognition Using Cell Phones

## 1. Project Overview

This project uses machine learning to recognize human activities from smartphone sensor measurements. It classifies sensor data into six activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying down.

The project compares Support Vector Machine (SVM), Random Forest, and K-Nearest Neighbors (KNN). It also investigates whether selecting 30 features instead of using all 561 features affects classification performance.

## 2. Dataset

We use the UCI Human Activity Recognition Using Smartphones dataset.

- **Subjects:** 30 volunteers
- **Activities:** 6
- **Input features:** 561
- **Training examples:** 7,352
- **Test examples:** 2,947
- **Sensor sampling frequency:** 50 Hz

Download the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecogni-tion%2Busing%2Bsmartphones).

Extract the dataset and place the `UCI HAR Dataset` folder inside the project's `data` directory.

The expected location is `data/UCI HAR Dataset/`.

## 3. Technologies Used

- Python
- NumPy and pandas
- scikit-learn
- Matplotlib and Seaborn
- Joblib

## 4. Setup Instructions

Create and activate a virtual environment.

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 5. Running the Project

Run these commands from the main project directory:

Check the dataset:

```powershell
python src/check_dataset.py
```

Load and summarize the dataset:

```powershell
python src/data_loader.py
```

Train and evaluate the SVM model:

```powershell
python src/train_model.py
```

Generate the SVM confusion matrix:

```powershell
python src/evaluate_model.py
```

Compare the three main models:

```powershell
python src/compare_models.py
```

Evaluate the 30-feature SVM:

```powershell
python src/feature_selection.py
```

Generate the combined comparison:

```powershell
python src/summarize_results.py
```

## 6. Experimental Results

The following values were obtained in the current experiment:

| Model | Features | Test Accuracy |
|---|---:|---:|
| SVM (RBF) | 561 | 96.20% |
| Random Forest | 561 | 92.57% |
| KNN (k=5) | 561 | 90.16% |
| SVM with 30 selected features | 30 | 88.16% |

The full-feature RBF SVM achieved the highest test accuracy in these experiments. Feature selection reduced the input dimensionality from 561 to 30 but also reduced test accuracy.

These results describe the current implementation and should not be interpreted as a direct reproduction of the reference paper's experiments.

## 7. Output Files

The `results` directory contains:

- `svm_model.joblib`: trained SVM model
- `svm_results.json`: SVM evaluation summary
- `svm_confusion_matrix.csv`: confusion matrix values
- `svm_confusion_matrix.png`: confusion matrix visualization
- `model_comparison.csv`: comparison of the three main models
- `model_comparison.png`: comparison graph
- `feature_selection_results.json`: results for the 30-feature experiment
- `selected_features.csv`: selected feature names
- `final_model_comparison.csv`: combined model comparison
- `final_model_comparison.png`: combined comparison graph

## 8. Conclusion

The experiments show that the RBF SVM achieved the best test accuracy among the evaluated models. Random Forest and KNN provided useful comparison baselines. The 30-feature experiment demonstrates the trade-off between reducing input dimensionality and maintaining predictive performance.

## 9. Reference

Reyes-Ortiz, J. L., et al. *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecogni-tion%2Busing%2Bsmartphones
