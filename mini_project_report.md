# Activity Recognition Using Cell Phones
## Machine Learning Mini-Project Report

**Course:** UE24CS352A – Machine Learning

### 1. Problem Statement

Human Activity Recognition (HAR) uses sensor measurements to identify the physical activity a person is performing. Smartphones contain accelerometers and gyroscopes that capture information about movement and orientation. Machine learning can use these measurements to classify activities automatically.

The objective of this project is to build and evaluate machine-learning models that recognize six human activities: walking, walking upstairs, walking downstairs, sitting, standing, and laying down.

### 2. Dataset Description

This project uses the UCI Human Activity Recognition Using Smartphones dataset. The dataset was collected from 30 volunteers who performed six activities while wearing a smartphone at the waist.

The original dataset contains 10,299 examples represented by 561 engineered features. Sensor signals were sampled at 50 Hz and divided into fixed windows. The provided dataset contains separate training and test sets.

| Dataset property | Description |
|---|---|
| Total examples | 10,299 |
| Training examples | 7,352 |
| Test examples | 2,947 |
| Input features | 561 |
| Activity classes | 6 |
| Missing values detected | 0 |

The six activity classes are WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS, SITTING, STANDING, and LAYING.

### 3. Methodology

The project follows these steps:

1. **Data loading:** Load the official training and test feature files and their activity labels.
2. **Data verification:** Check the file structure, dataset dimensions, and missing values.
3. **Model training:** Train three classifiers using the provided training set.
4. **Evaluation:** Evaluate predictions on the separate test set using accuracy and weighted F1-score.
5. **Feature selection:** Conduct an additional experiment using 30 selected features.
6. **Visualization:** Generate a confusion matrix and model-comparison graph.

The evaluated models are:

- **Support Vector Machine (SVM):** An RBF-kernel SVM learns nonlinear decision boundaries to distinguish activities.
- **Random Forest:** An ensemble of decision trees combines multiple predictions.
- **K-Nearest Neighbors (KNN):** The classifier predicts an activity based on nearby training examples.

An additional experiment uses `SelectKBest` with the ANOVA F-statistic to select 30 features from the original 561. Feature selection is fitted on training data only.

### 4. Implementation

The project was implemented in Python using NumPy, pandas, scikit-learn, Matplotlib, Seaborn, and Joblib.

The implementation is organized into separate Python files for dataset verification, loading, model training, evaluation, model comparison, feature selection, and results summarization.

The original train/test split supplied with the dataset is retained. The models are evaluated on the same test set to make their measured performance comparable.

The SVM confusion matrix is used to examine the activities most frequently misclassified. The comparison table records test accuracy, weighted F1-score, and training time.

### 5. Experimental Results

The following values were obtained during the project experiments.

| Model | Features | Test accuracy | Weighted F1 |
|---|---:|---:|---:|
| SVM (RBF) | 561 | 96.20% | 96.19% |
| Random Forest | 561 | 92.57% | 92.55% |
| KNN (k=5) | 561 | 90.16% | 90.07% |
| SVM with 30 selected features | 30 | 88.16% | 88.09% |

The full-feature RBF SVM achieved the highest test accuracy among the evaluated models. Random Forest ranked second, followed by KNN.

The confusion matrix shows that the model performs particularly well on the LAYING activity. One notable source of error is the distinction between sitting and standing: 50 sitting examples were predicted as standing, while 10 standing examples were predicted as sitting.

In the feature-selection experiment, the input dimensionality was reduced from 561 features to 30 features, a reduction of approximately 94.65%. However, the measured accuracy decreased from 96.20% to 88.16%.

This comparison suggests a trade-off between reducing the number of input features and maintaining predictive performance. The feature-selection experiment also uses standardization, so the observed difference cannot be attributed solely to the number of features.

### 6. Conclusion

This project demonstrates how smartphone sensor measurements can be used for human activity recognition with supervised machine learning.

Among the evaluated models, the RBF SVM achieved the highest test accuracy of 96.20%. Random Forest and KNN provided useful comparison baselines. The 30-feature experiment demonstrated that substantial dimensionality reduction was possible, although it came with a reduction in measured accuracy.

The confusion matrix also showed that activities with similar sensor patterns, particularly sitting and standing, can be harder to distinguish.

Future work could investigate alternative feature-selection strategies, model hyperparameter tuning, and performance across different subjects. Additional experiments should be conducted using a validation set before selecting hyperparameters for final testing.

### 7. Reference

Reyes-Ortiz, J. L., et al. *Human Activity Recognition Using Smartphones*. UCI Machine Learning Repository.

Dataset: https://archive.ics.uci.edu/dataset/240/human%2Bactivity%2Brecogni-tion%2Busing%2Bsmartphones

Reference project: *CS229 Final Report: Activity Recognition Using Cell Phones*, 2015.
