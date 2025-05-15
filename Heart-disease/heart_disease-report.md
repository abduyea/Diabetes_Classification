
1. Introduction

This project aims to develop a predictive model for diagnosing heart disease in patients using a dataset with various health-related attributes. Features include age, cholesterol level, blood pressure, smoking status, and exercise habits, with the target variable indicating the presence or absence of heart disease. The workflow involves comprehensive data preprocessing, feature engineering (including PCA for dimensionality reduction), rigorous model training, and thorough evaluation.

2. Dataset Overview

The heart disease dataset consists of the following key features:

Age: Patient's age (years).
Gender: Patient's biological sex.
Blood Pressure: Systolic blood pressure.
Cholesterol Level**: Cholesterol level (mg/dL).
Exercise Habits: Regular exercise indicator.
Smoking**: Smoking status.
Family History**: Family history of heart disease.
BMI: Body Mass Index (BMI).
Heart Disease Status: Binary target variable (Yes/No) indicating heart disease presence.

3. Data Preprocessing

Handling Missing Values:

Numeric Columns: Missing values were imputed with the median value.
Categorical Columns: Missing values were imputed with the most frequent value.

eature Engineering:

Label Encoding: Categorical features (e.g., Gender, Smoking, Exercise Habits) were encoded numerically using Label Encoding.
Standardization: Numerical features were standardized usingStandardScaler to improve model performance.
SMOTE: Synthetic Minority Over-sampling Technique was applied to balance the class distribution in the target variable.
Dimensionality Reduction (PCA):
Principal Component Analysis (PCA) reduced the dimensionality of the feature space, retaining 5 principal components that captured most of the data's variance.


4. Model Training and Evaluation

Several machine learning models were trained and evaluated:

K-Nearest Neighbors (KNN)
Naive Bayes
Logistic Regression
Decision Tree Classifier
Support Vector Machine (SVM)

Performance was assessed using 5-fold cross-validation** with the F1-score as the primary evaluation metric. Additionally, classification reports and confusion matrices were generated for each model to assess precision, recall, F1-score, and support.
5. Results

Model Performance (based on cross-validation F1-scores):

KNN: High recall for detecting heart disease cases but moderate precision.
Logistic Regression: Balanced performance with slightly lower recall compared to KNN.
SVM: Strong overall performance but sensitive to class imbalance.
Decision Tree: Tended to overfit, especially with SMOTE-generated data.
Naive Bayes: Fair performance, but not the top performer.

Visualizations of the confusion matrices and classification reports provided deeper insights into model strengths and weaknesses.


6. Visualizations

The following visualizations were created for analysis:

Heatmap of Feature Correlations: Displays correlations between principal components after PCA.
Class Distribution Count Plot: Shows the distribution of classes after applying SMOTE.
Histograms of Feature Distributions: Visualizes the distribution of values in the first principal component.
Pair Plo: Displays pairwise relationships between principal components, colored by the target variable.

7. Conclusion

Key findings suggest that heart disease prediction is influenced by both physiological factors (age, cholesterol, blood pressure) and lifestyle factors (smoking, exercise habits). PCA effectively reduced dimensionality while preserving essential information. SMOTE addressed class imbalance and improved model performance. The best-performing model,KNN, demonstrated high recall for detecting heart disease, but further tuning and adjustments are needed to improve precision and overall performance.