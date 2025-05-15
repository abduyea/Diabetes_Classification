
 Diabetes Classification Experiment

                             Overview

This project focuses on the classification of diabetes using various machine learning models. The dataset utilized contains several health-related features, such as glucose levels and Body Mass Index (BMI), with the primary objective of predicting whether a patient has diabetes.
                             Workflow:

1.Data Preprocessing: This stage involved addressing missing values, balancing the class distribution using the Synthetic Minority Over-sampling Technique (SMOTE), and applying feature scaling.

2.Dimensionality Reduction: Principal Component Analysis (PCA) was employed to reduce the dimensionality of the data to 5 principal components.

3.Modeling: Several machine learning models were trained and evaluated, including K-Nearest Neighbors (KNN), Naive Bayes, Logistic Regression, Decision Trees, and Support Vector Machines (SVM).

4.Evaluation: Model performance was assessed using cross-validation, the F1 score as a key metric, along with comprehensive classification reports and confusion matrices.

Requirements
- Python 3.x
- Required Python libraries: pandas, numpy, matplotlib, seaborn, scikit-learn, imbalanced-learn

Setup

Clone the Repository:
git clone https://github.com/abduyea/E-Commmerce_Project.git


Install Dependencies: Navigate to the cloned repository and run:

pip install -r requirements.txt

Run the Experiment: Open the notebook.ipynb file using Jupyter Notebook or JupyterLab and execute the cells sequentially to reproduce the experiment.

Data Preprocessing Details

Missing Values: Zero values, considered invalid for certain health measurements, were replaced with NaN and subsequently imputed using the median of the respective feature column.

Scaling: Numerical features were standardized using StandardScaler from scikit-learn to ensure consistent scaling across all features.

Class Balancing: The SMOTE technique from the imbalanced-learn library was applied to address the class imbalance present in the target variable.

Models Evaluated

The following machine learning models were included in this evaluation:
K-Nearest Neighbors (KNN)
Naive Bayes
Logistic Regression
Decision Trees
Support Vector Machines (SVM)

Results Summary

The performance of each model was evaluated based on F1 scores obtained through cross-validation and further analyzed using classification reports and confusion matrices on a test dataset. The results indicated that Logistic Regression and Support Vector Machines (SVM) generally outperformed the other models in this classification task.

Visualizations

The project includes several visualizations to aid in understanding the data and model performance:
Feature Correlation Heatmap: Illustrates the pairwise correlations between the features.

Class Distribution Before and After SMOTE: Shows the impact of the SMOTE technique on balancing the target classes.
Histograms of PCA Components: Displays the distribution of the data along the principal components after dimensionality reduction.

Pair Plot for Feature Relationships: Provides a visual representation of the relationships between different pairs of features.

Confusion Matrix Heatmaps: Offers a clear visualization of the classification performance of each model, showing true positives, false positives, true negatives, and false negatives.


