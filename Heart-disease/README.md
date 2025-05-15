

Heart Disease Classification Project

1. Overview

This project develops a machine learning model to predict the presence of heart disease using health-related attributes such as age, cholesterol, and blood pressure. The workflow includes data preprocessing, model training, evaluation, and visualization to assess the performance of several classification algorithms.


2. Key Steps

Data Preprocessing: Addressed missing values, balanced class distribution using SMOTE, and scaled features for model readiness.
Modeling: Evaluated five machine learning models: KNN, Logistic Regression, Decision Trees, Naive Bayes, and SVM.
Evaluation: Models were assessed using cross-validation*and the F1-scoreas the key metric, alongside classification reports and confusion matrices.
Visualization: Generated visualizations such as correlation heatmaps, class distributions, and confusion matrix heatmaps.


3. Requirements
Python: Version 3.x
Libraries:

   pandas: Data manipulation
   numpy: Numerical computations
   matplotlib & seaborn: Visualization
  scikit-learn: Machine learning models and metrics
  imbalanced-learn: For SMOTE class balancing

To install dependencies, run:

pip install -r requirements.txt

4.  Files

heart_disease.csv: Dataset for training and testing.
notebook.ipynb: Jupyter Notebook with the analysis and code.
requirements.txt: Python dependencies for the project.

5.   Setup Instructions

1. Clone the repository:


   git clone https://github.com/abduyea/E-Commmerce_Project.git
   
2. Navigate to the project directory:

   cd heart-disease-classification

3.Set up a virtual environment**:


   python -m venv env
   source env/bin/activate  # Linux/macOS
Windows: env\Scripts\activate
   
4.Install dependencies:
   pip install -r requirements.txt

5. Run the Jupyter Notebook:

   jupyter notebook notebook.ipynb


6. Data Preprocessing

Missing Values
Missing values were imputed using the median for numeric columns and the mode for categorical columns.

Feature Scaling
Numeric features were standardized using StandardScaler to ensure consistent scaling for distance-based algorithms.

SMOTE
SMOTE (Synthetic Minority Over-sampling Technique) was applied to address class imbalance in the target variable by generating synthetic samples of the minority class.

Models Evaluated
The following models were trained and evaluated:

K-Nearest Neighbors (KNN)

Naive Bayes

Logistic Regression

Decision Trees

Support Vector Machines (SVM)

Results
Performance was evaluated using stratified k-fold cross-validation, with the F1-score as the primary metric. Each model was assessed for precision, recall, and accuracy through classification reports and confusion matrices.

Visualizations
The following visualizations were generated:

Feature Correlation Heatmap: To visualize relationships between features.

Class Distribution (Before & After SMOTE): To compare the distribution of the target variable before and after SMOTE.

Confusion Matrix Heatmap: To visually represent model performance.

Future Work
Future improvements may include:

Hyperparameter Tuning
