# Employee HR Analytics: End-to-End Machine Learning Project

## Project Overview

This project applies machine learning techniques to an HR analytics dataset to explore employee attrition, predict monthly income, and identify employee segments through clustering.

The project covers the end-to-end machine learning workflow, including exploratory data analysis (EDA), data preprocessing, feature scaling, categorical encoding, model training, evaluation, threshold tuning, clustering, and model persistence.

**Note:** This is an educational project using a sample HR dataset. Its predictions should not be used to make actual employment decisions.

## Objectives

- Explore employee demographics, job characteristics, income, and attrition.
- Build a regression model to estimate monthly income.
- Build a classification model to predict the employee attrition label.
- Evaluate class imbalance and tune the classification decision threshold.
- Apply K-Means clustering to explore employee segments.
- Save trained models and demonstrate predictions using a Python script.

## Dataset

**Dataset:** IBM HR Analytics Employee Attrition & Performance

**Source:** [Kaggle dataset](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

The selected dataset contains 1,470 records and 14 columns, including employee age, department, job role, monthly income, work experience, overtime, and attrition.

The original dataset is a sample dataset and should not be interpreted as representative of every real-world workforce.

## Technologies Used

- Python
- Pandas and NumPy
- Matplotlib and Seaborn
- Scikit-learn
- Jupyter Notebook
- Joblib

## Project Workflow

### 1. Exploratory Data Analysis

- Inspected dataset structure and data types.
- Examined missing values and duplicate records.
- Analysed the distribution of attrition.
- Explored numerical and categorical employee features.

### 2. Data Preprocessing

- Selected relevant features.
- Standardized numerical features using `StandardScaler`.
- Encoded categorical features using `OneHotEncoder`.
- Combined preprocessing and model training using Scikit-learn pipelines.

### 3. Monthly Income Prediction — Regression

A Linear Regression model was trained to estimate monthly income using employee and job-related features.

| Metric | Result |
|---|---:|
| MAE | 891.96 |
| MSE | 1,374,017.37 |
| RMSE | 1,172.18 |
| R² | 0.9371 |

A mean-based baseline was also evaluated for comparison. The regression model performed better than this baseline on the evaluated test split.

### 4. Employee Attrition Prediction — Classification

Logistic Regression was used to predict the `Attrition` label.

The dataset is imbalanced, with substantially more `No` than `Yes` records. Therefore, accuracy alone is not sufficient to evaluate the model.

Experiments included:
- Default Logistic Regression.
- Class-weighted Logistic Regression.
- Decision-threshold tuning to examine the trade-off between precision and recall.

The selected threshold was 0.60 based on validation-set F1-score comparison.

| Evaluation setup | Accuracy | Precision (Yes) | Recall (Yes) | F1-score (Yes) | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Default Logistic Regression | 0.8571 | 0.67 | 0.21 | 0.32 | 0.7213 |
| Class-weighted Logistic Regression | 0.7279 | 0.33 | 0.68 | 0.44 | 0.7184 |
| Class-weighted, tuned threshold | 0.7891 | 0.39 | 0.60 | 0.47 | 0.7184 |

These results illustrate the trade-off between overall accuracy and detecting the minority `Yes` class. The threshold was selected using a validation split; further validation would be needed before any real-world use.

### 5. Employee Segmentation — K-Means Clustering

K-Means clustering was applied to standardized numerical employee features.

- Tested cluster counts from 2 to 10.
- Compared inertia and silhouette scores.
- Selected two clusters based on the highest silhouette score among the tested values.
- Examined cluster profiles and attrition rates.

The observed attrition rates were approximately 18.6% for Cluster 0 and 8.4% for Cluster 1. These are descriptive associations in this sample, not evidence that cluster membership causes attrition.

### 6. Model Persistence and Prediction

Trained models and supporting configuration are stored in the `models/` directory using Joblib and JSON.

The script `src/predict_employee.py` loads the saved regression and classification models and generates a sample monthly income estimate and attrition prediction.

The demonstration uses the regression model's predicted income as an input to the attrition classifier. The classifier's reported performance metrics were evaluated using the dataset's recorded monthly income, so the combined prediction workflow requires additional validation.

## Project Structure

```text
Employee-ML-Capstone/
├── data/
│   ├── employee_hr_data.csv
│   └── employee_hr_selected.csv
├── models/
│   ├── classification_config.json
│   ├── clustering_scaler.joblib
│   ├── employee_attrition_classifier.joblib
│   ├── employee_kmeans.joblib
│   └── monthly_income_regression.joblib
├── notebooks/
│   └── 01_employee_eda.ipynb
├── results/
│   ├── classification_model_comparison.csv
│   ├── cluster_attrition_rates.csv
│   ├── cluster_profiles.csv
│   └── employee_clusters.csv
├── src/
│   └── predict_employee.py
├── .gitignore
├── README.md
└── requirements.txt
```

## How to Run the Project

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Employee-ML-Capstone
```

### 2. Create and activate a virtual environment

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

Use a Python version compatible with the listed package versions.

### 4. Run the prediction script

```powershell
python src/predict_employee.py
```

The script prints the estimated monthly income, estimated attrition probability, predicted attrition label, and decision threshold for a sample employee.

### 5. Explore the notebook

Open `notebooks/01_employee_eda.ipynb` in VS Code or Jupyter to review the exploratory analysis and modelling workflow.

## Skills Demonstrated

Python, Pandas, NumPy, Exploratory Data Analysis, Data Preprocessing, Feature Engineering, Scikit-learn Pipelines, Linear Regression, Logistic Regression, Classification Metrics, Imbalanced Classification, Decision-Threshold Tuning, K-Means Clustering, Model Evaluation, Model Persistence, and Git/GitHub.

## Future Improvements

- Add cross-validation and systematic hyperparameter tuning.
- Evaluate additional regression and classification algorithms.
- Validate the full prediction workflow without relying on a predicted income feature unless justified by the modelling objective.
- Improve minority-class detection and probability calibration.
- Build an interactive application for exploring model predictions.
