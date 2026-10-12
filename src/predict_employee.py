
import json
from pathlib import Path

import joblib
import pandas as pd


# Find the main project folder
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"


# Features used by the regression model
REGRESSION_FEATURES = [
    "Age",
    "DistanceFromHome",
    "Education",
    "JobLevel",
    "NumCompaniesWorked",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "BusinessTravel",
    "Department",
    "JobRole",
    "OverTime",
]


# Features used by the classification model
CLASSIFICATION_FEATURES = [
    "Age",
    "BusinessTravel",
    "Department",
    "DistanceFromHome",
    "Education",
    "JobLevel",
    "JobRole",
    "MonthlyIncome",
    "NumCompaniesWorked",
    "TotalWorkingYears",
    "YearsAtCompany",
    "YearsInCurrentRole",
    "OverTime",
]


def load_models():
    regression_model = joblib.load(
        MODELS_DIR / "monthly_income_regression.joblib"
    )

    classification_model = joblib.load(
        MODELS_DIR / "employee_attrition_classifier.joblib"
    )

    with open(
        MODELS_DIR / "classification_config.json",
        "r",
        encoding="utf-8",
    ) as file:
        config = json.load(file)

    return regression_model, classification_model, config


def predict_employee(employee_data):
    regression_model, classification_model, config = load_models()

    employee_df = pd.DataFrame([employee_data])

    # Predict monthly income
    regression_input = employee_df[REGRESSION_FEATURES]
    income_prediction = regression_model.predict(regression_input)[0]

    # Use predicted income to predict attrition
    employee_df["MonthlyIncome"] = income_prediction
    classification_input = employee_df[CLASSIFICATION_FEATURES]

    probabilities = classification_model.predict_proba(
        classification_input
    )[0]

    classes = list(
        classification_model.named_steps["model"].classes_
    )

    yes_index = classes.index("Yes")
    probability_yes = probabilities[yes_index]

    threshold = config["decision_threshold"]
    attrition_prediction = (
        "Yes" if probability_yes >= threshold else "No"
    )

    return {
        "predicted_monthly_income": round(
            float(income_prediction), 2
        ),
        "attrition_probability": round(
            float(probability_yes), 4
        ),
        "predicted_attrition": attrition_prediction,
        "decision_threshold": threshold,
    }


if __name__ == "__main__":
    #example_employee = {
        #"Age": 30,
        #"DistanceFromHome": 5,
        #"Education": 3,
        #"JobLevel": 1,
        #"NumCompaniesWorked": 2,
        #"TotalWorkingYears": 6,
        #"YearsAtCompany": 3,
        #"YearsInCurrentRole": 2,
        #"BusinessTravel": "Travel_Rarely",
        #"Department": "Research & Development",
        #"JobRole": "Research Scientist",
        #"OverTime": "No",
    #}

    
    example_employee = {
    "Age": 45,
    "DistanceFromHome": 10,
    "Education": 4,
    "JobLevel": 3,
    "NumCompaniesWorked": 4,
    "TotalWorkingYears": 20,
    "YearsAtCompany": 12,
    "YearsInCurrentRole": 7,
    "BusinessTravel": "Travel_Rarely",
    "Department": "Research & Development",
    "JobRole": "Research Scientist",
    "OverTime": "Yes",
}


    result = predict_employee(example_employee)

    print("\nEmployee Prediction Results")
    print("-" * 30)

    for key, value in result.items():
        print(f"{key}: {value}")
