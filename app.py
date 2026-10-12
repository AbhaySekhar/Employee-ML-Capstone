
import streamlit as st
from src.predict_employee import predict_employee

st.set_page_config(
    page_title="Employee HR Analytics",
    page_icon="📊",
    layout="centered"
)

st.title("Employee HR Analytics")
st.write(
    "Estimate monthly income and explore attrition predictions "
    "using a machine learning model."
)

with st.form("employee_form"):
    age = st.number_input("Age", 18, 65, 30)
    distance = st.number_input("Distance From Home", 1, 29, 5)
    education = st.selectbox("Education", [1, 2, 3, 4, 5])
    job_level = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    companies = st.number_input("Previous Companies", 0, 10, 2)
    experience = st.number_input("Total Working Years", 0, 40, 5)
    years_company = st.number_input("Years At Company", 0, 40, 3)
    years_role = st.number_input("Years In Current Role", 0, 18, 2)

    travel = st.selectbox(
        "Business Travel",
        ["Travel_Rarely", "Travel_Frequently", "Non-Travel"]
    )

    department = st.selectbox(
        "Department",
        ["Research & Development", "Sales", "Human Resources"]
    )

    job_role = st.selectbox(
        "Job Role",
        [
            "Research Scientist",
            "Laboratory Technician",
            "Manufacturing Director",
            "Healthcare Representative",
            "Manager",
            "Sales Executive",
            "Sales Representative",
            "Research Director",
            "Human Resources"
        ]
    )

    overtime = st.selectbox("OverTime", ["No", "Yes"])

    submitted = st.form_submit_button("Predict")

if submitted:
    employee = {
        "Age": age,
        "DistanceFromHome": distance,
        "Education": education,
        "JobLevel": job_level,
        "NumCompaniesWorked": companies,
        "TotalWorkingYears": experience,
        "YearsAtCompany": years_company,
        "YearsInCurrentRole": years_role,
        "BusinessTravel": travel,
        "Department": department,
        "JobRole": job_role,
        "OverTime": overtime
    }

    try:
        result = predict_employee(employee)

        st.subheader("Prediction Results")
        st.metric(
            "Estimated Monthly Income",
            f"{result['predicted_monthly_income']:,.2f}"
        )
        st.metric(
            "Attrition Probability",
            f"{result['attrition_probability']:.1%}"
        )
        st.write("Predicted Attrition:", result["predicted_attrition"])
        st.caption(
            "Educational demonstration only. "
            "Predictions are not validated employment decisions."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")
