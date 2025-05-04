import pandas as pd
import numpy as np
import streamlit as st
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

# Set page config
st.set_page_config(
    page_title="Diabetes Prediction App",
    page_icon="🩺",
    layout="centered"
)

# App title and description
st.title("Diabetes Risk Prediction Tool")
st.markdown("Enter your health information to check your diabetes risk")

# Define the path for saving user input data
DATA_PATH = "user_input_data.csv"

@st.cache_resource
def load_model():
    # Load data
    df = pd.read_csv('Datasets/diabetes_binary_health_indicators_BRFSS2015.csv')
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Split features and target
    X = df.drop('Diabetes_binary', axis=1)
    y = df['Diabetes_binary']
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)
    
    return model, scaler, X.columns

# Function to save user input data to CSV
def save_to_csv(input_data, probability, prediction):
    # Create a copy of the input data
    data_to_save = input_data.copy()
    
    # Add timestamp, prediction result and probability
    data_to_save['timestamp'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    data_to_save['prediction'] = prediction
    data_to_save['probability'] = probability
    
    # Check if file exists
    if os.path.exists(DATA_PATH):
        # Append to existing file without writing the header
        data_to_save.to_csv(DATA_PATH, mode='a', header=False, index=False)
    else:
        # Create new file with header
        data_to_save.to_csv(DATA_PATH, index=False)
    
    return True

# Load model, scaler, and feature names
model, scaler, feature_names = load_model()

# Create two columns for input layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Basic Information")
    age_category = st.selectbox("Age", 
                        options=[
                            "18-24", "25-29", "30-34", "35-39", "40-44", 
                            "45-49", "50-54", "55-59", "60-64", "65-69", 
                            "70-74", "75-79", "80+"
                        ])
    # Convert age category to numeric value (1-13)
    age_mapping = {
        "18-24": 1, "25-29": 2, "30-34": 3, "35-39": 4, "40-44": 5,
        "45-49": 6, "50-54": 7, "55-59": 8, "60-64": 9, "65-69": 10,
        "70-74": 11, "75-79": 12, "80+": 13
    }
    age_value = age_mapping[age_category]
    
    sex = st.radio("Sex", ["Female", "Male"])
    sex_value = 1 if sex == "Male" else 0
    
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=25.0, step=0.1)
    
    education = st.selectbox("Education Level", 
                           ["Never attended school", "Elementary", "Some high school", 
                            "High school graduate", "Some college", "College graduate"])
    # Convert education to numeric value (1-6)
    education_mapping = {
        "Never attended school": 1, "Elementary": 2, "Some high school": 3,
        "High school graduate": 4, "Some college": 5, "College graduate": 6
    }
    education_value = education_mapping[education]
    
    income = st.selectbox("Income Level", 
                        ["Less than 10,000", "10,000-15,000", "15,000-20,000", 
                         "20,000-25,000", "25,000-35,000", "35,000-50,000", 
                         "50,000-75,000", "75,000 or more"])
    # Convert income to numeric value (1-8)
    income_mapping = {
        "Less than 10,000": 1, "10,000-15,000": 2, "15,000-20,000": 3,
        "20,000-25,000": 4, "25,000-35,000": 5, "35,000-50,000": 6,
        "50,000-75,000": 7, "75,000 or more": 8
    }
    income_value = income_mapping[income]

with col2:
    st.subheader("Health Indicators")
    high_bp = st.checkbox("High Blood Pressure")
    high_chol = st.checkbox("High Cholesterol")
    chol_check = st.checkbox("Cholesterol Check in Past 5 Years", value=True)
    smoker = st.checkbox("Smoked at least 100 cigarettes in life")
    stroke = st.checkbox("Ever had a stroke")
    heart_disease = st.checkbox("Heart Disease or Heart Attack")
    phys_activity = st.checkbox("Physical Activity in Past 30 Days", value=True)
    fruits = st.checkbox("Consume Fruits Daily", value=True)
    veggies = st.checkbox("Consume Vegetables Daily", value=True)
    heavy_alcohol = st.checkbox("Heavy Alcohol Consumption")
    healthcare = st.checkbox("Have Healthcare Coverage", value=True)
    no_doc_cost = st.checkbox("Could not see doctor due to cost")
    difficulty_walking = st.checkbox("Difficulty Walking or Climbing Stairs")
    
    general_health = st.select_slider(
        "General Health",
        options=["Excellent", "Very Good", "Good", "Fair", "Poor"],
        value="Good"
    )
    # Convert general health to numeric (1-5)
    gen_health_mapping = {"Excellent": 1, "Very Good": 2, "Good": 3, "Fair": 4, "Poor": 5}
    gen_health_value = gen_health_mapping[general_health]
    
    mental_health = st.slider("Days of Poor Mental Health (Past 30 Days)", 0, 30, 0)
    physical_health = st.slider("Days of Poor Physical Health (Past 30 Days)", 0, 30, 0)

# Prediction button
if st.button("Predict Diabetes Risk", type="primary"):
    # Create input data in correct format
    input_data = pd.DataFrame([[
        1 if high_bp else 0,
        1 if high_chol else 0,
        1 if chol_check else 0,
        bmi,
        1 if smoker else 0,
        1 if stroke else 0,
        1 if heart_disease else 0,
        1 if phys_activity else 0,
        1 if fruits else 0,
        1 if veggies else 0,
        1 if heavy_alcohol else 0,
        1 if healthcare else 0,
        1 if no_doc_cost else 0,
        gen_health_value,
        mental_health,
        physical_health,
        1 if difficulty_walking else 0,
        sex_value,
        age_value,
        education_value,
        income_value
    ]], columns=feature_names)
    
    # Save the original input values with human-readable labels for storage
    readable_input = pd.DataFrame([{
        'Age': age_category,
        'Sex': sex,
        'BMI': bmi,
        'Education': education,
        'Income': income,
        'HighBP': 'Yes' if high_bp else 'No',
        'HighChol': 'Yes' if high_chol else 'No',
        'CholCheck': 'Yes' if chol_check else 'No',
        'Smoker': 'Yes' if smoker else 'No',
        'Stroke': 'Yes' if stroke else 'No',
        'HeartDiseaseAttack': 'Yes' if heart_disease else 'No',
        'PhysActivity': 'Yes' if phys_activity else 'No',
        'Fruits': 'Yes' if fruits else 'No',
        'Veggies': 'Yes' if veggies else 'No',
        'HeavyAlcohol': 'Yes' if heavy_alcohol else 'No',
        'Healthcare': 'Yes' if healthcare else 'No',
        'NoDocCost': 'Yes' if no_doc_cost else 'No',
        'DiffWalk': 'Yes' if difficulty_walking else 'No',
        'GeneralHealth': general_health,
        'MentalHealthDays': mental_health,
        'PhysicalHealthDays': physical_health
    }])
    
    # Scale input for prediction
    input_scaled = scaler.transform(input_data)
    
    # Make prediction
    probability = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]
    
    # Save input data and prediction to CSV
    saved = save_to_csv(readable_input, probability, 'High' if prediction == 1 else 'Low')
    
    if saved:
        st.success("Input data and prediction saved successfully!")
    
    # Display results
    st.markdown("---")
    st.header("Prediction Result")
    
    if prediction == 1:
        st.error(f"Risk of Diabetes: HIGH ({probability:.1%} probability)")
    else:
        st.success(f"Risk of Diabetes: LOW ({probability:.1%} probability)")
    
    # Risk factors summary
    st.subheader("Key Risk Factors Identified")
    risk_factors = []
    
    if high_bp:
        risk_factors.append("High blood pressure")
    if high_chol:
        risk_factors.append("High cholesterol")
    if bmi >= 30:
        risk_factors.append(f"BMI in obese range ({bmi:.1f})")
    elif bmi >= 25:
        risk_factors.append(f"BMI in overweight range ({bmi:.1f})")
    if smoker:
        risk_factors.append("Smoking history")
    if stroke:
        risk_factors.append("Previous stroke")
    if heart_disease:
        risk_factors.append("Heart disease")
    if gen_health_value >= 4:
        risk_factors.append("Fair to poor general health")
    if mental_health >= 14:
        risk_factors.append("Poor mental health")
    if physical_health >= 14:
        risk_factors.append("Poor physical health")
    if difficulty_walking:
        risk_factors.append("Mobility issues")
    if age_value >= 7:  # Age 50+
        risk_factors.append("Age factor (50+ years)")
    
    if risk_factors:
        for factor in risk_factors:
            st.write(f"• {factor}")
    else:
        st.write("No significant risk factors identified")
        
    # Disclaimer
    st.markdown("---")
    st.caption("This prediction is for informational purposes only and does not constitute medical advice.")

# Add information about the model
with st.expander("About this prediction model"):
    st.write("""
    This diabetes prediction model uses logistic regression trained on the Behavioral Risk Factor Surveillance System (BRFSS) 2015 dataset.
    
    The model considers various health indicators and demographic factors to estimate the probability of diabetes. The prediction is based on statistical patterns found in population-level data and should be used for informational purposes only.
    """)

# Add a section to view saved data (for admin purposes)
with st.expander("View Saved Data "):
    if os.path.exists(DATA_PATH):
        saved_data = pd.read_csv(DATA_PATH)
        st.dataframe(saved_data)
        
        # Option to download the data
        csv = saved_data.to_csv(index=False)
        st.download_button(
            label="Download Data as CSV",
            data=csv,
            file_name="diabetes_prediction_data.csv",
            mime="text/csv",
        )
    else:
        st.info("No data has been saved yet.")