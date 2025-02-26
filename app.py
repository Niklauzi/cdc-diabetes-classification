import streamlit as st
import pickle
import pandas as pd
import os
import numpy as np

# Set page config
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Load the saved model and scaler
def load_model_and_scaler():
    try:
        model_path = os.path.join("saved_models", "diabetes_prediction_xgb_2.pkl")
        scaler_path = os.path.join("saved_models", "scaler.pkl")
        
        with open(model_path, "rb") as file:
            model = pickle.load(file)
        with open(scaler_path, "rb") as file:
            scaler = pickle.load(file)
            
        return model, scaler
    except Exception as e:
        st.error(f"Error loading model or scaler: {e}")
        return None, None

# Load model and scaler
model, scaler = load_model_and_scaler()

# Title and description
st.title("Diabetes Prediction System")
st.write("Enter your health information to predict diabetes risk")

# Create input sections using columns
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Medical Conditions")
    highbp = st.selectbox("High Blood Pressure", ["No", "Yes"],
        help="Have you ever been told by a doctor, nurse, or other health professional that you have high blood pressure?")
    highchol = st.selectbox("High Cholesterol", ["No", "Yes"],
        help="Have you ever been told by a health professional that your blood cholesterol is high?")
    cholcheck = st.selectbox("Cholesterol Check in Past 5 Years", ["No", "Yes"],
        help="Have you had your cholesterol checked within the past 5 years?")
    stroke = st.selectbox("Ever Had Stroke", ["No", "Yes"],
        help="Have you ever been told by a health professional that you had a stroke?")
    heart_disease = st.selectbox("Heart Disease/Attack", ["No", "Yes"],
        help="Have you ever been told by a health professional that you had coronary heart disease (CHD) or myocardial infarction (MI)?")
    bmi = st.number_input("BMI", min_value=10.0, max_value=70.0, value=25.0,
        help="Body Mass Index: weight in kilograms divided by height in meters squared (kg/m²)")

with col2:
    st.subheader("Lifestyle & Health Status")
    smoker = st.selectbox("Smoked 100+ Cigarettes in Life", ["No", "Yes"],
        help="Have you smoked at least 100 cigarettes in your entire life? [Note: 5 packs = 100 cigarettes]")
    phys_activity = st.selectbox("Physical Activity in Past 30 Days", ["No", "Yes"],
        help="Have you participated in any physical activities or exercises in the past 30 days, other than your regular job?")
    fruits = st.selectbox("Consume Fruits 1+ Times per Day", ["No", "Yes"],
        help="Do you consume fruits one or more times per day?")
    veggies = st.selectbox("Consume Vegetables 1+ Times per Day", ["No", "Yes"],
        help="Do you consume vegetables one or more times per day?")
    hvy_alcohol = st.selectbox("Heavy Alcohol Consumption", ["No", "Yes"],
        help="Adult men having more than 14 drinks per week and adult women having more than 7 drinks per week")
    gen_health = st.selectbox("General Health Rating from 1 (excellent) to 5 (poor)", [1, 2, 3, 4, 5],
        help="Would you say that in general your health is:\n1 = Excellent\n2 = Very good\n3 = Good\n4 = Fair\n5 = Poor")
    phys_health = st.number_input("Days of Poor Physical Health", min_value=0, max_value=30, value=0,
        help="Now thinking about your physical health, which includes physical illness and injury, for how many days during the past 30 days was your physical health not good?")
    ment_health = st.number_input("Days of Poor Mental Health", min_value=0, max_value=30, value=0,
        help="Now thinking about your mental health, which includes stress, depression, and problems with emotions, for how many days during the past 30 days was your mental health not good?")
    diff_walk = st.selectbox("Difficulty Walking/Climbing Stairs", ["No", "Yes"],
        help="Do you have serious difficulty walking or climbing stairs?")

with col3:
    st.subheader("Demographics & Healthcare")
    sex = st.selectbox("Sex", ["Male", "Female"])
    
    age_groups = {
        1: "18-24 years old",
        2: "25-29 years old",
        3: "30-34 years old",
        4: "35-39 years old",
        5: "40-44 years old",
        6: "45-49 years old",
        7: "50-54 years old",
        8: "55-59 years old",
        9: "60-64 years old",
        10: "65-69 years old",
        11: "70-74 years old",
        12: "75-79 years old",
        13: "80 years and older"
    }
    age = st.selectbox("Age Group", 
        options=list(age_groups.keys()),
        format_func=lambda x: f"{x} ({age_groups[x]})",
        help="Please select your age group")
    
    education_levels = {
        1: "Never attended school or only kindergarten",
        2: "Grades 1-8 (Elementary)",
        3: "Grades 9-11 (Some high school)",
        4: "Grade 12 or GED (High school graduate)",
        5: "College 1-3 years (Some college or technical school)",
        6: "College 4 years or more (College graduate)"
    }
    education = st.selectbox("Education Level",
        options=list(education_levels.keys()),
        format_func=lambda x: f"{x} ({education_levels[x]})",
        help="What is the highest grade or year of school you completed?")
    
    income_levels = {
        1: "Less than $10,000",
        2: "$10,000 to less than $15,000",
        3: "$15,000 to less than $20,000",
        4: "$20,000 to less than $25,000",
        5: "$25,000 to less than $35,000",
        6: "$35,000 to less than $50,000",
        7: "$50,000 to less than $75,000",
        8: "$75,000 or more"
    }
    income = st.selectbox("Income Level",
        options=list(income_levels.keys()),
        format_func=lambda x: f"{x} ({income_levels[x]})",
        help="What is your annual household income?")
    
    any_healthcare = st.selectbox("Have Healthcare Coverage", ["No", "Yes"],
        help="Do you have any kind of health care coverage, including health insurance, prepaid plans such as HMOs, or government plans such as Medicare or Indian Health Services?")

# Create prediction button
if st.button("Predict Diabetes Risk"):
    # Convert Yes/No to 1/0
    binary_map = {"Yes": 1, "No": 0, "Male": 1, "Female": 0}
    
    # Create input data dictionary
    input_data = {
        'PhysHlth': phys_health,
        'HvyAlcoholConsump': binary_map[hvy_alcohol],
        'Smoker': binary_map[smoker],
        'CholCheck': binary_map[cholcheck],
        'Stroke': binary_map[stroke],
        'GenHlth': gen_health,
        'MentHlth': ment_health,
        'Sex': binary_map[sex],
        'Income': income,
        'BMI': bmi,
        'PhysActivity': binary_map[phys_activity],
        'HeartDiseaseorAttack': binary_map[heart_disease],
        'AnyHealthcare': binary_map[any_healthcare],
        'DiffWalk': binary_map[diff_walk],
        'HighChol': binary_map[highchol],
        'Veggies': binary_map[veggies],
        'Education': education,
        'Age': age,
        'HighBP': binary_map[highbp],
        'Fruits': binary_map[fruits]
    }
    
    # Convert to DataFrame
    input_df = pd.DataFrame([input_data])
    
    if model is not None and scaler is not None:
        # Scale the features
        scaled_features = scaler.transform(input_df)
        
        # Make prediction
        prediction = model.predict(scaled_features)[0]
        prediction_proba = model.predict_proba(scaled_features)[0]
        
        # Display result
        st.subheader("Prediction Result")
        if prediction == 0:
            st.success("Based on the provided information, you likely don't have Diabetes.")
        else:
            st.warning("Based on the provided information, you might have Diabetes.")
        
        # Display probability
        st.write(f"Probability of having diabetes: {prediction_proba[1]:.2%}")
        
        # Add disclaimer
        st.info("Note: This prediction is based on statistical analysis and should not be considered as medical advice. Please consult with a healthcare professional for proper diagnosis (Results are screening tools, not diagnoses!!!).")
    else:
        st.error("Error: Model or scaler not loaded properly")

# Add information about the features
with st.expander("Feature Information"):
    st.write("""
    ### Feature Descriptions:
    
    #### Medical Conditions/Metrics:
    - High Blood Pressure (HighBP): Whether you have high blood pressure
    - High Cholesterol (HighChol): Whether you have high cholesterol
    - BMI: Body Mass Index - a measure of body fat based on height and weight
    - Stroke: Whether you've ever had a stroke
    - Heart Disease/Attack: History of heart disease or heart attack
    
    #### Lifestyle Factors:
    - Smoker: Whether you've smoked at least 100 cigarettes in your life
    - Physical Activity: Whether you've been physically active in the past 30 days
    - Fruits/Vegetables: Whether you consume fruits/vegetables at least once per day
    - Heavy Alcohol Consumption: Adult men having more than 14 drinks per week and adult women having more than 7 drinks per week
    
    #### Health Status:
    - General Health: Rating from 1 (excellent) to 5 (poor)
    - Mental/Physical Health: Number of days of poor health in past 30 days
    - Difficulty Walking: Whether you have difficulty walking or climbing stairs
    
    #### Demographics:
    - Age: Age group categories from 18-24 years old to 80 years and older
    - Education: Education level from never attended school to college graduate
    - Income: Annual household income categories from less than $10,000 to $75,000 or more
    """)