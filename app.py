import streamlit as st
import pandas as pd
import pickle
import numpy as np
from streamlit_lottie import st_lottie
import requests

# Page Configuration
st.set_page_config(page_title="AI Student Impact Predictor", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .prediction-card {
        padding: 20px;
        border-radius: 10px;
        background-color: white;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)

# Function to load Lottie animations
def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

lottie_student = load_lottieurl("https://assets5.lottiefiles.com/packages/lf20_96bovdur.json")

# Header
st_lottie(lottie_student, height=200, key="coding")
st.title("🎓 Student AI Impact Predictor")
st.write("Determine the impact of AI tools on student grades using your KNN Model.")

# Load the Model
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# User Input Section
st.subheader("📊 Enter Student Details")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=10, max_value=100, value=20)
    gender = st.selectbox("Gender", options=[0, 1], format_func=lambda x: "Male" if x == 0 else "Female")
    edu_level = st.selectbox("Education Level", options=[0, 1, 2], help="0: High School, 1: Undergrad, 2: Postgrad")
    city = st.number_input("City Code", min_value=0, step=1)

with col2:
    ai_tool = st.number_input("AI Tool Used (ID)", min_value=0, step=1)
    usage_hours = st.slider("Daily Usage Hours", 0.0, 24.0, 2.0)
    purpose = st.number_input("Purpose (ID)", min_value=0, step=1)
    student_id = st.number_input("Student ID", min_value=0, step=1)

# Preparation for prediction
# Based on model.pkl, the features are: 
# Student_ID, Age, Gender, Education_Level, City, AI_Tool_Used, Daily_Usage_Hours, Purpose
input_data = pd.DataFrame([[
    student_id, age, gender, edu_level, city, ai_tool, usage_hours, purpose
]], columns=['Student_ID', 'Age', 'Gender', 'Education_Level', 'City', 'AI_Tool_Used', 'Daily_Usage_Hours', 'Purpose'])

# Prediction
if st.button("Predict Impact on Grades"):
    try:
        prediction = model.predict(input_data)
        
        st.markdown("---")
        st.subheader("Result")
        
        result_color = "#2ecc71" if prediction[0] == 1 else "#e74c3c"
        
        st.markdown(f"""
            <div class="prediction-card">
                <h3>Predicted Grade Impact Class:</h3>
                <h1 style="color: {result_color};">{prediction[0]}</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if prediction[0] == 1:
            st.balloons()
            
    except Exception as e:
        st.error(f"Error making prediction: {e}")

st.sidebar.info("This app uses a KNeighborsClassifier model  to analyze academic impact.")
