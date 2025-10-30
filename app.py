import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('rf2.pkl', 'rb'))

# Page configuration
st.set_page_config(
    page_title="Medical Insurance Claim Predictor",
    page_icon="🏥",
    layout="centered"
)

# App header
st.markdown(
    """
    <h1 style='text-align: center; color: #2F4F4F;'>🏥 Medical Insurance Claim Prediction</h1>
    <p style='text-align: center; color: #555;'>Predict if a person is likely to claim insurance based on their details</p>
    """, unsafe_allow_html=True
)

st.write("---")

# Layout in two columns
col1, col2 = st.columns(2)

with col1:
    age = st.number_input('Age', min_value=0, max_value=120, value=25)
    bmi = st.slider('BMI', min_value=0.0, max_value=50.0, value=22.0, step=0.1)

with col2:
    children = st.number_input('Number of Children', min_value=0, max_value=10, value=0)
    smoker = st.selectbox('Smoker', ('No', 'Yes'))
    charges = st.number_input('Charges', min_value=0.0, max_value=100000.0, value=5000.0, step=100.0)

st.write("---")

# Predict button
if st.button('Predict', key='predict_button'):
    # Convert categorical input to numeric
    smoker_num = 1 if smoker == 'Yes' else 0

    # Prepare features (without sex and region)
    features = np.array([[age, bmi, children, smoker_num, charges]])
    result = model.predict(features)[0]

    # Display result
    if result == 1:
        st.success("✅ Likely to Claim Insurance")
    else:
        st.warning("❌ Unlikely to Claim Insurance")
