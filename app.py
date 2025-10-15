import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Loan Approval Prediction System",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with dark theme
st.markdown("""
    <style>
    /* Main page background */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #262730;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #fafafa !important;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        font-size: 16px;
        border: none;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    
    /* Title styling */
    .title {
        color: #fafafa;
        text-align: center;
        font-size: 2.5em;
        margin-bottom: 1em;
        padding: 20px;
        background-color: #262730;
        border-radius: 10px;
    }
    
    /* Subtitle styling */
    .subtitle {
        color: #fafafa;
        text-align: center;
        font-size: 1.5em;
        margin-bottom: 2em;
    }
    
    /* Footer styling */
    .footer {
        text-align: center;
        color: #fafafa;
        margin-top: 2em;
        padding: 20px;
        background-color: #262730;
        border-radius: 10px;
    }
    
    /* Table styling */
    .stTable {
        background-color: #262730 !important;
    }
    
    /* Success message styling */
    .stSuccess {
        background-color: #4CAF50 !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 10px !important;
    }
    
    /* Error message styling */
    .stError {
        background-color: #ff6b6b !important;
        color: white !important;
        padding: 20px !important;
        border-radius: 10px !important;
    }
    </style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<h1 class="title">Loan Approval Prediction System</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict your loan approval status with our advanced AI model</p>', unsafe_allow_html=True)

# Sidebar for user input
with st.sidebar:
    st.markdown('<h2 style="color: #fafafa;">Loan Application Details</h2>', unsafe_allow_html=True)
    
    # Input fields
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.number_input("Applicant Income", min_value=0, value=5000)
    coapplicant_income = st.number_input("Coapplicant Income", min_value=0, value=0)
    loan_amount = st.number_input("Loan Amount", min_value=0, value=100000)
    loan_amount_term = st.number_input("Loan Amount Term (months)", min_value=0, value=360)
    credit_history = st.selectbox("Credit History", ["1", "0"])
    property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# Main content area
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h2 style="color: #fafafa;">Loan Details Summary</h2>', unsafe_allow_html=True)
    
    # Create a summary DataFrame
    summary_data = {
        "Feature": ["Gender", "Married", "Dependents", "Education", "Self Employed",
                   "Applicant Income", "Coapplicant Income", "Loan Amount", "Loan Term",
                   "Credit History", "Property Area"],
        "Value": [gender, married, dependents, education, self_employed,
                 f"${applicant_income:,.2f}", f"${coapplicant_income:,.2f}",
                 f"${loan_amount:,.2f}", f"{loan_amount_term} months",
                 credit_history, property_area]
    }
    
    summary_df = pd.DataFrame(summary_data)
    st.table(summary_df)

with col2:
    st.markdown('<h2 style="color: #fafafa;">Income Analysis</h2>', unsafe_allow_html=True)
    
    # Create a pie chart for income distribution
    income_data = {
        "Category": ["Applicant Income", "Coapplicant Income"],
        "Amount": [applicant_income, coapplicant_income]
    }
    
    fig = px.pie(income_data, values="Amount", names="Category",
                 title="Income Distribution",
                 color_discrete_sequence=['#00CED1', '#FFD700'])
    
    # Update layout for dark theme
    fig.update_layout(
        paper_bgcolor='#0e1117',
        plot_bgcolor='#0e1117',
        font=dict(color='#fafafa'),
        title_font=dict(color='#fafafa'),
        legend_font=dict(color='#fafafa')
    )
    st.plotly_chart(fig, use_container_width=True)

# Prediction button
if st.button("Predict Loan Approval"):
    try:
        # Load the saved model and preprocessing objects
        model = joblib.load('loan_model.pkl')
        label_encoders = joblib.load('label_encoders.pkl')
        scaler = joblib.load('scaler.pkl')
        
        # Preprocess the input data
        input_data = {
            'Gender': [gender],
            'Married': [married],
            'Dependents': [dependents],
            'Education': [education],
            'Self_Employed': [self_employed],
            'ApplicantIncome': [applicant_income],
            'CoapplicantIncome': [coapplicant_income],
            'LoanAmount': [loan_amount],
            'Loan_Amount_Term': [loan_amount_term],
            'Credit_History': [int(credit_history)],
            'Property_Area': [property_area]
        }
        
        # Convert to DataFrame
        input_df = pd.DataFrame(input_data)
        
        # Apply label encoding to categorical variables
        for column in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
            input_df[column] = label_encoders[column].transform(input_df[column])
        
        # Scale numerical features
        numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term']
        input_df[numerical_columns] = scaler.transform(input_df[numerical_columns])
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0]
        
        # Display results
        st.markdown("---")
        st.markdown('<h2 style="color: #fafafa;">Prediction Results</h2>', unsafe_allow_html=True)
        
        if prediction == 1:
            st.success("🎉 Congratulations! Your loan is likely to be approved!")
            st.write(f"Approval Probability: {probability[1]*100:.2f}%")
        else:
            st.error("😔 We're sorry, your loan might not be approved at this time.")
            st.write(f"Rejection Probability: {probability[0]*100:.2f}%")
            
        # Display probability distribution
        fig = go.Figure(data=[
            go.Bar(name='Probability',
                  x=['Rejected', 'Approved'],
                  y=[probability[0]*100, probability[1]*100],
                  marker_color=['#ff6b6b', '#4CAF50'])
        ])
        
        # Update layout for dark theme
        fig.update_layout(
            title="Loan Approval Probability Distribution",
            yaxis_title="Probability (%)",
            showlegend=False,
            paper_bgcolor='#0e1117',
            plot_bgcolor='#0e1117',
            font=dict(color='#fafafa'),
            title_font=dict(color='#fafafa')
        )
        st.plotly_chart(fig, use_container_width=True)
        
    except FileNotFoundError:
        st.error("""
        Model files not found. Please run the following steps:
        1. Make sure you have the 'loan_data.csv' file in the same directory
        2. Run the training script: `python train_model.py`
        3. Refresh this page and try again
        """)

# Footer
st.markdown("---")
st.markdown('<p class="footer">Created by Sanket Uphade</p>', unsafe_allow_html=True) 