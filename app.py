import streamlit as st
import pandas as pd

st.title("Social Media Promotion AI")
st.write("Welcome to the AI-powered promotional content generator!")
st.write("This tool will help you analyze social media data and generate promotional posts.")

# File uploader
uploaded_file = st.file_uploader("Upload a CSV file with social media data", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded file
    data = pd.read_csv(uploaded_file)
    st.write("Uploaded Data Preview:")
    st.write(data.head())
