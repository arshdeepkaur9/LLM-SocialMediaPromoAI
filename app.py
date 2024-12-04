import streamlit as st
import pandas as pd
import openai

# Set up OpenAI API key (ensure to add your key securely)
openai.api_key = st.secrets["openai_api_key"]

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
    
    # Allow user to select a column for content analysis
    column = st.selectbox("Select a column to analyze", data.columns)
    
    # Input prompt for AI
    prompt = st.text_area(
        "Describe your promotional goal or theme (e.g., 'Fantasy drama promotion')",
        placeholder="Type your goal here...",
    )
    
    # Generate button
    if st.button("Generate Promotional Text"):
        if prompt and column:
            # Fetch data from the selected column
            text_sample = " ".join(data[column].dropna().sample(5))
            input_prompt = f"{prompt}\nBased on these inputs: {text_sample}"
            
            try:
                # Call OpenAI API
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=input_prompt,
                    max_tokens=150,
                    temperature=0.7,
                )
                st.subheader("Generated Promotional Content")
                st.write(response.choices[0].text.strip())
            except Exception as e:
                st.error(f"Error: {str(e)}")
        else:
            st.warning("Please upload a file, select a column, and provide a prompt!")
