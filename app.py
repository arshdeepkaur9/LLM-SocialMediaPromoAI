# Set up OpenAI API key
import os
import openai
import pandas as pd
import streamlit as st
import numpy as np
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_openai import ChatOpenAI
from transformers import pipeline
import dotenv
dotenv.load_dotenv()

# Set up OpenAI API key (ensure to add your key securely via environment variable)
openai.api_key = os.getenv("OPENAI_API_KEY")
openai_api_key = os.getenv('OPENAI_API_KEY')

# Initialize LangChain LLM
def get_llm(temperature=0.7, model="gpt-4o-mini"):
    return ChatOpenAI(api_key=os.environ["OPENAI_API_KEY"], model_name=model, temperature=temperature)

llm = get_llm()

client = openai.OpenAI()

def get_completion(prompt, model="gpt-4o-mini"):
    """Completes the input text using OpenAI's chat-based completion models.
    Args:
        prompt (str): The text prompt to be completed.
        model (str): The model to use, e.g., 'gpt-3.5-turbo' or 'gpt-4'.

    Returns:
        str: The completed text.
    """
    messages = [
        {"role": "system", "content": "You are a helpful assistant that completes text."},
        {"role": "user", "content": prompt}
    ]
    response = client.chat.completions.create(
        model=model,
        messages=messages
    )
    return response.choices[0].message.content


# Process uploaded files
def process_file(file):
    if file.name.endswith(".csv"):
        return pd.read_csv(file)
    elif file.name.endswith(".xlsx"):
        return pd.read_excel(file)
    else:
        st.warning("Unsupported file format. Please upload a CSV or Excel file.")
        return None

# Engagement Analysis
def perform_engagement_analysis(data, like_column, view_column):
    data[like_column] = pd.to_numeric(data[like_column], errors="coerce")
    data[view_column] = pd.to_numeric(data[view_column], errors="coerce")
    data["Engagement"] = data[like_column] / data[view_column]
    data = data.dropna(subset=["Engagement"])
    return data

model_path = "nlptown/bert-base-multilingual-uncased-sentiment"
model = pipeline(
    model=model_path,
    return_all_scores=True
)

# Sentiment Analysis
def perform_sentiment_analysis(data, text_column, model):
    sentiment_scores = []
    for index, row in data.iterrows():
        text = row[text_column]
        if pd.notna(text):  # Ensure text is not NaN
            # Run sentiment model on the text
            output = model(text)  # Replace 'model' with your sentiment model (e.g., Hugging Face pipeline)
            # Assuming output is a list of dictionaries, access the 'score' key
            probabilities = np.array([d['score'] for d in output[0]])  
            sentiment_values = np.arange(1, len(probabilities) + 1)  # Sentiment values
            sentiment = np.dot(probabilities, sentiment_values) / len(sentiment_values)  # Weighted average
            
            # Normalize sentiment to [0, 1]
            sentiment_scores.append(sentiment)
        else:
            sentiment_scores.append(None)

    # Add results to DataFrame
    data["Sentiment_Score"] = sentiment_scores
    return data

# Display Analysis Results
def display_analysis_results(data, analysis_type):
    if analysis_type == "Engagement Analysis":
        st.subheader("Engagement Analysis Results")
        st.write("### Top 5 High Engagement Posts:")
        st.write(data.sort_values(by="Engagement", ascending=False).head(5)["text"])
        st.write("### Top 5 Low Engagement Posts:")
        st.write(data.sort_values(by="Engagement", ascending=True).head(5)["text"])
    elif analysis_type == "Sentiment Analysis":
        st.subheader("Sentiment Analysis Results")

        # Add a temporary row index for stable sorting
        data = data.reset_index()

        st.write("### Top 5 High Sentiment Posts:")
        st.write(data.sort_values(by=["Sentiment_Score", "index"], ascending=[False, True]).head(5)["text"])

        st.write("### Top 5 Low Sentiment Posts:")
        st.write(data.sort_values(by=["Sentiment_Score", "index"], ascending=[True, True]).head(5)["text"])

        # Drop the temporary index column
        data.drop(columns=["index"], inplace=True, errors="ignore")

# Enhance Low-Performing Posts (Only for Option 3)
def enhance_low_posts(data, text_column, analysis_type):
    st.subheader("Enhanced Low-Performing Posts")
    if analysis_type == "Engagement Analysis":
        low_engagement = data.sort_values(by="Engagement", ascending=True).head(5)
    elif analysis_type == "Sentiment Analysis":
        low_engagement = data.sort_values(by="Sentiment_Score", ascending=True).head(5)

    # Generate enhanced content
    for _, row in low_engagement.iterrows():
        prompt = f"Enhance this post to improve {analysis_type.lower()}:\n{row[text_column]}"
        chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["text"], template="{text}"))
        enhanced_post = chain.run({"text": prompt})
        st.write(f"- Original: {row[text_column]}")
        st.write(f"- Enhanced: {enhanced_post.strip()}")
        st.write("---")

# Generate Social Media Content
def generate_social_media_content(data, text_column, analysis_type, content_type):
    st.subheader("Generated Social Media Content")
    # Map analysis_type to column names
    column_map = {
        "Engagement Analysis": "Engagement",
        "Sentiment Analysis": "Sentiment_Score",
        "Both Engagement and Sentiment Analysis": ["Engagement", "Sentiment_Score"]
    }
    sort_column = column_map[analysis_type]

    # Get top 5 content
    top_5_content = data.sort_values(by=sort_column, ascending=False).head(5)[text_column].tolist()
    
    # Combine top 5 content into a single string
 #   combined_content = " ".join(top_5_content)
    
    prompt = f"""You are social media content evaluator. Go through the top 5 posted content based on {analysis_type}.\n
              Top 5 social media posted content: {top_5_content}\n
              Now as an evaluator your job is to find the commonality in top 5 posted content.
              Your response should only inlcude common theme."""
    common_theme = get_completion(prompt)          
    st.write(f"**Common Theme for Top 5 Social Media posts:** {common_theme}")

    # Generate a single piece of content
    if content_type == "images":
        prompt = f"""You are an expert image generator which generate theme based images for for social media.\n
                Your theme is: {common_theme}.
                Make sure to generate good image the perfectly depicts this theme:{common_theme} """
        generated_images = dalle_generate_image(prompt)
        for image_url in generated_images:
            # Using a generic caption instead of row[text_column]
            st.image(image_url, caption=f"Generated Image based on common theme") 
    else:
        prompt = f"Generate a {content_type} based on the common themes, suitable for social media:\n{common_theme}. Make sure display best results"
        chain = LLMChain(llm=llm, prompt=PromptTemplate(input_variables=["text"], template="{text}"))
        generated_content = chain.run({"text": prompt})
        st.write(f"- Generated:\n {generated_content.strip()}")

# Generate Dalle Image Content
def dalle_generate_image(prompt, num_images=1, size="512x512"):
    """
    Generate an image using the updated DALL·E API based on the given prompt.
    Args:
        prompt (str): The text description for the image generation.
        num_images (int): Number of images to generate.
        size (str): Size of the generated image. Options are "256x256", "512x512", or "1024x1024".
    Returns:
        List of image URLs.
    """
    try:
        response = openai.images.generate(
            prompt=prompt,
            n=num_images,
            size=size,
            response_format="url"
        )

        image_urls = [data.url for data in response.data]  # Access using data.url
        return image_urls
    except openai.OpenAIError as e:  # Use the correct exception class
        st.error(f"Error generating image: {e}")
        return []

# Streamlit App
st.title("Social Buddy- Social Media Content Generator And Analyzer")

# Task Selection
task = st.radio("What would you like to do?", [
    "Create promotional content based on previous content",
    "Create promotional content based on a new idea",
    "Analyze previous content"
])

if task == "Create promotional content based on previous content":
    uploaded_file = st.file_uploader("Upload a CSV or Excel file:", type=["csv", "xlsx"])
    if uploaded_file:
        data = process_file(uploaded_file)
        if data is not None:
            st.write("Uploaded Data Preview:")
            st.dataframe(data.head())

            # Initialize text_column to None
            text_column = None

            # Allow user to select the type of analysis
            analysis_type = st.selectbox(
                "Select basis of content generation:",
                ["Select", "Engagement Analysis", "Sentiment Analysis"]
            )

            if analysis_type == "Engagement Analysis":
                st.write("Select parameters for engagement calculations")
                like_column = st.selectbox("Select the column for 'Likes/Favorites':", ["Select"] + list(data.columns))
                view_column = st.selectbox("Select the column for 'Views':", ["Select"] + list(data.columns))
                if like_column != "Select" and view_column != "Select":
                    data = perform_engagement_analysis(data, like_column, view_column)
                    st.success("Engagement Analysis Successful")

            if analysis_type == "Sentiment Analysis":
                text_column = st.selectbox("Select the column for text content:", ["Select"] + list(data.columns))
                if text_column != "Select":
                    data = perform_sentiment_analysis(data, text_column, model)
                    st.success("Sentiment Analysis Successful")

            # Ensure that the user selected a valid analysis
            if analysis_type != "Select":
                st.write("You can now generate content based on the analysis.")

                # Generate content after analysis
                content_type = st.selectbox(
                    "What type of content do you want to generate?",
                    ["Select", "instagram post", "tweeter tweets", "images", "hashtags"]
                )
                if st.button("Generate Content"):
                    if analysis_type == "Engagement Analysis":
                        generate_social_media_content(data, "text", analysis_type, content_type)
                    elif analysis_type == "Sentiment Analysis" and text_column:
                        generate_social_media_content(data, text_column, analysis_type, content_type)

elif task == "Analyze previous content":
    uploaded_file = st.file_uploader("Upload a CSV or Excel file:", type=["csv", "xlsx"])
    if uploaded_file:
        data = process_file(uploaded_file)
        if data is not None:
            st.write("Uploaded Data Preview:")
            st.dataframe(data.head())
            analysis_type = st.selectbox(
                "Select the type of analysis:",
                ["Select", "Engagement Analysis", "Sentiment Analysis"]
            )

            if analysis_type == "Engagement Analysis":
                st.write("Select parameters for engagement calculations")
                like_column = st.selectbox("Select the column for 'Likes/Favorites':", ["Select"] + list(data.columns))
                view_column = st.selectbox("Select the column for 'Views':", ["Select"] + list(data.columns))
                if like_column != "Select" and view_column != "Select":
                    data = perform_engagement_analysis(data, like_column, view_column)
                    display_analysis_results(data, "Engagement Analysis")
                    enhance_low_posts(data, "text", "Engagement Analysis")

            if analysis_type == "Sentiment Analysis":
                text_column = st.selectbox("Select the column for text content:", ["Select"] + list(data.columns))
                if text_column != "Select":
                    data = perform_sentiment_analysis(data, text_column, model)
                    display_analysis_results(data, "Sentiment Analysis")
                    enhance_low_posts(data, text_column, "Sentiment Analysis")

elif task == "Create promotional content based on a new idea":
    st.subheader("Generate Content from a New Idea")
    idea = st.text_area("Describe your idea:")

    if st.button("Generate Content"):
        # Generate Text Content
        content_prompt = PromptTemplate(
            input_variables=["idea"],
            template="Generate engaging social media content based on this idea:\n{idea}\n\nInclude captions, hashtags, and post suggestions."
        )
        content_chain = LLMChain(llm=llm, prompt=content_prompt)
        generated_content = content_chain.run({"idea": idea})
        st.write("Generated Content:\n")
        st.write(generated_content)

        # Generate Images
        st.subheader("Generated Images")
        image_prompt = f"Generate realistic image that perfectly relfects this idea: {idea}"
        generated_images = dalle_generate_image(image_prompt, num_images=3)
        for image_url in generated_images:
            st.image(image_url, caption=f"Generated Image for: {idea}")
