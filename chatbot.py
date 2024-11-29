import streamlit as st
import openai
import pandas as pd
import os
from fuzzywuzzy import process
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# OpenAI API key from environment variable
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    st.error("OpenAI API key is not set. Please configure it as an environment variable.")
    st.stop()

# Load FAQ data
try:
    faq_data = pd.read_csv("faq_data.csv")
    faq_data.columns = faq_data.columns.str.strip()  # Ensure no trailing spaces
    required_columns = {"Question", "Answer", "Category"}  # Include Category for filtering
    if not required_columns.issubset(faq_data.columns):
        raise ValueError(f"The CSV file must contain columns: {required_columns}")
except FileNotFoundError:
    st.error("The FAQ data file (faq_data.csv) was not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading FAQ data: {e}")
    st.stop()

# Sidebar for category selection
st.sidebar.title("FAQ Categories")
categories = faq_data["Category"].unique()
category_selection = st.sidebar.selectbox("Select a category:", ["All"] + list(categories))

# Filter FAQs based on selected category
if category_selection != "All":
    filtered_faq_data = faq_data[faq_data["Category"] == category_selection]
else:
    filtered_faq_data = faq_data

# Chatbot interface
st.title(f"FAQ Chatbot - {category_selection}")
user_input = st.text_input("Ask a question:")

if user_input:
    try:
        # Find the closest match in the FAQ dataset using fuzzy matching
        question_list = filtered_faq_data["Question"].tolist()
        match, score = process.extractOne(user_input, question_list)

        # Display confidence score
        st.write(f"Confidence Score: {score}%")

        if score > 75:  # Confidence threshold
            answer = filtered_faq_data.loc[filtered_faq_data["Question"] == match, "Answer"].values[0]
            st.write(f"Answer: {answer}")
        else:
            # Fallback: Use OpenAI API for answering
            response = openai.ChatCompletion.create(
                model="gpt-4",
                messages=[
                    {"role": "system", "content": "You are a helpful FAQ chatbot."},
                    {"role": "user", "content": user_input}
                ],
                max_tokens=150
            )
            st.write(f"Answer: {response['choices'][0]['message']['content'].strip()}")

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error("An unexpected error occurred. Please check the logs for details.")
