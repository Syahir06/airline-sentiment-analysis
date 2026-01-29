import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# 1. Securely download NLTK data
@st.cache_resource
def load_nltk():
    nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = load_nltk()

# 2. Load model & vectorizer with error handling
@st.cache_resource
def load_assets():
    if not os.path.exists("model.pkl") or not os.path.exists("vectorizer.pkl"):
        return None, None
    model = pickle.load(open("model.pkl", "rb"))
    vectorizer = pickle.load(open("vectorizer.pkl", "rb"))
    return model, vectorizer

model, vectorizer = load_assets()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# ---------------- UI ---------------- #
st.set_page_config(page_title="Airline Sentiment AI", page_icon="✈️")
st.title("✈️ Airline Tweet Sentiment Analysis")

if model is None:
    st.error("Error: 'model.pkl' and 'vectorizer.pkl' not found in the root directory!")
else:
    user_input = st.text_area("Enter a tweet about an airline:")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            # Clean and then Transform
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            result = model.predict(vectorized)[0]

            if result == "positive":
                st.success(f"Result: {result.upper()} 😊")
            elif result == "neutral":
                st.info(f"Result: {result.upper()} 😐")
            else:
                st.error(f"Result: {result.upper()} 😠")
