import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
import os

# 1. Securely download NLTK data
@st.cache_resource
def load_nltk():
    try:
        # Check if already exists to prevent repeated downloads
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords')
    return set(stopwords.words('english'))

stop_words = load_nltk()

# 2. Load model & vectorizer with the NEW folder path
@st.cache_resource
def load_assets():
    # Use the new folder name 'models'
    model_path = os.path.join("models", "model.pkl")
    vec_path = os.path.join("models", "vectorizer.pkl")
    
    if not os.path.exists(model_path) or not os.path.exists(vec_path):
        return None, None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(vec_path, "rb") as f:
        vectorizer = pickle.load(f)
        
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
    # Updated error message to show where it's looking
    st.error("Error: Could not find 'model.pkl' or 'vectorizer.pkl' inside the '.py' folder.")
    st.info("Current files in '.py/' folder: " + str(os.listdir(".py") if os.path.exists(".py") else "Folder not found"))
else:
    user_input = st.text_area("Enter a tweet about an airline:")

    if st.button("Analyze Sentiment"):
        if user_input.strip() == "":
            st.warning("Please enter some text.")
        else:
            cleaned = clean_text(user_input)
            vectorized = vectorizer.transform([cleaned])
            result = model.predict(vectorized)[0]

            if result == "positive":
                st.success(f"Result: {result.upper()} 😊")
            elif result == "neutral":
                st.info(f"Result: {result.upper()} 😐")
            else:
                st.error(f"Result: {result.upper()} 😠")
