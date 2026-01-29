import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load model & vectorizer
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Text cleaning function (same as training)
def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

# Prediction function
def predict_sentiment(text):
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return prediction

# ---------------- UI ---------------- #

st.set_page_config(page_title="Airline Tweet Sentiment AI", page_icon="✈️")

st.title("✈️ Airline Tweet Sentiment Analysis")
st.write("This AI model predicts whether a tweet is **Positive**, **Neutral**, or **Negative**.")

user_input = st.text_area("Enter a tweet:")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        result = predict_sentiment(user_input)

        if result == "positive":
            st.success(f"Predicted Sentiment: {result.upper()} 😊")
        elif result == "neutral":
            st.info(f"Predicted Sentiment: {result.upper()} 😐")
        else:
            st.error(f"Predicted Sentiment: {result.upper()} 😠")

st.markdown("---")
st.caption("Built with Machine Learning + NLP + Streamlit")
