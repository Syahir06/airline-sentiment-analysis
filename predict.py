import pickle
from preprocessing import clean_text

model = pickle.load(open("../model.pkl", "rb"))
vectorizer = pickle.load(open("../vectorizer.pkl", "rb"))

def predict_sentiment(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

print(predict_sentiment("The flight was delayed and terrible"))
