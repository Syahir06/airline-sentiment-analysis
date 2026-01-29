import pickle

model = pickle.load(open("../model.pkl", "rb"))
vectorizer = pickle.load(open("../vectorizer.pkl", "rb"))

def predict_sentiment(text):
    text_vec = vectorizer.transform([text])
    return model.predict(text_vec)[0]

# Test
print(predict_sentiment("The flight was amazing!"))
print(predict_sentiment("Worst airline ever"))
