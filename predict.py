def predict_sentiment(text):
    text = clean_text(text)
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)
    return prediction[0]

print(predict_sentiment("The flight was delayed and staff were rude"))

