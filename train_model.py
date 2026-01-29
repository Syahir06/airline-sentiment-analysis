import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from preprocessing import clean_text   # from src folder

# 1️⃣ Load dataset
df = pd.read_csv("../data/Tweets.csv")

# 2️⃣ Preprocess text
df['clean_text'] = df['text'].apply(clean_text)

# 3️⃣ Feature extraction (TF-IDF)
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['airline_sentiment']

# 4️⃣ Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 5️⃣ Train model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# 6️⃣ Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# 7️⃣ Save model & vectorizer
pickle.dump(model, open("../model.pkl", "wb"))
pickle.dump(vectorizer, open("../vectorizer.pkl", "wb"))

print("Model and vectorizer saved successfully!")
