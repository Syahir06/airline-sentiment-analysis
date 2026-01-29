import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report

from preprocessing import clean_text  # ✅ correct

# Load dataset
df = pd.read_csv("../data/Tweets.csv")

# Preprocess
df['clean_text'] = df['text'].apply(clean_text)

# Vectorize
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df['clean_text'])
y = df['airline_sentiment']

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Save
pickle.dump(model, open("../model.pkl", "wb"))
pickle.dump(vectorizer, open("../vectorizer.pkl", "wb"))

print("Training complete ✅")
