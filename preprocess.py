import pandas as pd
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)       # remove links
    text = re.sub(r'@\w+', '', text)          # remove mentions
    text = re.sub(r'[^a-zA-Z]', ' ', text)    # remove symbols
    words = text.split()
    words = [w for w in words if w not in stop_words]
    return " ".join(words)

def load_and_clean(path):
    df = pd.read_csv(path)
    df = df[['text', 'airline_sentiment']]
    df.dropna(inplace=True)
    df['clean_text'] = df['text'].apply(clean_text)
    return df

if __name__ == "__main__":
    df = load_and_clean("../data/Tweets.csv")
    print(df.head())
