import pandas as pd
import numpy as np
import requests
import logging
import datetime
import hashlib
import pickle
import os
import re
import sys
import time
import tensorflow as tf
from tqdm import tqdm
from keras.preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from newspaper import Article
from googleapiclient.discovery import build
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s', handlers=[logging.StreamHandler(sys.stdout)])

# API Keys
GOOGLE_NEWS_API_KEY = os.getenv('GOOGLE_NEWS_API_KEY')
if not GOOGLE_NEWS_API_KEY:
    logging.error("Google News API key not set in environment variables.")
    exit(1)

# Global settings
EXPIRATION_LENGTH = datetime.timedelta(days=2)
CACHE_DIR = 'cache'

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

# Dataset loading and preprocessing
def load_and_preprocess_dataset():
    logging.info("Loading dataset...")
    df = pd.read_csv('/content/drive/MyDrive/CAIS Curriculum Project/projfiles/all-data.csv', delimiter=',', encoding='latin-1')
    df.columns = ['sentiment', 'Message']
    df['sentiment'] = df['sentiment'].map({'positive': 0, 'neutral': 1, 'negative': 2})

    def clean_text(text):
        text = re.sub(r'http\S+', '<URL>', text)
        text = text.lower()
        return text

    df['Message'] = df['Message'].apply(clean_text)

    tokenizer = Tokenizer(num_words=500000, lower=True)
    tokenizer.fit_on_texts(df['Message'].values)
    X = tokenizer.texts_to_sequences(df['Message'].values)
    X = pad_sequences(X, maxlen=50)
    Y = pd.get_dummies(df['sentiment']).values

    return train_test_split(X, Y, test_size=0.15, random_state=42), tokenizer

# LSTM model creation and training
def create_and_train_model(X_train, Y_train):
    model = Sequential()
    model.add(Embedding(input_dim=500000, output_dim=100, input_length=50))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dense(3, activation="softmax"))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=['accuracy'])

    batch_size = 32
    model.fit(X_train, Y_train, epochs=5, batch_size=batch_size, verbose=2)
    return model

# Fetch articles using Google News API
def get_google_news_articles(topic, num_articles):
    service = build("customsearch", "v1", developerKey=GOOGLE_NEWS_API_KEY)
    res = service.cse().list(q=topic, cx='YOUR_SEARCH_ENGINE_ID', num=num_articles).execute()
    articles = res.get('items', [])
    return [{'title': item['title'], 'url': item['link']} for item in articles]

def get_article_content(url, title):
    article = Article(url)
    try:
        article.download()
        article.parse()
        return article.text if article.text else title
    except Exception as e:
        logging.error(f"Error fetching article content: {e}")
        return title

# Sentiment caching and analysis
def analyze_and_cache_sentiments(articles, model, tokenizer, cache_file):
    sentiment_cache = load_cache(cache_file)
    now = datetime.datetime.now()

    results = []

    for article in articles:
        url = article['url']
        title = article['title']

        if url in sentiment_cache and (now - sentiment_cache[url]['cached_time']) < EXPIRATION_LENGTH:
            results.append(sentiment_cache[url])
            continue

        content = get_article_content(url, title)
        seq = tokenizer.texts_to_sequences([content])
        padded = pad_sequences(seq, maxlen=50)
        prediction = model.predict(padded)
        sentiment = np.argmax(prediction)

        entry = {
            'title': title,
            'url': url,
            'sentiment': sentiment,
            'confidence': float(np.max(prediction)),
            'cached_time': now
        }

        results.append(entry)
        sentiment_cache[url] = entry

    save_cache(cache_file, sentiment_cache)
    return results

def load_cache(cache_file):
    if os.path.exists(cache_file):
        with open(cache_file, 'rb') as f:
            return pickle.load(f)
    return {}

def save_cache(cache_file, cache):
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)

# Summarize results
def summarize_results(results, topic):
    sentiments = ['Very Bearish', 'Bearish', 'Neutral', 'Bullish', 'Very Bullish']
    counts = {sent: 0 for sent in sentiments}

    for result in results:
        counts[sentiments[result['sentiment']]] += 1

    total = len(results)
    summary = f"Sentiment Analysis for '{topic}':\n"
    for sent, count in counts.items():
        percentage = (count / total) * 100
        summary += f"{sent}: {count} articles ({percentage:.2f}%)\n"
    return summary

# Main function
def main():
    parser = argparse.ArgumentParser(description="Sentiment Analysis of Financial News")
    parser.add_argument('-n', '--num_articles', type=int, default=5, help='Number of articles to analyze')
    parser.add_argument('-t', '--topic', type=str, default='Financial News', help='Topic to fetch news for')
    parser.add_argument('-c', '--cache_file', type=str, default=f'{CACHE_DIR}/sentiment_cache.pkl', help='Path to sentiment cache file')
    args = parser.parse_args()

    (X_train, X_test, Y_train, Y_test), tokenizer = load_and_preprocess_dataset()
    model = create_and_train_model(X_train, Y_train)

    articles = get_google_news_articles(args.topic, args.num_articles)
    results = analyze_and_cache_sentiments(articles, model, tokenizer, args.cache_file)

    summary = summarize_results(results, args.topic)
    print(summary)

if __name__ == "__main__":
    main()
