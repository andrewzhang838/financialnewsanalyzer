# Financial News Sentiment Analyzer

This Python script is designed for sentiment analysis of financial news articles. It integrates machine learning (LSTM) and Google News API to classify articles into sentiments (e.g., Very Bullish, Bullish, Neutral, Bearish, Very Bearish). It also employs a caching mechanism to store sentiment analysis results for efficiency.

**Key Features:**
Dataset Loading and Preprocessing:
- Prepares a labeled dataset for training the LSTM model.
- Uses text cleaning and tokenization to preprocess input text.

LSTM Model:
- A deep learning model is trained to classify sentiments into three categories: positive, neutral, and negative.
- The model is trained on a preprocessed dataset.

Google News Integration:
- Fetches articles related to a specific topic using the Google News API.
- Downloads and processes the content of each article.

Sentiment Analysis with Caching:
- Performs sentiment analysis on fetched articles.
- Stores results in a cache to avoid redundant API calls and predictions.

Result Summarization:
- Generates a summary of the sentiment analysis, showing the distribution of sentiments across analyzed articles.
