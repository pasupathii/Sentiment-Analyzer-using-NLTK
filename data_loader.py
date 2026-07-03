"""
data_loader.py – Multi-dataset loading, cleaning, and preprocessing pipeline.

Combines three datasets:
1. Twitter US Airline Sentiment (local dataset.csv)
2. NLTK Movie Reviews corpus
3. NLTK Twitter Samples corpus

Produces a unified, balanced DataFrame with columns: text, sentiment
"""

import os
import re
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords, movie_reviews, twitter_samples
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

# ---------------------------------------------------------------------------
# Ensure NLTK data is available
# ---------------------------------------------------------------------------

NLTK_PACKAGES = [
    "stopwords", "wordnet", "punkt", "punkt_tab",
    "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng",
    "movie_reviews", "twitter_samples",
    "vader_lexicon", "omw-1.4",
]


def ensure_nltk_data():
    """Download all required NLTK data packages (idempotent)."""
    for pkg in NLTK_PACKAGES:
        nltk.download(pkg, quiet=True)


# ---------------------------------------------------------------------------
# Text cleaning
# ---------------------------------------------------------------------------

_lemmatizer = WordNetLemmatizer()


def clean_text(text: str) -> str:
    """Normalise a single text string for model training."""
    if not isinstance(text, str):
        return ""
    # Remove URLs
    text = re.sub(r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE)
    # Remove @mentions and #hashtags
    text = re.sub(r"@\w+|#\w+", "", text)
    # Remove HTML entities
    text = re.sub(r"&\w+;", "", text)
    # Keep only letters and spaces
    text = re.sub(r"[^a-zA-Z\s]", " ", text)
    # Remove single characters
    text = re.sub(r"\s+[a-zA-Z]\s+", " ", text)
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    # Lowercase
    text = text.lower()
    # Tokenize → lemmatize → remove stopwords
    stop_words = set(stopwords.words("english"))
    tokens = word_tokenize(text)
    tokens = [
        _lemmatizer.lemmatize(tok)
        for tok in tokens
        if tok not in stop_words and len(tok) > 2
    ]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# Dataset loaders
# ---------------------------------------------------------------------------


def load_airline_dataset(csv_path: str = "dataset.csv") -> pd.DataFrame:
    """Load the Twitter US Airline Sentiment CSV."""
    # Try absolute path first, then relative
    if not os.path.isabs(csv_path):
        base = os.path.dirname(os.path.abspath(__file__))
        csv_path = os.path.join(base, csv_path)

    df = pd.read_csv(csv_path)
    df = df[["text", "airline_sentiment"]].copy()
    df.columns = ["text", "sentiment"]
    # Standardise labels
    df["sentiment"] = df["sentiment"].str.lower().str.strip()
    df = df.dropna(subset=["text", "sentiment"])
    return df


def load_movie_reviews() -> pd.DataFrame:
    """Load NLTK movie_reviews corpus (binary: pos / neg)."""
    rows = []
    for category in movie_reviews.categories():  # 'pos', 'neg'
        for fileid in movie_reviews.fileids(category):
            text = movie_reviews.raw(fileid)
            sentiment = "positive" if category == "pos" else "negative"
            rows.append({"text": text, "sentiment": sentiment})
    return pd.DataFrame(rows)


def load_twitter_samples() -> pd.DataFrame:
    """Load NLTK twitter_samples corpus (binary: positive / negative)."""
    pos_tweets = twitter_samples.strings("positive_tweets.json")
    neg_tweets = twitter_samples.strings("negative_tweets.json")
    rows = [{"text": t, "sentiment": "positive"} for t in pos_tweets]
    rows += [{"text": t, "sentiment": "negative"} for t in neg_tweets]
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

LABEL_MAP = {"positive": 0, "negative": 1, "neutral": 2}
LABEL_NAMES = {v: k for k, v in LABEL_MAP.items()}


def load_and_prepare(balance: bool = True, verbose: bool = True):
    """
    Load all datasets, clean, and optionally balance.

    Returns
    -------
    texts : list[str]       – cleaned text strings
    labels : np.ndarray     – integer-encoded labels (0=pos, 1=neg, 2=neu)
    """
    ensure_nltk_data()

    if verbose:
        print("[*] Loading datasets ...")

    df_airline = load_airline_dataset()
    df_movies = load_movie_reviews()
    df_twitter = load_twitter_samples()

    if verbose:
        print(f"   Airline : {len(df_airline):>6} rows")
        print(f"   Movies  : {len(df_movies):>6} rows")
        print(f"   Twitter : {len(df_twitter):>6} rows")

    # Combine
    df = pd.concat([df_airline, df_movies, df_twitter], ignore_index=True)
    df = df.dropna(subset=["text", "sentiment"])

    if verbose:
        print(f"   Combined: {len(df):>6} rows")
        print("\n[*] Cleaning text ...")

    # Clean
    df["text"] = df["text"].apply(clean_text)
    # Drop empty after cleaning
    df = df[df["text"].str.len() > 0].reset_index(drop=True)

    if verbose:
        print(f"   After clean: {len(df)} rows")
        print("\n[*] Label distribution (before balancing):")
        print(df["sentiment"].value_counts().to_string())

    # Encode labels
    df["label"] = df["sentiment"].map(LABEL_MAP)

    # Balance via oversampling minority classes to match majority
    if balance:
        max_count = df["label"].value_counts().max()
        balanced_parts = []
        for lbl in df["label"].unique():
            subset = df[df["label"] == lbl]
            if len(subset) < max_count:
                # Oversample: duplicate minority samples to match majority
                oversampled = subset.sample(n=max_count, replace=True, random_state=42)
                balanced_parts.append(oversampled)
            else:
                balanced_parts.append(subset)
        df = pd.concat(balanced_parts, ignore_index=True)
        # Shuffle
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        if verbose:
            print(f"\n[*] After balancing ({max_count} per class via oversampling):")
            print(df["sentiment"].value_counts().to_string())

    texts = df["text"].tolist()
    labels = df["label"].values

    if verbose:
        print(f"\n[OK] Final dataset: {len(texts)} samples")

    return texts, labels


if __name__ == "__main__":
    texts, labels = load_and_prepare(verbose=True)
    print(f"\nSample text: {texts[0][:100]}...")
    print(f"Sample label: {LABEL_NAMES[labels[0]]}")
