import os
import re
from typing import Iterable, Optional

import nltk
import pandas as pd


def ensure_nltk() -> None:
    # Lazily download required resources; no-op if present
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords")
    try:
        nltk.data.find("corpora/wordnet")
    except LookupError:
        nltk.download("wordnet")


USERNAME_REGEX = re.compile(r"@\w+")
HASHTAG_REGEX = re.compile(r"#\w+")
URL_REGEX = re.compile(r"http\S+|www\S+")
PUNCTUATION_REGEX = re.compile(r"[^\w\s]")
ELONGATED_REGEX = re.compile(r"(.)\1{2,}")
MULTISPACE_REGEX = re.compile(r" +")


def extract_username(tweet: str) -> Optional[str]:
    match = USERNAME_REGEX.search(tweet or "")
    return match.group(0) if match else None


def normalize_text(text: str) -> str:
    text = str(text).lower()
    text = HASHTAG_REGEX.sub("", text)
    text = USERNAME_REGEX.sub("", text)
    text = URL_REGEX.sub("", text)
    text = PUNCTUATION_REGEX.sub(" ", text)
    text = ELONGATED_REGEX.sub(r"\1\1", text)
    text = MULTISPACE_REGEX.sub(" ", text)
    text = text.replace(" vs ", " ").replace("\tim\t", " i am ")
    return text.strip()


def preprocess_dataframe(df: pd.DataFrame, tweet_col: str = "Tweet", drop_spammers_threshold: int = 20) -> pd.DataFrame:
    # Lowercase early for username extraction consistency
    df = df.copy()
    df[tweet_col] = df[tweet_col].astype(str).str.lower()

    # Remove spammers based on frequency of mentioned usernames
    df["__username"] = df[tweet_col].apply(extract_username)
    username_counts = df["__username"].value_counts()
    spammer_usernames = username_counts[username_counts > drop_spammers_threshold].index
    df = df[~df["__username"].isin(spammer_usernames)]
    df = df.drop(columns=["__username"])  # internal column

    # Normalize text
    df[tweet_col] = df[tweet_col].apply(normalize_text)
    return df


def read_many_csvs(directory: str) -> pd.DataFrame:
    frames = []
    for filename in os.listdir(directory):
        if not filename.lower().endswith(".csv"):
            continue
        path = os.path.join(directory, filename)
        frames.append(pd.read_csv(path))
    if not frames:
        raise FileNotFoundError(f"No CSV files found in {directory}")
    return pd.concat(frames, ignore_index=True)


