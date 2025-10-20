import argparse
import os
import pickle

import numpy as np
import pandas as pd

from tweets_labelling.preprocessing import preprocess_dataframe, read_many_csvs
from tweets_labelling.embeddings import load_glove, embed_dataframe, aggregate_periods


def main():
    parser = argparse.ArgumentParser(description="Prepare dataset: preprocess, embed, aggregate, split, and save pickles")
    parser.add_argument("--train-dir", required=True, help="Directory containing training CSVs (Tweet, Timestamp, MatchID, PeriodID, ID, EventType)")
    parser.add_argument("--out-dir", required=True, help="Directory to write processed pickles")
    parser.add_argument("--vector-size", type=int, default=200, help="Embedding size for glove-twitter-200")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Read and preprocess
    df = read_many_csvs(args.train_dir)
    df = preprocess_dataframe(df, tweet_col="Tweet")

    # Embed
    model = load_glove("glove-twitter-200")
    vectors = embed_dataframe(df, text_col="Tweet", model=model, vector_size=args.vector_size)
    df_feat = pd.concat([df.reset_index(drop=True), vectors.reset_index(drop=True)], axis=1)

    # Drop unused raw columns and aggregate per period
    drop_cols = [c for c in ["Timestamp", "Tweet"] if c in df_feat.columns]
    period_features = df_feat.drop(columns=drop_cols)
    period_features = aggregate_periods(period_features)

    # Build X, y
    X = period_features.drop(columns=["EventType", "MatchID", "PeriodID", "ID"]).values
    y = period_features["EventType"].values

    # Train/test split
    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Save pickles
    with open(os.path.join(args.out_dir, "X.pkl"), "wb") as f:
        pickle.dump(X, f)
    with open(os.path.join(args.out_dir, "y.pkl"), "wb") as f:
        pickle.dump(y, f)
    with open(os.path.join(args.out_dir, "X_train.pkl"), "wb") as f:
        pickle.dump(X_train, f)
    with open(os.path.join(args.out_dir, "X_test.pkl"), "wb") as f:
        pickle.dump(X_test, f)
    with open(os.path.join(args.out_dir, "y_train.pkl"), "wb") as f:
        pickle.dump(y_train, f)
    with open(os.path.join(args.out_dir, "y_test.pkl"), "wb") as f:
        pickle.dump(y_test, f)

    print("Saved processed datasets to:", args.out_dir)


if __name__ == "__main__":
    main()


