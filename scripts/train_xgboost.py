import argparse
import os
import pickle

import pandas as pd
from xgboost import XGBClassifier

from tweets_labelling.preprocessing import preprocess_dataframe, read_many_csvs
from tweets_labelling.embeddings import load_glove, embed_dataframe, aggregate_periods


def main():
    parser = argparse.ArgumentParser(description="Train XGBoost and optionally generate predictions for eval set")
    parser.add_argument("--data-dir", required=True, help="Directory containing processed pickles (X.pkl, y.pkl, X_train.pkl, ...)")
    parser.add_argument("--eval-dir", default=None, help="Optional directory with evaluation CSVs to predict")
    parser.add_argument("--out-file", default="xgb.csv", help="Output CSV filename for predictions")
    args = parser.parse_args()

    # Load processed data
    with open(os.path.join(args.data_dir, "X_train.pkl"), "rb") as f:
        X_train = pickle.load(f)
    with open(os.path.join(args.data_dir, "y_train.pkl"), "rb") as f:
        y_train = pickle.load(f)
    with open(os.path.join(args.data_dir, "X_test.pkl"), "rb") as f:
        X_test = pickle.load(f)
    with open(os.path.join(args.data_dir, "y_test.pkl"), "rb") as f:
        y_test = pickle.load(f)

    # Train model
    model = XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6, subsample=0.8, colsample_bytree=0.8, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    score = model.score(X_test, y_test)
    print("Validation accuracy:", score)

    # Optional predictions on eval CSVs
    if args.eval_dir:
        eval_df = read_many_csvs(args.eval_dir)
        eval_df = preprocess_dataframe(eval_df, tweet_col="Tweet")
        glove = load_glove("glove-twitter-200")
        vectors = embed_dataframe(eval_df, text_col="Tweet", model=glove, vector_size=200)
        df_feat = pd.concat([eval_df.reset_index(drop=True), vectors.reset_index(drop=True)], axis=1)

        drop_cols = [c for c in ["Timestamp", "Tweet"] if c in df_feat.columns]
        period_features = df_feat.drop(columns=drop_cols)
        period_features = aggregate_periods(period_features)
        X_eval = period_features.drop(columns=["MatchID", "PeriodID", "ID"]).values

        preds = model.predict(X_eval)
        out = period_features[["ID"]].copy()
        out["EventType"] = preds
        out.to_csv(args.out_file, index=False)
        print("Wrote:", args.out_file)


if __name__ == "__main__":
    main()


