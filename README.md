## Tweets Sentiment/Event Classification

A clean, modular repository for preprocessing tweets, generating embeddings, and training models (XGBoost, LSTM) for event/sentiment classification.

### Project Structure

```
.
├─ src/
│  └─ tweets_labelling/
│     ├─ __init__.py
│     ├─ preprocessing.py
│     └─ embeddings.py
├─ scripts/
│  ├─ prepare_dataset.py
│  ├─ train_xgboost.py
│  └─ train_lstm.py
├─ data/
│  ├─ raw/                  # put raw CSVs here (e.g., challenge_data/*)
│  └─ processed/            # generated datasets (pickles, features)
├─ experiments/legacy/      # original scripts kept for reference
├─ README.md
├─ requirements.txt
├─ .gitignore
└─ LICENSE
```

### Quickstart

1) Install dependencies

```bash
pip install -r requirements.txt
```

2) Place your raw data

- Copy your training CSVs into `data/raw/train_tweets/`
- Copy your evaluation CSVs into `data/raw/eval_tweets/`

3) Prepare dataset (preprocess, embed, aggregate)

```bash
python scripts/prepare_dataset.py --train-dir data/raw/train_tweets --out-dir data/processed
```

This creates `X.pkl`, `y.pkl`, `X_train.pkl`, `X_test.pkl`, `y_train.pkl`, `y_test.pkl` in `data/processed/`.

4) Train models

```bash
python scripts/train_xgboost.py --data-dir data/processed --out-file xgb.csv
python scripts/train_lstm.py --data-dir data/processed
```

### Notes

- Reusable preprocessing lives in `src/tweets_labelling/preprocessing.py`.
- Embedding utilities live in `src/tweets_labelling/embeddings.py`.
- Original one-off scripts are retained under `experiments/legacy/` for transparency.

### License

Released under the MIT License. See `LICENSE`.


