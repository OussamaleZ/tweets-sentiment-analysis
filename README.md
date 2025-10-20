## Tweets Sentiments Analysis

This project was developed as part of the Machine and Deep Learning course at École Polytechnique. It served as our submission for a Kaggle challenge focused on sentiment classification of tweets.

This repository implements the pipeline explored in `final_notebook.ipynb`: preprocessing tweets, creating dense text embeddings with either GloVe averages or Sentence-Transformers, then training classic ML models (Logistic Regression, Linear SVM, Decision Tree, Random Forest, XGBoost, MLP). The best-performing baselines in the notebook use averaged GloVe or MiniLM embeddings with linear models or tree ensembles.

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
├─ experiments/    
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
# XGBoost (as in the notebook experiments)
python scripts/train_xgboost.py --data-dir data/processed --out-file xgb.csv

# Optional: simple LSTM on period-level vectors (illustrative)
python scripts/train_lstm.py --data-dir data/processed
```

### Methods reflected from `final_notebook.ipynb`

- Embeddings
  - GloVe via Gensim (`glove-twitter-200`) with average pooling per tweet
  - Sentence-Transformers (`all-MiniLM-L6-v2`) to encode tweets, then average per period
- Models and selection
  - Logistic Regression (GridSearch over C, solvers)
  - Linear/Kernel SVM (GridSearch over C, kernel, gamma)
  - Decision Tree (GridSearch over depth/criteria)
  - Random Forest (GridSearch over n_estimators, max_features, etc.)
  - XGBoost (GridSearch over depth, learning_rate, subsample, colsample)
  - MLP classifier (hidden sizes, activation, alpha)
- Data aggregation
  - Drop `Timestamp`, embed `Tweet`, group by `MatchID`, `PeriodID`, `ID`, then split train/test

To reproduce the notebook’s GloVe path, you can pre-download and cache the KeyedVectors and point the scripts to `data/raw/*`. The scripts default to GloVe averaging for speed and reproducibility; Sentence-Transformers usage is shown in the notebook for comparison.

### Notes

- Reusable preprocessing lives in `src/tweets_labelling/preprocessing.py`.
- Embedding utilities live in `src/tweets_labelling/embeddings.py`.
- Original exploratory workflows and extended grid searches are documented in `final_notebook.ipynb`.



