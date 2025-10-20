from typing import Iterable, Tuple

import numpy as np
import pandas as pd
import gensim.downloader as api


def load_glove(model_name: str = "glove-twitter-200"):
    return api.load(model_name)


def average_embedding(text: str, model, vector_size: int = 200) -> np.ndarray:
    words = (text or "").split()
    word_vectors = [model[word] for word in words if word in model]
    if not word_vectors:
        return np.zeros(vector_size, dtype=float)
    return np.mean(word_vectors, axis=0)


def embed_dataframe(df: pd.DataFrame, text_col: str, model, vector_size: int = 200) -> pd.DataFrame:
    vectors = np.vstack([average_embedding(t, model, vector_size) for t in df[text_col]])
    cols = [f"emb_{i}" for i in range(vector_size)]
    return pd.DataFrame(vectors, columns=cols, index=df.index)


def aggregate_periods(df_with_vectors: pd.DataFrame) -> pd.DataFrame:
    # Expecting MatchID, PeriodID, ID and embedding columns
    return (
        df_with_vectors
        .groupby(["MatchID", "PeriodID", "ID"], as_index=False)
        .mean()
    )


