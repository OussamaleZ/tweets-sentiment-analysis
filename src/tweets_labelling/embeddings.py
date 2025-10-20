from typing import Iterable, Tuple, Optional

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


def embed_dataframe_glove(df: pd.DataFrame, text_col: str, model, vector_size: int = 200) -> pd.DataFrame:
    vectors = np.vstack([average_embedding(t, model, vector_size) for t in df[text_col]])
    cols = [f"emb_{i}" for i in range(vector_size)]
    return pd.DataFrame(vectors, columns=cols, index=df.index)


def load_sentence_transformer(model_name: str = "all-MiniLM-L6-v2", device: Optional[str] = None):
    from sentence_transformers import SentenceTransformer
    st_model = SentenceTransformer(model_name, device=device)
    return st_model


def embed_dataframe_sbert(df: pd.DataFrame, text_col: str, st_model, batch_size: int = 32) -> pd.DataFrame:
    texts = df[text_col].astype(str).tolist()
    vectors = st_model.encode(texts, show_progress_bar=True, batch_size=batch_size, convert_to_tensor=False)
    vectors = np.asarray(vectors)
    cols = [f"emb_{i}" for i in range(vectors.shape[1])]
    return pd.DataFrame(vectors, columns=cols, index=df.index)


def aggregate_periods(df_with_vectors: pd.DataFrame) -> pd.DataFrame:
    # Expecting MatchID, PeriodID, ID and embedding columns
    return (
        df_with_vectors
        .groupby(["MatchID", "PeriodID", "ID"], as_index=False)
        .mean()
    )


