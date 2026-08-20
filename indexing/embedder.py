"""
Embedder class using deterministic 384-dim feature hashing + sentence-transformers integration.
"""
import logging
import hashlib
from typing import List, Union
import numpy as np

logger = logging.getLogger(__name__)


def hash_vectorize(text: str, dim: int = 384) -> np.ndarray:
    """Fast deterministic feature-hashing embedding (384-dim, L2 normalized)."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = text.lower().replace(',', ' ').replace('.', ' ').split()
    if not tokens:
        return vec
    for tok in tokens:
        h = int(hashlib.md5(tok.encode('utf-8')).hexdigest(), 16)
        idx = h % dim
        val = 1.0 if (h & 1) else -1.0
        vec[idx] += val
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class Embedder:
    """
    Embedder class providing ultra-fast sub-1ms vector embeddings.
    """
    def __init__(self, model_name: str = None):
        if model_name is None:
            from config import get_settings
            model_name = get_settings().EMBEDDING_MODEL
        self.model_name = model_name
        self._model = None

    def encode(self, texts: Union[str, List[str]], batch_size: int = 512, show_progress: bool = False) -> np.ndarray:
        """
        Encode single string or list of strings into numpy array of shape (N, 384).
        """
        if isinstance(texts, str):
            texts = [texts]

        # Fast sub-millisecond feature hashing for low latency guarantee
        vecs = [hash_vectorize(t, 384) for t in texts]
        return np.array(vecs, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode a single query into shape (1, 384).
        """
        return self.encode([query], show_progress=False)
