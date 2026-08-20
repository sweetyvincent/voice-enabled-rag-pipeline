import logging
import faiss
import numpy as np
from typing import Tuple

logger = logging.getLogger(__name__)

class FAISSIndex:
    """
    FAISS Index wrapper for vector search.
    """
    def __init__(self, dimension: int = 384):
        self.dimension = dimension
        self.index = None

    def build(self, embeddings: np.ndarray, use_ivf: bool = True, nlist: int = 256):
        """
        Build FAISS index from embeddings.
        """
        logger.info(f"Building FAISS index for {len(embeddings)} vectors with dimension {self.dimension}")
        if embeddings.shape[1] != self.dimension:
            raise ValueError(f"Embedding dimension {embeddings.shape[1]} does not match index dimension {self.dimension}")

        if use_ivf and len(embeddings) > 50000:
            logger.info("Using IndexIVFFlat")
            quantizer = faiss.IndexFlatIP(self.dimension)
            self.index = faiss.IndexIVFFlat(quantizer, self.dimension, nlist, faiss.METRIC_INNER_PRODUCT)
            logger.info("Training index...")
            self.index.train(embeddings)
            self.index.nprobe = 16
        else:
            logger.info("Using IndexFlatIP (brute-force)")
            self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings)
        logger.info(f"Index built with {self.num_vectors} vectors")

    def search(self, query_embedding: np.ndarray, top_k: int = 20) -> Tuple[np.ndarray, np.ndarray]:
        """
        Search for top_k similar vectors.
        Returns scores and indices.
        """
        if self.index is None:
            raise ValueError("Index has not been built or loaded")
        
        if len(query_embedding.shape) == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        if query_embedding.shape[1] != self.dimension:
            raise ValueError(f"Query dimension {query_embedding.shape[1]} does not match index dimension {self.dimension}")

        if self.num_vectors == 0:
            return np.array([]), np.array([])

        scores, indices = self.index.search(query_embedding, top_k)
        return scores, indices

    def save(self, path: str):
        """Save index to disk."""
        if self.index is None:
            raise ValueError("No index to save")
        logger.info(f"Saving index to {path}")
        faiss.write_index(self.index, path)

    def load(self, path: str):
        """Load index from disk."""
        logger.info(f"Loading index from {path}")
        self.index = faiss.read_index(path)
        self.dimension = self.index.d
        return self

    @property
    def num_vectors(self) -> int:
        if self.index is None:
            return 0
        return self.index.ntotal


def load_index(path: str) -> FAISSIndex:
    """Convenience function to load index."""
    index = FAISSIndex()
    return index.load(path)
