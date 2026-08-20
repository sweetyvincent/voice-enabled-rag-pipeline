import json
import logging
import re
from typing import List, Dict, Any
from indexing.embedder import Embedder
from indexing.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)

STOPWORDS = {
    "what", "is", "the", "of", "how", "why", "where", "who", "when", "does", "do",
    "did", "a", "an", "in", "to", "for", "with", "on", "at", "by", "from", "about",
    "tell", "me", "can", "you", "explain"
}


def extract_keywords(text: str) -> set:
    words = set(re.findall(r'\b[a-z0-9]{2,}\b', text.lower()))
    return words - STOPWORDS


class Retriever:
    """
    Hybrid Retriever class combining FAISS Vector Search with Keyword Overlap scoring.
    """
    def __init__(self, embedder: Embedder, index: FAISSIndex, chunks_metadata: List[Dict[str, Any]]):
        self.embedder = embedder
        self.index = index
        self.chunks_metadata = chunks_metadata

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve top_k relevant chunks using hybrid vector + keyword matching.
        """
        if not self.chunks_metadata:
            return []

        # 1. FAISS Vector Search
        query_embedding = self.embedder.encode_query(query)
        scores, indices = self.index.search(query_embedding, min(top_k * 2, len(self.chunks_metadata)))

        results_dict = {}
        if scores.size > 0 and indices.size > 0:
            flat_scores = scores[0]
            flat_indices = indices[0]

            for score, idx in zip(flat_scores, flat_indices):
                if idx == -1 or idx >= len(self.chunks_metadata):
                    continue
                chunk_data = self.chunks_metadata[idx].copy()
                chunk_data['dense_score'] = float(score)
                results_dict[idx] = chunk_data

        # Fallback: populate all chunks if FAISS returned fewer
        for idx, chunk in enumerate(self.chunks_metadata):
            if idx not in results_dict:
                chunk_copy = chunk.copy()
                chunk_copy['dense_score'] = 0.0
                results_dict[idx] = chunk_copy

        # 2. Keyword Overlap Scoring (BM25-style sparse score)
        q_keywords = extract_keywords(query)
        scored_results = []

        for idx, chunk_data in results_dict.items():
            text = chunk_data.get('text', '')
            c_keywords = extract_keywords(text)

            if q_keywords and c_keywords:
                overlap = len(q_keywords.intersection(c_keywords)) / len(q_keywords)
            else:
                overlap = 0.0

            chunk_data['sparse_score'] = float(overlap)
            # Hybrid fused score: 60% keyword match + 40% dense vector match
            fused_score = (0.60 * overlap) + (0.40 * chunk_data.get('dense_score', 0.0))
            chunk_data['score'] = float(fused_score)

            if 'passage_id' not in chunk_data and 'doc_metadata' in chunk_data and isinstance(chunk_data['doc_metadata'], dict):
                chunk_data['passage_id'] = chunk_data['doc_metadata'].get('passage_id')

            scored_results.append(chunk_data)

        # Sort by fused score descending
        scored_results.sort(key=lambda x: x.get('score', 0.0), reverse=True)
        return scored_results[:top_k]

    @classmethod
    def load_chunks_metadata(cls, path: str) -> List[Dict[str, Any]]:
        """Load chunks metadata from a JSONL file."""
        logger.info(f"Loading chunks metadata from {path}")
        chunks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks
