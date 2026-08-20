import json
import logging
from typing import List, Dict, Any
from indexing.embedder import Embedder
from indexing.faiss_index import FAISSIndex

logger = logging.getLogger(__name__)

class Retriever:
    """
    Retriever class for semantic search using FAISS.
    """
    def __init__(self, embedder: Embedder, index: FAISSIndex, chunks_metadata: List[Dict[str, Any]]):
        self.embedder = embedder
        self.index = index
        self.chunks_metadata = chunks_metadata

    def retrieve(self, query: str, top_k: int = 20) -> List[Dict[str, Any]]:
        """
        Retrieve top_k relevant chunks for a given query.
        """
        query_embedding = self.embedder.encode_query(query)
        scores, indices = self.index.search(query_embedding, top_k)
        
        results = []
        if scores.size == 0 or indices.size == 0:
            return results

        # Flatten arrays
        scores = scores[0]
        indices = indices[0]

        for score, idx in zip(scores, indices):
            if idx == -1 or idx >= len(self.chunks_metadata):
                continue
            
            chunk_data = self.chunks_metadata[idx].copy()
            chunk_data['score'] = float(score)
            
            # Extract passage_id from doc_metadata if not present at root level
            if 'passage_id' not in chunk_data and 'doc_metadata' in chunk_data and isinstance(chunk_data['doc_metadata'], dict):
                chunk_data['passage_id'] = chunk_data['doc_metadata'].get('passage_id')
                
            results.append(chunk_data)

        return results

    @classmethod
    def load_chunks_metadata(cls, path: str) -> List[Dict[str, Any]]:
        """
        Load chunks metadata from a JSONL file.
        """
        logger.info(f"Loading chunks metadata from {path}")
        chunks = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    chunks.append(json.loads(line))
        return chunks
