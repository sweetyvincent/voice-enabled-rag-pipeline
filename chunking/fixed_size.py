from typing import List, Dict, Any, Optional
from chunking.base import Chunk, ChunkingStrategy, simple_tokenize

class FixedSizeChunking(ChunkingStrategy):
    name = "fixed_size"

    def __init__(self, chunk_size: int = 256, overlap: int = 64):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, passage: str, passage_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        metadata = metadata or {}
        tokens = simple_tokenize(passage)
        if not tokens:
            return []

        step = max(1, self.chunk_size - self.overlap)
        token_chunks = []
        for i in range(0, len(tokens), step):
            token_chunks.append(tokens[i:i + self.chunk_size])

        total_chunks = len(token_chunks)
        chunks = []
        for i, t_chunk in enumerate(token_chunks):
            chunk_text = " ".join(t_chunk)
            chunk_id = f"{self.name}_{passage_id}_{i}"
            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=chunk_id,
                passage_id=passage_id,
                strategy=self.name,
                chunk_index=i,
                total_chunks=total_chunks,
                metadata=metadata.copy()
            ))
        return chunks
