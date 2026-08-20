from typing import List, Dict, Any, Optional
from chunking.base import Chunk, ChunkingStrategy, simple_sentence_split, simple_tokenize

class SemanticChunking(ChunkingStrategy):
    name = "semantic"

    def __init__(self, soft_limit: int = 200):
        self.soft_limit = soft_limit

    def chunk(self, passage: str, passage_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        metadata = metadata or {}
        sentences = simple_sentence_split(passage)
        if not sentences:
            return []

        chunks_text = []
        current_chunk = []
        current_tokens = 0

        for sentence in sentences:
            sentence_tokens = len(simple_tokenize(sentence))
            if current_tokens + sentence_tokens > self.soft_limit and current_chunk:
                chunks_text.append(" ".join(current_chunk))
                current_chunk = [sentence]
                current_tokens = sentence_tokens
            else:
                current_chunk.append(sentence)
                current_tokens += sentence_tokens

        if current_chunk:
            chunks_text.append(" ".join(current_chunk))

        total_chunks = len(chunks_text)
        chunks = []
        for i, text in enumerate(chunks_text):
            chunk_id = f"{self.name}_{passage_id}_{i}"
            chunks.append(Chunk(
                text=text,
                chunk_id=chunk_id,
                passage_id=passage_id,
                strategy=self.name,
                chunk_index=i,
                total_chunks=total_chunks,
                metadata=metadata.copy()
            ))
        return chunks
