from typing import List, Dict, Any, Optional
from chunking.base import Chunk, ChunkingStrategy, simple_sentence_split, simple_tokenize

class PassageAwareChunking(ChunkingStrategy):
    name = "passage_aware"

    def __init__(self, max_tokens: int = 300):
        self.max_tokens = max_tokens

    def chunk(self, passage: str, passage_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        metadata = metadata or {}
        query_type = metadata.get('query_type', 'unknown')
        header = f"[QueryType: {query_type}] "
        tokens = simple_tokenize(passage)

        chunks_text = []
        if len(tokens) <= self.max_tokens:
            chunks_text.append(header + passage)
        else:
            sentences = simple_sentence_split(passage)
            current_chunk = []
            current_tokens = 0
            
            for sentence in sentences:
                sentence_tokens = len(simple_tokenize(sentence))
                if current_tokens + sentence_tokens > self.max_tokens and current_chunk:
                    chunks_text.append(header + " ".join(current_chunk))
                    current_chunk = [sentence]
                    current_tokens = sentence_tokens
                else:
                    current_chunk.append(sentence)
                    current_tokens += sentence_tokens
            if current_chunk:
                chunks_text.append(header + " ".join(current_chunk))

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
