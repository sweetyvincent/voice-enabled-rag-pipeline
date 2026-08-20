import re
from typing import List, Dict, Any, Optional
from chunking.base import Chunk, ChunkingStrategy, simple_sentence_split, simple_tokenize

class HybridRecursiveChunking(ChunkingStrategy):
    name = "hybrid_recursive"

    def __init__(self, p_limit: int = 400, s_limit: int = 100):
        self.p_limit = p_limit
        self.s_limit = s_limit
        self.clause_markers = ['; ', ' — ', ', and ', ', but ', ', or ', ', which ', ', that ']

    def split_by_clauses(self, sentence: str) -> List[str]:
        pattern = '(' + '|'.join(map(re.escape, self.clause_markers)) + ')'
        parts = re.split(pattern, sentence)
        res = []
        temp = ""
        for i, part in enumerate(parts):
            temp += part
            if part in self.clause_markers or i == len(parts) - 1:
                if temp.strip():
                    res.append(temp.strip())
                temp = ""
        return res

    def chunk(self, passage: str, passage_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        metadata = metadata or {}
        paragraphs = [p.strip() for p in passage.split('\n\n') if p.strip()]
        
        raw_chunks = []
        
        for p in paragraphs:
            if len(simple_tokenize(p)) <= self.p_limit:
                raw_chunks.append((p, 1))
            else:
                sentences = simple_sentence_split(p)
                for s in sentences:
                    if len(simple_tokenize(s)) <= self.s_limit:
                        raw_chunks.append((s, 2))
                    else:
                        clauses = self.split_by_clauses(s)
                        for c in clauses:
                            raw_chunks.append((c, 3))

        chunks_text_depth = []
        prev_sentence = ""
        
        for text, depth in raw_chunks:
            combined = (prev_sentence + " " + text).strip() if prev_sentence else text
            chunks_text_depth.append((combined, depth))
            
            sentences_in_text = simple_sentence_split(text)
            if sentences_in_text:
                prev_sentence = sentences_in_text[-1]
            else:
                prev_sentence = text

        total_chunks = len(chunks_text_depth)
        chunks = []
        for i, (text, depth) in enumerate(chunks_text_depth):
            chunk_id = f"{self.name}_{passage_id}_{i}"
            meta = metadata.copy()
            meta['hierarchy_depth'] = depth
            chunks.append(Chunk(
                text=text,
                chunk_id=chunk_id,
                passage_id=passage_id,
                strategy=self.name,
                chunk_index=i,
                total_chunks=total_chunks,
                metadata=meta
            ))
        return chunks
