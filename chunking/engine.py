import logging
from typing import List, Dict, Any, Optional
from chunking.base import Chunk, simple_tokenize
from chunking.fixed_size import FixedSizeChunking
from chunking.semantic import SemanticChunking
from chunking.passage_aware import PassageAwareChunking
from chunking.hybrid_recursive import HybridRecursiveChunking

try:
    from config import get_settings
except ImportError:
    # Dummy fallback if config doesn't exist yet
    def get_settings():
        class Settings:
            pass
        return Settings()

logger = logging.getLogger(__name__)

class ChunkingEngine:
    def __init__(self):
        settings = get_settings()
        self.strategies = [
            SemanticChunking(),
            PassageAwareChunking(),
            HybridRecursiveChunking(),
            FixedSizeChunking()
        ]
        self.priority = {
            "semantic": 4,
            "passage_aware": 3,
            "hybrid_recursive": 2,
            "fixed_size": 1
        }

    def chunk_passage(self, passage: str, passage_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        all_chunks = []
        for strategy in self.strategies:
            try:
                chunks = strategy.chunk(passage, passage_id, metadata)
                all_chunks.extend(chunks)
            except Exception as e:
                logger.error(f"Error in strategy {strategy.name} for passage {passage_id}: {e}")
                
        return self._deduplicate(all_chunks)

    def chunk_passages(self, passages: List[Dict[str, Any]]) -> List[Chunk]:
        logger.info(f"Batch chunking {len(passages)} passages...")
        all_chunks = []
        for p in passages:
            passage_text = p.get('text', '')
            passage_id = p.get('id', 'unknown')
            metadata = p.get('metadata', {})
            chunks = self.chunk_passage(passage_text, passage_id, metadata)
            all_chunks.extend(chunks)
        logger.info(f"Generated {len(all_chunks)} chunks total.")
        return all_chunks

    def _deduplicate(self, chunks: List[Chunk]) -> List[Chunk]:
        deduplicated = []
        for chunk in chunks:
            is_duplicate = False
            chunk_tokens = set(simple_tokenize(chunk.text.lower()))
            if not chunk_tokens:
                continue

            for existing in deduplicated:
                existing_tokens = set(simple_tokenize(existing.text.lower()))
                intersection = len(chunk_tokens.intersection(existing_tokens))
                union = len(chunk_tokens.union(existing_tokens))
                jaccard = intersection / union if union > 0 else 0

                if jaccard > 0.85:
                    is_duplicate = True
                    curr_priority = self.priority.get(chunk.strategy, 0)
                    exist_priority = self.priority.get(existing.strategy, 0)
                    if curr_priority > exist_priority:
                        deduplicated.remove(existing)
                        deduplicated.append(chunk)
                    break
            
            if not is_duplicate:
                deduplicated.append(chunk)

        return deduplicated
