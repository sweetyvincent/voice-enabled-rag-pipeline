import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

@dataclass
class Chunk:
    text: str
    chunk_id: str
    passage_id: str
    strategy: str
    chunk_index: int
    total_chunks: int
    metadata: Dict[str, Any] = field(default_factory=dict)

class ChunkingStrategy(ABC):
    name: str

    @abstractmethod
    def chunk(self, passage: str, passage_id: str, metadata: Optional[Dict[str, Any]] = None) -> List[Chunk]:
        pass

def simple_tokenize(text: str) -> List[str]:
    """Fast approximation of token count by splitting on whitespace."""
    return text.split()

def simple_sentence_split(text: str) -> List[str]:
    """Splits text into sentences using regex."""
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])', text.strip())
    return [s.strip() for s in sentences if s.strip()]
