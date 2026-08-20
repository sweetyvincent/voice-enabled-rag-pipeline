from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum
import time

class StageStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"

class StageTrace(BaseModel):
    name: str
    status: StageStatus = StageStatus.PENDING
    latency_ms: float = 0.0
    error: Optional[str] = None
    details: dict = {}

class RetrievedChunk(BaseModel):
    text: str
    chunk_id: str
    strategy: str
    score: float
    passage_id: str
    metadata: dict = {}

class PipelineRequest(BaseModel):
    text: Optional[str] = None  # text query
    audio_bytes: Optional[bytes] = None  # raw audio bytes
    session_id: Optional[str] = None

class PipelineResult(BaseModel):
    transcript: Optional[str] = None  # STT output (if audio input)
    query: str  # final query text (transcript or direct text)
    retrieved_chunks: List[RetrievedChunk] = []
    answer: Optional[str] = None
    guardrail_verdict: str = "pass"  # pass, refuse_off_topic, refuse_unsafe, etc.
    guardrail_reason: Optional[str] = None
    stages: List[StageTrace] = []
    total_latency_ms: float = 0.0
    stt_latency_ms: float = 0.0  # separate tracking for STT
    retrieval_latency_ms: float = 0.0  # chunking+embedding+FAISS+rerank
    generation_latency_ms: float = 0.0  # LLM call
    error: Optional[str] = None
    success: bool = True
