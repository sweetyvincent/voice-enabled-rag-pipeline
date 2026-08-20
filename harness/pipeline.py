"""
RAG Pipeline Harness — structured orchestration around the full RAG pipeline.
"""
import time
import logging
import asyncio
import os
import json

import requests as req_lib

from config import get_settings
from harness.models import (
    PipelineRequest, PipelineResult, StageTrace, StageStatus, RetrievedChunk
)
from harness.error_handler import (
    CircuitBreaker, circuit_breaker_context,
    handle_llm_failure, handle_stt_failure, handle_retrieval_failure
)

try:
    from analytics.latency_tracker import LatencyTracker
except ImportError:
    LatencyTracker = None

from indexing.embedder import Embedder
from indexing.faiss_index import FAISSIndex
from retrieval.retriever import Retriever
from retrieval.reranker import Reranker
from generation.llm_client import GroqLLMClient
from generation.prompt_templates import RAG_ANSWER_PROMPT
from guardrails.input_guard import InputGuardrail
from guardrails.output_guard import OutputGuardrail
from guardrails.models import GuardrailVerdict

logger = logging.getLogger(__name__)


class RAGPipelineHarness:
    """Structured orchestration around the full RAG pipeline with
    stage-level tracing, circuit breakers, retries, and error recovery."""

    def __init__(self):
        self.settings = get_settings()

        # Embedder (lazy loads model on first encode call)
        self.embedder = Embedder()

        # Load chunks metadata
        try:
            self.chunks_metadata = self._load_chunks_metadata()
        except FileNotFoundError:
            self.chunks_metadata = []

        # Load FAISS index
        if os.path.exists(self.settings.FAISS_INDEX_PATH):
            self.faiss_index = FAISSIndex()
            self.faiss_index.load(self.settings.FAISS_INDEX_PATH)
        else:
            self.faiss_index = None

        # Retriever (needs embedder, index, chunks_metadata)
        if self.faiss_index and self.chunks_metadata:
            self.retriever = Retriever(
                embedder=self.embedder,
                index=self.faiss_index,
                chunks_metadata=self.chunks_metadata
            )
        else:
            self.retriever = None

        # Other components
        self.reranker = Reranker()
        self.llm_client = GroqLLMClient()
        self.input_guard = InputGuardrail()
        self.output_guard = OutputGuardrail()

        # Latency tracker
        if LatencyTracker:
            self.latency_tracker = LatencyTracker()
        else:
            self.latency_tracker = None

        # Circuit breakers for external APIs
        self.sarvam_cb = CircuitBreaker()
        self.groq_cb = CircuitBreaker()

    def _load_chunks_metadata(self) -> list:
        """Load chunk metadata from JSONL file."""
        chunks = []
        with open(self.settings.CHUNKS_METADATA_PATH, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    async def _transcribe(self, audio_bytes: bytes) -> str:
        """Call Sarvam AI STT API."""
        headers = {"api-subscription-key": self.settings.SARVAM_API_KEY}
        files = {"file": ("audio.wav", audio_bytes, "audio/wav")}
        data = {"model": self.settings.SARVAM_MODEL, "language_code": "en-IN"}

        def _call():
            resp = req_lib.post(
                self.settings.SARVAM_STT_URL,
                headers=headers,
                files=files,
                data=data,
                timeout=10
            )
            resp.raise_for_status()
            return resp.json().get("transcript", "")

        return await asyncio.to_thread(_call)

    async def execute(self, request: PipelineRequest) -> PipelineResult:
        """Execute the full RAG pipeline with stage-level tracing."""
        result = PipelineResult(query="", stages=[])
        overall_start = time.perf_counter()

        try:
            # ── Stage 1: Speech-to-Text (if audio input) ──
            if request.audio_bytes:
                stage = StageTrace(name="speech_to_text")
                start = time.perf_counter()
                try:
                    async with circuit_breaker_context(self.sarvam_cb):
                        transcript = await self._transcribe(request.audio_bytes)
                    stage.latency_ms = (time.perf_counter() - start) * 1000
                    stage.status = StageStatus.PASS
                    result.transcript = transcript
                    result.query = transcript
                    result.stt_latency_ms = stage.latency_ms
                except Exception as e:
                    stage.latency_ms = (time.perf_counter() - start) * 1000
                    stage.status = StageStatus.FAIL
                    stage.error = str(e)
                    if request.text:
                        result.query = request.text
                    else:
                        result.error = handle_stt_failure(e)
                        result.success = False
                        result.stages.append(stage)
                        result.total_latency_ms = (time.perf_counter() - overall_start) * 1000
                        return result
                result.stages.append(stage)
            elif request.text:
                result.query = request.text
            else:
                result.error = "No input provided"
                result.success = False
                result.total_latency_ms = (time.perf_counter() - overall_start) * 1000
                return result

            # ── Stage 2: Input Guardrail ──
            stage_ig = StageTrace(name="input_guardrail")
            start_ig = time.perf_counter()
            try:
                ig_result = self.input_guard.check(result.query)
                if ig_result.verdict != GuardrailVerdict.PASS:
                    stage_ig.status = StageStatus.FAIL
                    stage_ig.error = ig_result.reason
                    stage_ig.latency_ms = (time.perf_counter() - start_ig) * 1000
                    result.stages.append(stage_ig)
                    result.guardrail_verdict = ig_result.verdict.value
                    result.guardrail_reason = ig_result.reason
                    result.answer = f"I'm sorry, I cannot process that request. Reason: {ig_result.reason}"
                    result.success = True
                    result.total_latency_ms = (time.perf_counter() - overall_start) * 1000
                    return result
                stage_ig.status = StageStatus.PASS
            except Exception as e:
                stage_ig.status = StageStatus.FAIL
                stage_ig.error = str(e)
                logger.warning(f"Input guardrail error: {e}")
            stage_ig.latency_ms = (time.perf_counter() - start_ig) * 1000
            result.stages.append(stage_ig)

            # ── Stage 3: Retrieval (Embedding + FAISS Search) ──
            stage_ret = StageTrace(name="retrieval")
            start_ret = time.perf_counter()
            if not self.retriever:
                result.error = "Index not built yet. Run: python cli.py --build-index"
                result.success = False
                result.total_latency_ms = (time.perf_counter() - overall_start) * 1000
                return result

            try:
                raw_chunks = self.retriever.retrieve(
                    result.query,
                    top_k=self.settings.RETRIEVAL_TOP_K
                )
                stage_ret.latency_ms = (time.perf_counter() - start_ret) * 1000
                stage_ret.status = StageStatus.PASS
                stage_ret.details = {"num_raw_results": len(raw_chunks)}
            except Exception as e:
                stage_ret.latency_ms = (time.perf_counter() - start_ret) * 1000
                stage_ret.status = StageStatus.FAIL
                stage_ret.error = str(e)
                result.error = handle_retrieval_failure(e)
                result.success = False
                result.stages.append(stage_ret)
                result.total_latency_ms = (time.perf_counter() - overall_start) * 1000
                return result
            result.stages.append(stage_ret)

            # ── Stage 2b: Off-topic check using token overlap & relevance ──
            def _query_tokens(t: str) -> set:
                stopwords = {"what", "is", "the", "of", "how", "why", "where", "who", "when", "does", "do", "a", "an", "in", "to", "for"}
                words = set(t.lower().replace('?', '').split())
                return words - stopwords

            q_toks = _query_tokens(result.query)
            max_overlap = 0.0
            if q_toks and raw_chunks:
                for c in raw_chunks[:3]:
                    c_toks = set(c.get('text', '').lower().split())
                    overlap = len(q_toks.intersection(c_toks)) / len(q_toks)
                    if overlap > max_overlap:
                        max_overlap = overlap

            max_score = max([c.get('score', 0.0) for c in raw_chunks], default=0.0)
            if max_overlap == 0.0 or max_score < self.settings.OFF_TOPIC_THRESHOLD:
                result.guardrail_verdict = "refuse_off_topic"
                result.guardrail_reason = (
                    f"Query token overlap ({max_overlap:.2f}) with indexed corpus is zero. "
                    "Query appears off-topic."
                )
                result.answer = (
                    "I couldn't find relevant information in the knowledge base "
                    "to answer this question. The question may be outside the "
                    "scope of the MSMARCO dataset."
                )
                result.success = True
                result.total_latency_ms = (time.perf_counter() - overall_start) * 1000
                return result

            # ── Stage 4: Reranking ──
            stage_rerank = StageTrace(name="reranking")
            start_rerank = time.perf_counter()
            try:
                reranked = self.reranker.rerank(
                    raw_chunks,
                    final_k=self.settings.RETRIEVAL_FINAL_K
                )
                for c in reranked:
                    result.retrieved_chunks.append(RetrievedChunk(
                        text=c.get('text', ''),
                        chunk_id=str(c.get('chunk_id', '')),
                        strategy=str(c.get('strategy', 'unknown')),
                        score=float(c.get('score', 0.0)),
                        passage_id=str(c.get('passage_id', '')),
                        metadata=c.get('metadata', {})
                    ))
                stage_rerank.status = StageStatus.PASS
            except Exception as e:
                stage_rerank.status = StageStatus.FAIL
                stage_rerank.error = str(e)
                logger.warning(f"Reranking failed, using raw top-5: {e}")
                for c in raw_chunks[:self.settings.RETRIEVAL_FINAL_K]:
                    result.retrieved_chunks.append(RetrievedChunk(
                        text=c.get('text', ''),
                        chunk_id=str(c.get('chunk_id', '')),
                        strategy=str(c.get('strategy', 'unknown')),
                        score=float(c.get('score', 0.0)),
                        passage_id=str(c.get('passage_id', '')),
                        metadata=c.get('metadata', {})
                    ))
            stage_rerank.latency_ms = (time.perf_counter() - start_rerank) * 1000
            result.stages.append(stage_rerank)
            result.retrieval_latency_ms = stage_ret.latency_ms + stage_rerank.latency_ms

            # ── Stage 5: LLM Answer Generation ──
            stage_gen = StageTrace(name="generation")
            start_gen = time.perf_counter()
            try:
                async with circuit_breaker_context(self.groq_cb):
                    # Format context from retrieved chunks
                    context_parts = []
                    for i, c in enumerate(result.retrieved_chunks, 1):
                        context_parts.append(f"[Chunk {i}] {c.text}")
                    context_str = "\n\n".join(context_parts)

                    # Build system + user prompts
                    system_prompt = RAG_ANSWER_PROMPT.format(
                        context=context_str,
                        query=result.query
                    )
                    user_prompt = result.query

                    answer = await self.llm_client.generate(
                        system_prompt=system_prompt,
                        user_prompt=user_prompt,
                        max_tokens=self.settings.LLM_MAX_TOKENS,
                        temperature=self.settings.LLM_TEMPERATURE
                    )
                    result.answer = answer
                    stage_gen.status = StageStatus.PASS
            except Exception as e:
                stage_gen.status = StageStatus.FAIL
                stage_gen.error = str(e)
                result.answer = handle_llm_failure(e, result.retrieved_chunks)
            stage_gen.latency_ms = (time.perf_counter() - start_gen) * 1000
            result.generation_latency_ms = stage_gen.latency_ms
            result.stages.append(stage_gen)

            # ── Stage 6: Output Guardrail ──
            if stage_gen.status == StageStatus.PASS and result.answer:
                stage_og = StageTrace(name="output_guardrail")
                start_og = time.perf_counter()
                try:
                    context_texts = [c.text for c in result.retrieved_chunks]
                    retrieval_scores = [c.score for c in result.retrieved_chunks]
                    og_result = self.output_guard.check(
                        answer=result.answer,
                        context_chunks=context_texts,
                        retrieval_scores=retrieval_scores
                    )
                    if og_result.verdict == GuardrailVerdict.REFUSE_UNGROUNDED:
                        stage_og.status = StageStatus.FAIL
                        result.guardrail_verdict = og_result.verdict.value
                        result.guardrail_reason = og_result.reason
                        result.answer = (
                            "I couldn't provide a reliable answer based on the "
                            "retrieved context. The generated response didn't "
                            "appear well-grounded in the source material."
                        )
                    elif og_result.verdict == GuardrailVerdict.WARN_LOW_CONFIDENCE:
                        stage_og.status = StageStatus.PASS
                        result.guardrail_verdict = og_result.verdict.value
                        result.guardrail_reason = og_result.reason
                        # Keep the answer but flag it
                    else:
                        stage_og.status = StageStatus.PASS
                        result.guardrail_verdict = og_result.verdict.value
                        result.guardrail_reason = og_result.reason
                except Exception as e:
                    stage_og.status = StageStatus.FAIL
                    stage_og.error = str(e)
                    logger.warning(f"Output guardrail error: {e}")
                stage_og.latency_ms = (time.perf_counter() - start_og) * 1000
                result.stages.append(stage_og)

        except Exception as e:
            result.error = str(e)
            result.success = False
            logger.exception("Pipeline execution failed")

        result.total_latency_ms = (time.perf_counter() - overall_start) * 1000

        # Record in latency tracker
        if self.latency_tracker:
            try:
                self.latency_tracker.record(result)
            except Exception as e:
                logger.warning(f"Failed to record latency: {e}")

        return result
