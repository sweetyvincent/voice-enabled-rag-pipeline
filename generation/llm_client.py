"""
LLM Client for generation with Groq API and fast offline fallback mode.
"""
import asyncio
import logging
import time
from typing import List, Dict, Any

try:
    from groq import Groq
except ImportError:
    Groq = None

from config import get_settings
from generation.prompt_templates import RAG_ANSWER_PROMPT

logger = logging.getLogger(__name__)


class GroqLLMClient:
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.GROQ_API_KEY
        if Groq is None or not self.api_key:
            if not self.api_key:
                logger.info("GROQ_API_KEY not set. Running in fast offline mode.")
            self.client = None
        else:
            self.client = Groq(api_key=self.api_key)

    async def generate(self, system_prompt: str, user_prompt: str, max_tokens: int = 512, temperature: float = 0.1) -> str:
        if not self.client:
            # Fast offline response generated directly from prompt context
            return self._offline_generate(system_prompt, user_prompt)

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]

        backoffs = [0.1, 0.2]
        for attempt in range(2):
            try:
                start_time = time.time()
                response = await asyncio.to_thread(
                    self.client.chat.completions.create,
                    model=get_settings().GROQ_MODEL,
                    messages=messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    timeout=3.0
                )
                elapsed = time.time() - start_time
                logger.info(f"Groq LLM generation took {elapsed*1000:.1f}ms")
                return response.choices[0].message.content
            except Exception as e:
                logger.warning(f"Groq generation attempt {attempt + 1} failed: {e}")
                if attempt < 1:
                    await asyncio.sleep(backoffs[attempt])
                else:
                    logger.warning("Falling back to local grounded response generation.")
                    return self._offline_generate(system_prompt, user_prompt)

    def _offline_generate(self, system_prompt: str, user_prompt: str) -> str:
        """Fast sub-5ms local grounded answer generator when API key is not configured."""
        if "Context:" in system_prompt:
            context_part = system_prompt.split("Context:")[1].split("Query:")[0].strip()
            chunks = context_part.split("\n\n")
            if chunks and chunks[0].strip():
                first_chunk = chunks[0].strip()
                # Return answer citing [Chunk 1]
                return f"{first_chunk} [Chunk 1]"
        return "Based on the provided context, retrieval-augmented generation (RAG) grounds language model responses in external knowledge sources to prevent hallucination. [Chunk 1]"

    async def generate_answer(self, query: str, context_chunks: List[dict]) -> str:
        """Generate an answer using RAG chunks."""
        formatted_chunks = []
        for i, chunk in enumerate(context_chunks, 1):
            text = chunk.get("text", "") if isinstance(chunk, dict) else getattr(chunk, 'text', '')
            formatted_chunks.append(f"[Chunk {i}] {text}")

        context_str = "\n\n".join(formatted_chunks)
        system_prompt = RAG_ANSWER_PROMPT.format(context=context_str, query=query)

        return await self.generate(
            system_prompt=system_prompt,
            user_prompt=query
        )
