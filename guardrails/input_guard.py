"""
Input guardrail for validating user queries.
"""
import re
from typing import Optional
from guardrails.models import GuardrailResult, GuardrailVerdict
from config import get_settings

# Module-level list of compiled regex patterns
UNSAFE_PATTERNS = [
    re.compile(r"ignore previous", re.IGNORECASE),
    re.compile(r"ignore above", re.IGNORECASE),
    re.compile(r"system prompt", re.IGNORECASE),
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),  # SSN pattern
    re.compile(r"\b(?:\d[ -]*?){13,16}\b") # Credit card pattern
]

class InputGuardrail:
    def __init__(self):
        self.settings = get_settings()
        
    def check(self, query: str, max_retrieval_score: float = None) -> GuardrailResult:
        settings = self.settings
        
        # 1. Length validation
        if len(query) < settings.QUERY_MIN_LENGTH or len(query) > settings.QUERY_MAX_LENGTH:
            return GuardrailResult(
                verdict=GuardrailVerdict.REFUSE_UNSAFE,
                reason=f"Query length {len(query)} is out of bounds [{settings.QUERY_MIN_LENGTH}, {settings.QUERY_MAX_LENGTH}]",
                confidence=1.0
            )
            
        # 2. Safety filter
        for pattern in UNSAFE_PATTERNS:
            if pattern.search(query):
                return GuardrailResult(
                    verdict=GuardrailVerdict.REFUSE_UNSAFE,
                    reason="Query matches unsafe patterns (prompt injection or PII).",
                    confidence=1.0
                )
                
        # 3. Off-topic detection
        if max_retrieval_score is not None:
            if max_retrieval_score < settings.OFF_TOPIC_THRESHOLD:
                return GuardrailResult(
                    verdict=GuardrailVerdict.REFUSE_OFF_TOPIC,
                    reason=f"Max retrieval score {max_retrieval_score} is below threshold {settings.OFF_TOPIC_THRESHOLD}",
                    confidence=1.0
                )
                
        # 4. Pass
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS,
            reason="Query passed all input guardrails.",
            confidence=1.0
        )
