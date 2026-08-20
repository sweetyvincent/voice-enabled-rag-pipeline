"""
Output guardrail for validating LLM answers.
"""
import re
from typing import List
from guardrails.models import GuardrailResult, GuardrailVerdict
from config import get_settings

STOP_WORDS = {
    "the", "a", "an", "is", "are", "was", "were", "and", "or", "but", "if", 
    "then", "else", "when", "at", "from", "by", "for", "with", "about", 
    "against", "between", "into", "through", "during", "before", "after", 
    "above", "below", "to", "of", "in", "on", "it", "that", "this", "these", "those"
}

class OutputGuardrail:
    def __init__(self):
        self.settings = get_settings()
        
    def check(self, answer: str, context_chunks: List[str], retrieval_scores: List[float]) -> GuardrailResult:
        settings = self.settings
        
        # 1. Empty answer check
        if not answer or not answer.strip():
            return GuardrailResult(
                verdict=GuardrailVerdict.REFUSE_UNGROUNDED,
                reason="Empty answer provided.",
                confidence=1.0
            )
            
        # 2. Grounding check
        def get_tokens(text: str) -> set:
            tokens = set(re.findall(r'\b\w+\b', text.lower()))
            return tokens - STOP_WORDS
            
        answer_tokens = get_tokens(answer)
        if answer_tokens:
            context_tokens = set()
            for chunk in context_chunks:
                context_tokens.update(get_tokens(chunk))
                
            overlap = len(answer_tokens.intersection(context_tokens))
            overlap_ratio = overlap / len(answer_tokens)
            
            if overlap_ratio < settings.GROUNDING_MIN_OVERLAP:
                return GuardrailResult(
                    verdict=GuardrailVerdict.REFUSE_UNGROUNDED,
                    reason=f"Answer lacks grounding. Token overlap ratio {overlap_ratio:.2f} < {settings.GROUNDING_MIN_OVERLAP}.",
                    confidence=1.0,
                    details={"overlap_ratio": overlap_ratio}
                )
                
        # 3. Low confidence check
        if retrieval_scores:
            mean_score = sum(retrieval_scores) / len(retrieval_scores)
            if mean_score < settings.RELEVANCE_THRESHOLD:
                return GuardrailResult(
                    verdict=GuardrailVerdict.WARN_LOW_CONFIDENCE,
                    reason=f"Mean retrieval score {mean_score:.2f} is below relevance threshold {settings.RELEVANCE_THRESHOLD}.",
                    confidence=mean_score
                )
                
        # 4. Citation check
        chunk_refs = re.findall(r'\[Chunk (\d+)\]', answer)
        for ref in chunk_refs:
            if int(ref) > len(context_chunks) or int(ref) < 1:
                return GuardrailResult(
                    verdict=GuardrailVerdict.WARN_LOW_CONFIDENCE,
                    reason=f"Answer cites [Chunk {ref}] which is out of bounds (1 to {len(context_chunks)}).",
                    confidence=1.0
                )
                
        # 5. Pass
        return GuardrailResult(
            verdict=GuardrailVerdict.PASS,
            reason="Answer passed all output guardrails.",
            confidence=1.0
        )
