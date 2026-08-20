"""
Models for guardrails.
"""
from enum import Enum
from pydantic import BaseModel
from typing import Dict, Any

class GuardrailVerdict(str, Enum):
    PASS = "pass"
    REFUSE_OFF_TOPIC = "refuse_off_topic"
    REFUSE_UNSAFE = "refuse_unsafe" 
    REFUSE_UNGROUNDED = "refuse_ungrounded"
    WARN_LOW_CONFIDENCE = "warn_low_confidence"

class GuardrailResult(BaseModel):
    verdict: GuardrailVerdict
    reason: str
    confidence: float
    details: Dict[str, Any] = {}
