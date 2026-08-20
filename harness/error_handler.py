import time
from contextlib import asynccontextmanager

class CircuitBreaker:
    def __init__(self, failure_threshold=5, recovery_timeout=30.0):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.failures = 0
        self.last_failure_time = 0.0
        
    def record_success(self):
        self.failures = 0
        self.last_failure_time = 0.0
        
    def record_failure(self):
        self.failures += 1
        self.last_failure_time = time.time()
        
    @property
    def is_open(self) -> bool:
        if self.failures >= self.failure_threshold:
            if time.time() - self.last_failure_time < self.recovery_timeout:
                return True
            else:
                # Half-open state
                pass
        return False

@asynccontextmanager
async def circuit_breaker_context(cb: CircuitBreaker):
    if cb.is_open:
        raise Exception("Circuit breaker is open")
    try:
        yield
        cb.record_success()
    except Exception as e:
        cb.record_failure()
        raise e

class PipelineError(Exception):
    def __init__(self, error_code: str, user_message: str, stage: str):
        super().__init__(user_message)
        self.error_code = error_code
        self.user_message = user_message
        self.stage = stage

def handle_stt_failure(error: Exception) -> str:
    return "Audio processing failed. Please try providing text input instead."

def handle_llm_failure(error: Exception, chunks: list) -> str:
    if not chunks:
        return "I'm currently experiencing issues and couldn't find any relevant information to answer your query."
    
    fallback = "I'm having trouble connecting to my language model right now, but here is some relevant information I found:\n\n"
    for i, chunk in enumerate(chunks, 1):
        text = getattr(chunk, 'text', '') if hasattr(chunk, 'text') else chunk.get('text', '')
        fallback += f"{i}. {text}\n\n"
    return fallback

def handle_retrieval_failure(error: Exception) -> str:
    return "I encountered an error while searching for information. Please try your query again later."
