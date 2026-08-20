"""
Prompt templates for LLM generation.
"""

RAG_ANSWER_PROMPT = """You are a helpful and precise assistant. 
Your task is to answer the user's question ONLY based on the provided context chunks.
- Cite sources using [Chunk N] notation.
- If the context doesn't contain enough information to answer, say so explicitly.
- Be concise and direct.
- Never make up information not in the context.

Context:
{context}

Query:
{query}
"""

OFF_TOPIC_CLASSIFICATION_PROMPT = """You are a classification model.
Classify whether the given user query is relevant to general knowledge/QA topics (the MSMARCO domain).
Respond with a JSON object in the following format exactly:
{"is_on_topic": bool, "reason": str}

Query: {query}
"""

GROUNDING_CHECK_PROMPT = """You are an accuracy verification model.
Check whether every claim in the answer is fully supported by the provided context.
Respond with a JSON object in the following format exactly:
{"is_grounded": bool, "unsupported_claims": [str]}

Context:
{context}

Answer:
{answer}
"""
