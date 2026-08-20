import asyncio
import os
import json
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional

from config import get_settings
from harness.pipeline import RAGPipelineHarness
from harness.models import PipelineRequest, PipelineResult
from analytics.latency_tracker import LatencyTracker

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="Voice-Enabled RAG Pipeline",
    description="MSMARCO-XI powered voice RAG system with multi-strategy chunking",
    version="1.0.0"
)

# CORS for local frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline instance (lazy-loaded)
pipeline: Optional[RAGPipelineHarness] = None
latency_tracker = LatencyTracker()

@app.on_event("startup")
async def startup():
    global pipeline
    try:
        pipeline = RAGPipelineHarness()
        logger.info("Pipeline initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize pipeline: {e}")
        logger.info("Server running in degraded mode. Build index first: python cli.py --build-index")

class QueryRequest(BaseModel):
    text: str

@app.post("/api/query", response_model=PipelineResult)
async def query_text(req: QueryRequest):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Index might be missing.")
    try:
        request = PipelineRequest(text=req.text)
        result = await pipeline.execute(request)
        return result
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice", response_model=PipelineResult)
async def query_voice(audio_file: UploadFile = File(...)):
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized. Index might be missing.")
    try:
        audio_bytes = await audio_file.read()
        request = PipelineRequest(audio_bytes=audio_bytes)
        result = await pipeline.execute(request)
        return result
    except Exception as e:
        logger.error(f"Error executing voice query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    index_loaded = pipeline is not None
    num_chunks = 0
    if index_loaded and hasattr(pipeline, "retriever") and pipeline.retriever is not None:
        num_chunks = getattr(pipeline.retriever, "num_chunks", getattr(pipeline.retriever.index, "ntotal", 0)) if hasattr(pipeline.retriever, "index") else 0
    
    return {
        "status": "ok",
        "index_loaded": index_loaded,
        "num_chunks": num_chunks
    }

@app.get("/api/stats")
async def get_stats():
    return latency_tracker.get_summary()

@app.get("/")
async def serve_frontend():
    # Serve the HTML frontend file
    html_path = os.path.join(os.path.dirname(__file__), "voice_rag_hacker_goa_house_2026_no_fetch_error.html")
    if os.path.exists(html_path):
        return FileResponse(html_path)
    return {"error": "Frontend file not found"}

if __name__ == "__main__":
    import uvicorn
    settings = get_settings()
    uvicorn.run(app, host=settings.SERVER_HOST, port=settings.SERVER_PORT)
