"""
FastAPI backend.
Exposes the RAG pipeline over HTTP so any frontend can consume it.

Endpoints:
  POST /ask         — full response, returns JSON
  POST /ask/stream  — streaming response via Server-Sent Events
"""

import sys
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from generate import generate_answer, stream_answer

app = FastAPI(title="Medical Q&A API")


class AskRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(req: AskRequest):
    """
    Full RAG pipeline. Returns answer, sources, and latency_ms as JSON.
    """
    result = generate_answer(req.query, top_k=req.top_k)
    return result


@app.post("/ask/stream")
def ask_stream(req: AskRequest):
    """
    Streaming version. Emits Server-Sent Events:
      - text chunks:  data: {"type": "text", "content": "..."}
      - final meta:   data: {"type": "meta", "sources": [...], "latency_ms": ..., "answer": "..."}
      - done signal:  data: [DONE]
    """
    def event_stream():
        for chunk in stream_answer(req.query, top_k=req.top_k):
            if isinstance(chunk, dict):
                payload = {"type": "meta", **chunk}
            else:
                payload = {"type": "text", "content": chunk}
            yield f"data: {json.dumps(payload)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
