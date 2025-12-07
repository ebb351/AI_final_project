#!/usr/bin/env python3
"""
FastAPI Server for GraphRAG AI Tutor

Endpoints:
- GET /health - Health check
- POST /query - Answer question with citations
"""
import logging
import os
import time
from typing import Dict, List, Any, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, Field
import uvicorn
import gradio as gr

# Import query system
from src.query.processor import QueryProcessor
from src.query.context_builder import ContextBuilder
from src.query.generator import AnswerGenerator, AnswerResponse
from src.database.init_data import is_mongodb_populated, auto_import_data, ensure_ollama_model

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global instances
query_processor: Optional[QueryProcessor] = None
context_builder: Optional[ContextBuilder] = None
answer_generator: Optional[AnswerGenerator] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for FastAPI app."""
    global query_processor, context_builder, answer_generator

    logger.info("Initializing GraphRAG services...")

    # Initialize query processor (needed for DB connection)
    query_processor = QueryProcessor()

    # Check if MongoDB needs data import
    db = query_processor._get_db()
    if not is_mongodb_populated(db):
        logger.warning("MongoDB is empty - attempting auto-import...")
        mongodb_uri = os.getenv("MONGODB_URI", "mongodb://localhost:27017/")
        database_name = os.getenv("MONGODB_DATABASE", "graphrag_course_db")
        auto_import_data(mongodb_uri, database_name)
    else:
        logger.info("MongoDB already populated - skipping import")

    # Initialize other components
    context_builder = ContextBuilder()
    answer_generator = AnswerGenerator()

    # Ensure Ollama model is available
    ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    model_name = answer_generator.model
    if not ensure_ollama_model(ollama_host, model_name):
        logger.warning(f"Ollama model '{model_name}' not available - queries may fail")

    # Pre-load embeddings
    logger.info("Pre-loading entity embeddings...")
    try:
        query_processor._load_entity_embeddings()
        logger.info("Embeddings loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load embeddings: {e}")

    yield

    # Cleanup
    logger.info("Shutting down GraphRAG services...")
    if query_processor:
        query_processor.close()


# Create FastAPI app
app = FastAPI(
    title="GraphRAG AI Tutor API",
    description="AI Course Tutor powered by GraphRAG",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models
class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    question: str = Field(..., description="Question to answer", min_length=3)
    max_entities: int = Field(20, description="Maximum entities to retrieve", ge=1, le=50)
    max_chunks: int = Field(10, description="Maximum source chunks", ge=1, le=20)


class Citation(BaseModel):
    """Citation model."""
    title: str
    url: str


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    answer: str
    entities_used: List[str]
    citations: List[Citation]
    model: str
    latency_ms: int


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    mongodb: str
    ollama: str
    embeddings_loaded: int
    model: str


# Endpoints
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Check service health."""
    global query_processor, answer_generator

    # Check MongoDB
    mongodb_status = "unknown"
    embeddings_count = 0
    try:
        if query_processor:
            db = query_processor._get_db()
            db.command("ping")
            embeddings_count = db.embeddings.count_documents({})
            mongodb_status = "connected"
    except Exception as e:
        mongodb_status = f"error: {str(e)}"

    # Check Ollama
    ollama_status = "unknown"
    if answer_generator:
        status = answer_generator.check_ollama_status()
        if status.get("available"):
            if status.get("target_model_available"):
                ollama_status = "ready"
            else:
                ollama_status = f"model not found: {answer_generator.model}"
        else:
            ollama_status = f"error: {status.get('error', 'unknown')}"

    # Overall status
    overall_status = "healthy"
    if mongodb_status != "connected" or ollama_status != "ready":
        overall_status = "degraded"

    return HealthResponse(
        status=overall_status,
        mongodb=mongodb_status,
        ollama=ollama_status,
        embeddings_loaded=embeddings_count,
        model=answer_generator.model if answer_generator else "unknown"
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Answer a question using GraphRAG.

    Process:
    1. Embed query and find similar entities
    2. Expand subgraph around entities
    3. Build context from entities, relationships, and chunks
    4. Generate answer using Ollama
    """
    global query_processor, context_builder, answer_generator

    if not query_processor or not context_builder or not answer_generator:
        raise HTTPException(status_code=503, detail="Services not initialized")

    start_time = time.time()

    try:
        # Step 1: Process query
        logger.info(f"Processing query: {request.question[:100]}...")
        query_result = query_processor.process_query(
            query=request.question,
            top_k_entities=request.max_entities,
            max_chunks=request.max_chunks
        )

        # Step 2: Build context
        context = context_builder.build_context(query_result)

        # Step 3: Generate answer
        response = answer_generator.generate_answer(
            query=request.question,
            context=context
        )

        # Build citations
        citations = [
            Citation(title=s.get("title", "Unknown"), url=s.get("url", ""))
            for s in response.sources
            if s.get("url")
        ]

        latency_ms = int((time.time() - start_time) * 1000)

        return QueryResponse(
            answer=response.answer,
            entities_used=response.entities_used[:10],  # Limit to top 10
            citations=citations,
            model=response.model,
            latency_ms=latency_ms
        )

    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint - redirect to chat interface."""
    return RedirectResponse(url="/chat")


@app.get("/api")
async def api_info():
    """API info endpoint."""
    return {
        "name": "GraphRAG AI Tutor API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "query": "/query",
            "chat": "/chat"
        }
    }


# Mount Gradio chat interface
from src.frontend.chat import create_chat_interface

chat_demo = create_chat_interface()
app = gr.mount_gradio_app(app, chat_demo, path="/chat")


def main():
    """Run server."""
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info(f"Starting GraphRAG API server on {host}:{port}")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
