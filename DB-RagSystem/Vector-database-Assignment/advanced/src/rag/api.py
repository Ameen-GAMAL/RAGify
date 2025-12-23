"""
FastAPI backend for RAG system - CORRECTED VERSION
Uses the new simplified pipeline without heavy LLM
"""
from __future__ import annotations

from typing import List, Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.rag.pipeline import RAGPipeline, RetrievalResult
from src.learn.feedback_store import FeedbackStore


# ============================================================================
# Request/Response Models
# ============================================================================

class QueryRequest(BaseModel):
    query: str = Field(..., description="User query", min_length=1)
    top_k: int = Field(5, description="Number of results", ge=1, le=20)
    return_sources: bool = Field(True, description="Include source documents")


class SourceResponse(BaseModel):
    chunk_id: str
    score: float
    rank: int
    question: str
    answer: str
    snippet: str
    boosted: bool = False
    original_score: float = 0.0
    metadata: Dict[str, Any] = {}


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceResponse]
    retrieval_time_ms: float
    generation_time_ms: float
    total_time_ms: float


class FeedbackRequest(BaseModel):
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[Dict[str, Any]] = Field(..., description="Retrieved sources")
    rating: int = Field(..., description="Rating: 1=helpful, -1=not helpful", ge=-1, le=1)
    comment: str = Field("", description="Optional comment")
    session_id: Optional[str] = Field(None, description="Session ID")


class FeedbackResponse(BaseModel):
    feedback_id: str
    message: str
    boosts_updated: bool


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    index_vectors: int
    model_name: str


# ============================================================================
# Initialize FastAPI App
# ============================================================================

app = FastAPI(
    title="E-commerce Q&A RAG API",
    description="Retrieval-Augmented Generation system with self-learning",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global instances (initialized on startup)
rag_pipeline: Optional[RAGPipeline] = None
feedback_store: Optional[FeedbackStore] = None


# ============================================================================
# Startup/Shutdown Events
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize RAG pipeline and feedback store"""
    global rag_pipeline, feedback_store
    
    print("\n" + "="*70)
    print(" Starting E-commerce Q&A RAG API")
    print("="*70)
    
    try:
        # Initialize RAG pipeline
        rag_pipeline = RAGPipeline(
            index_dir="data/processed",
            model_name="sentence-transformers/multi-qa-mpnet-base-dot-v1",
        )
        
        # Initialize feedback store
        feedback_store = FeedbackStore(
            feedback_path="data/processed/feedback.jsonl",
            boosts_path="data/processed/boosts.json",
        )
        
        print("\n API Ready!")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\nFailed to initialize API: {e}")
        print("="*70 + "\n")
        raise


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("\n" + "="*70)
    print(" Shutting down E-commerce Q&A RAG API")
    print("="*70 + "\n")


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/", response_model=HealthResponse)
async def root():
    """Health check endpoint"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        index_vectors=rag_pipeline.index.ntotal,
        model_name=rag_pipeline.index_meta["model_name"],
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Query the RAG system.
    
    Retrieves relevant Q&A pairs and generates an answer.
    """
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        import time
        start = time.time()
        
        # Run RAG pipeline
        response = rag_pipeline.query(
            query=request.query,
            top_k=request.top_k,
            return_sources=request.return_sources,
        )
        
        total_time_ms = (time.time() - start) * 1000
        
        # Convert sources to response format
        sources_response = []
        if request.return_sources:
            for src in response.sources:
                # Create snippet from question (first 200 chars)
                snippet = src.metadata.get('question', '')[:200]
                if len(src.metadata.get('question', '')) > 200:
                    snippet += '...'
                
                sources_response.append(SourceResponse(
                    chunk_id=src.chunk_id,
                    score=src.score,
                    rank=src.rank,
                    question=src.metadata.get('question', ''),
                    answer=src.metadata.get('answer', ''),
                    snippet=snippet,
                    boosted=src.boosted,
                    original_score=src.original_score,
                    metadata={
                        'question_type': src.metadata.get('question_type', ''),
                        'char_len': src.metadata.get('char_len', 0),
                        'source_file': src.metadata.get('source_file', ''),
                    }
                ))
        
        return QueryResponse(
            query=response.query,
            answer=response.answer,
            sources=sources_response,
            retrieval_time_ms=response.retrieval_time_ms,
            generation_time_ms=response.generation_time_ms,
            total_time_ms=total_time_ms,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {str(e)}")


@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """
    Submit user feedback for self-learning.
    
    Updates ranking boosts based on feedback.
    """
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")
    
    try:
        # Add feedback
        feedback_id = feedback_store.add_feedback(
            query=request.query,
            answer=request.answer,
            sources=request.sources,
            rating=request.rating,
            comment=request.comment,
            session_id=request.session_id,
        )
        
        return FeedbackResponse(
            feedback_id=feedback_id,
            message="Feedback recorded successfully",
            boosts_updated=True,
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")


@app.get("/feedback/summary")
async def feedback_summary():
    """Get feedback summary statistics"""
    if feedback_store is None:
        raise HTTPException(status_code=503, detail="Feedback store not initialized")
    
    try:
        summary = feedback_store.get_feedback_summary()
        boosts_summary = feedback_store.get_boosts_summary()
        
        return {
            "feedback": summary,
            "boosts": boosts_summary,
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get summary: {str(e)}")


@app.get("/stats")
async def stats():
    """Get system statistics"""
    if rag_pipeline is None:
        raise HTTPException(status_code=503, detail="RAG pipeline not initialized")
    
    try:
        q_types = rag_pipeline.index_meta["statistics"]["question_types"]
        
        return {
            "index": {
                "total_vectors": rag_pipeline.index.ntotal,
                "embedding_dim": rag_pipeline.index.d,
                "model_name": rag_pipeline.index_meta["model_name"],
                "created_at": rag_pipeline.index_meta["created_at"],
            },
            "dataset": {
                "total_chunks": len(rag_pipeline.chunks),
                "avg_chunk_length": rag_pipeline.index_meta["statistics"]["avg_chunk_length"],
                "question_types": q_types,
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {str(e)}")


# ============================================================================
# Run with: uvicorn src.rag.api:app --reload --host 0.0.0.0 --port 8000
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)