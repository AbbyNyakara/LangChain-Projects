'''
Docstring for api.main

Creates a minimalistic API to access the medical RAG
'''
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from src.rag.pipeline import MedicalRAGPipeline
from typing import Optional
from contextlib import asynccontextmanager
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Global pipeline instance - for Medical RAG Pipeline to be reused
pipeline: Optional[MedicalRAGPipeline] = None

@asynccontextmanager
async def lifespan(app:FastAPI):
    """
    Application lifespan manager
    - Startup: Initialize pipeline before requests
    - Shutdown: Cleanup resources after requests
    """
    global pipeline

    logger.info("Starting RAG API")

    try:
        pipeline = MedicalRAGPipeline(
            s3_bucket="medical-rag-docs-abigael-2026",
            llm_config={
                'model': 'gpt-4-turbo',
                'temperature': 0.2,
                'max_tokens': 500
            })
        logger.info("Pipeline initialized")
    except Exception as e:
        logger.error("Failed to initialize pipeline %s", e)

    yield #Yield control after setup and code below yield will run on shutdown

    logger.info("Shutting down API")
    # Cleanup if needed - delete the Embeddings, the docs form s3 bucket
    pipeline = None
    logger.info("Cleanup complete")

app = FastAPI(
    title="Medical RAG API",
    description="Upload medical documents and ask questions",
    version='1.0.0',
    lifespan=lifespan
)


## Request and Response models

class DocumentUploadResponse(BaseModel):
    """Response for document upload"""
    success: bool
    document_id: Optional[str] = None
    filename: str
    total_chunks: int
    vectors_stored: int
    message: str
    error: Optional[str] = None

class QueryRequest(BaseModel):
    """Request for asking a question"""
    question: str
    top_k: int = 5


class QueryResponse(BaseModel):
    """Response for a question"""
    success: bool
    question: str
    answer: str
    num_sources: int
    processing_time_seconds: float
    error: Optional[str] = None

# class PipelineStats(BaseModel):
#     """Pipeline statistics"""
#     total_vectors: int
#     dimension: int
#     is_ready: bool

## Create the endpoints:
@app.post('/upload', response_model=DocumentUploadResponse, summary="Upload and index document")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a file for indexing
    - The file is to be extracted for text, chunked, and vectorized and stored in a vector DB
    - Ready for querying
    """
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF Files allowed") # Check which files allow for aws texttract
    try:
        result = pipeline.index_document(file_path=file) # Extracts, chunks and embeds
        logger.info("Indexed pdf document")
    except Exception as e:
        logger.error("Error indexing document %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"Error uploading document: {str(e)}"
        )
    

@app.post('/query', response_model=QueryResponse, summary="Ask question from uploaded document")




