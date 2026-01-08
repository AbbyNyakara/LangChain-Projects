'''
Docstring for api.main

Creates a minimalistic API to access the medical RAG
'''
import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

from fastapi import FastAPI
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from src.rag.pipeline import RAGPipeline
import uvicorn

load_dotenv()

app = FastAPI(title="Medical RAG Api")

rag_pipeline = RAGPipeline()


class UserQuery(BaseModel):
    question: str = Field(
        ...,  # Requiredd field
        example="What are the patient's symptoms?",
        description="Medical question or query to be processed by the RAG pipeline"
    )


class QueryResponse(BaseModel):
    question: str
    answer: str
    num_sources: int


@app.post('/query', response_model=QueryResponse)
def query(request: UserQuery):
    """Sends Query to the medical RAG"""
    if rag_pipeline is None:
        return QueryResponse(
            question=request.question,
            answer="Error: RAG pipeline not initialized",
            num_sources=0
        )

    result = rag_pipeline.query(request.question)
    return QueryResponse(
        question=result['question'], 
        answer=result['answer'],
        num_sources=result['num_sources']
    )
    
# if __name__ == "__main__":
#     uvicorn.run(app, host="0.0.0.0", port=8000)
