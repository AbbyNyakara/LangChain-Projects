'''
Docstring for api.main

Creates a minimalistic API to access the medical RAG
'''
import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

import uvicorn
# from rag.pipeline import RAGPipeline
from src.rag.pipeline import RAGPipeline
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from fastapi import FastAPI


load_dotenv()

app = FastAPI(title="Medical RAG Api")

rag_pipeline = RAGPipeline()


class UserQuery(BaseModel):
    question: str = Field(
        ..., #Requiredd field
        example="What are the patient's symptoms?",
        description="Medical question or query to be processed by the RAG pipeline"
    )


@app.post('/query')
def query(request: UserQuery):
    """Sends Query to the medical RAG"""
    result = rag_pipeline.query(request.question)
    return {
        "answer": result["answer"],
        "num_sources": result['num_sources']
    }


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
