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
from pydantic import BaseModel
from rag.pipeline import RAGPipeline
import uvicorn

load_dotenv()

app = FastAPI(title="Medical RAG Api")

rag_pipeline = RAGPipeline()


class UserQuery(BaseModel):
    question: str


@app.post('/query')
def query(request: UserQuery):
    """Sends Query to the medical RAG"""
    result = rag_pipeline.query(request.question)
    return {
        "answer": result["answer"],
        "num_sources": result['num_sources']
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

