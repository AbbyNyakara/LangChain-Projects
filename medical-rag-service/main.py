import sys
from pathlib import Path
from contextlib import asynccontextmanager

# Setup paths
src_dir = Path(__file__).parent / "src"
sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
import uvicorn
from api.main import app
from src.rag.pipeline import RAGPipeline

load_dotenv()

rag_pipeline = None

@asynccontextmanager
async def lifespan(app):
    """Startup and shutdown logic"""
    global rag_pipeline
    
    # STARTUP
    project_root = Path(__file__).parent
    chroma_db_path = project_root / "chroma_db"
    pdf_path = project_root / "data" / "fake-aps.pdf"
    
    rag_pipeline = RAGPipeline(persist_directory=str(chroma_db_path))
    
    # Auto-ingest if vector store is empty
    result = rag_pipeline.query("What are the patient's symptoms?")
    if result["num_sources"] == 0 and pdf_path.exists():
        rag_pipeline.ingest(pdf_path)
    
    # Set pipeline in the app
    import api.main
    api.main.rag_pipeline = rag_pipeline
    
    yield
    

# Add lifespan to app
app.router.lifespan = lifespan

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
