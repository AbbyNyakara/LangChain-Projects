medical-rag-service/
│
├── pyproject.toml                 # Poetry config & dependencies
├── poetry.lock
├── .env
├── .gitignore
├── README.md
│
├── bin/
│   └── dev                        # Startup script
│
├── main.py                        # Entry point (runs the FastAPI app)
│
├── config.py                      # Configuration & environment variables
│
├── src/                           # Main source code
│   ├── __init__.py
│   │
│   ├── document/                  # Document processing pipeline
│   │   ├── __init__.py
│   │   └── extractor.py           # Text extraction 
│   ├── chunking/                  # Document chunking strategies
│   │   ├── __init__.py
│   │   └── semantic.py           # Semantic chunking
│   │
│   ├── embedding/                 # Embedding generation
│   │   ├── __init__.py
│   │   └── openai_embeddings.py   # OpenAI embeddings (optional)
│   │
│   ├── retrieval/                 # Vector DB & retrieval
│   │   ├── __init__.py
│   │   └── chroma_db.py         # Chroma implementation
│   │
│   ├── generation/                # LLM-based answer generation
│   │   ├── __init__.py
│   │   ├── openai_llm.py          # OpenAI integration
│   │   ├── meditron_llm.py        # Meditron integration (Specialized medical LLM by Llama)
│   │   └── prompt_template.py     # Prompt engineering
│   │
│   ├── rag/                       # RAG orchestration
│   │   ├── __init__.py
│   │   └── pipeline.py            # Main RAG pipeline (ties everything together)
│   │
│   ├── api/                       # FastAPI endpoints
│   │   ├── __init__.py
│   │   ├── router.py              # API routes
│   │   ├── models.py              # Pydantic models (request/response)
│   │   └── dependencies.py        # Dependency injection
│   │
│   └── utils/                     # Utility functions
│       ├── __init__.py
│       ├── logging.py             # Logging setup
│       └── validators.py          # Input validation
│
├── tests/                         # Test suite
│   ├── __init__.py
│   ├── conftest.py                # Pytest fixtures
│   ├── test_loader.py
│   ├── test_chunking.py
│   ├── test_embedding.py
│   ├── test_retrieval.py
│   ├── test_generation.py
│   ├── test_rag_pipeline.py
│   └── test_api.py
│
├── data/                          # Medical documents
│   ├── raw/                       # Original documents
│   │   ├── document1.pdf
│   │   └── document2.pdf
│   ├── processed/                 # Processed/extracted text
│   │   ├── document1.txt
│   │   └── document2.txt
│   └── embeddings/                # Cached embeddings (optional)
│       └── index.faiss
│
└── scripts/                       # Utility scripts
    ├── process_documents.py       # Batch process documents
    ├── build_index.py             # Build vector index
    └── test_rag.py                # Test RAG system
