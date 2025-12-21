import sys
from pathlib import Path

src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from dotenv import load_dotenv
from generation.openai_llm import GenerateService
from generation.prompt_template import MEDICAL_ASSISTANT_PROMPT
from retrieval.chroma_db import VectorStoreService
from embedding.openai_embeddings import EmbeddingService
from chunking.semantic import chunk_text
from document.extractor import extract_pdf_text
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

class RAGPipeline:
    def __init__(self, collection_name='medical_notes', model: str = "gpt-4o-mini", persist_directory: str = None):

        if persist_directory is None:
            project_root = src_dir.parent
            persist_directory = str(project_root / "chroma_db")

        # Initialize services
        self.vector_store_service = VectorStoreService(
            collection_name=collection_name,
            persist_directory=persist_directory)

        self.generate_service = GenerateService(model=model)
        self.prompt_template = MEDICAL_ASSISTANT_PROMPT

        # Load vector Store 
        try:
            self.vector_store_service.load_vector_store()
        except Exception as e: # incase it hasnt been created
            print(f"Note: Could not load existing vector store: {e}")
            print("A new vector store will be created upon ingestion.")

        # Then build the chain
        self.rag_chain = self._build_chain()

    def _build_chain(self):
        """Builds the LangChain retriever -> context preparation -> LLM """
        retriever = self.vector_store_service.as_retriever(k=3)

        rag_chain = (
            {"context": retriever | self._format_documents,
             "question": RunnablePassthrough()}
            | self.prompt_template
            | self.generate_service.llm
        )

        return rag_chain

    @staticmethod
    def _format_documents(docs):
        """Format retrieved documents into context string."""
        if not docs:
            return "No relevant documents found."
        return "\n\n".join([doc.page_content for doc in docs])

    def query(self, question: str) -> dict:
        '''Invokes the chain'''
        try:
            
            answer = self.rag_chain.invoke(question)
            retriever = self.vector_store_service.as_retriever(k=3)
            source_docs = retriever.invoke(question)

            return {
                "question": question,
                "answer": answer.content if hasattr(answer, 'content') else str(answer),
                "sources": source_docs,
                "num_sources": len(source_docs),
            }
        except Exception as e:
            return {
                "question": question,
                "answer": f"Error processing query: {str(e)}",
                "sources": [],
                "num_sources": 0,
            }

    def ingest(self, pdf_path: Path):
        """Ingests a pdf document into the vector store"""
        try:
            # Extract pdf
            text = extract_pdf_text(pdf_path)
            print(f"✓ Extracted {len(text)} characters from PDF")

            # Generate the chunks
            chunks = chunk_text(text=text)
            print(f"✓ Created {len(chunks)} chunks from the document")

            # Store in vector DB
            self.vector_store_service.create_vector_store(chunks=chunks)
            print("✓ Successfully stored chunks in vector database")

            # Rebuild chain with updated retriever
            self.rag_chain = self._build_chain()
            print("✓ RAG chain rebuilt with updated documents")

            return chunks
        except Exception as e:
            print(f"Error during ingestion: {e}")
            raise

