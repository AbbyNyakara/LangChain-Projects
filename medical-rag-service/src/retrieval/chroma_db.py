"""
Docstring for src.vector_store.chroma
This module handles the Chroma DB storage and retrival
"""
import sys
from pathlib import Path
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from extractor.extractor import PDFExtractor
from chunking.recursive_chunking import TextChunker


load_dotenv()


class VectorStoreService:
    '''
    Manages all vector store operations
    '''

    def __init__(self, collection_name: str = 'medical_notes', persist_directory: str = "./chroma_db"):
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        self.collection_name = collection_name
        self.persist_dir = persist_directory
        self.vector_store = None

    def create_vector_store(self, chunks: list[str]):
        '''
        Store Chunks in chroma DB  in the root directory
        '''

        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir
        )

        print(f"Created vector store with {len(chunks)} chunks")
        return self.vector_store

    def load_vector_store(self):
        '''
        Loads the stored vector database
        '''
        self.vector_store = Chroma(
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
            collection_name=self.collection_name
        )

        return self.vector_store

    def as_retriever(self, k: int = 3, search_type: str = "similarity"):
        """
        Returns Langchain retriever backed by the vector store
        """
        if self.vector_store is None:
            self.load_vector_store()
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs={'k': k}
        )

    def retrieve_and_hydrate(self, query: str, k: int=3 ) -> dict:
        retriever = self.as_retriever(k=k)
        docs = retriever.invoke(query)  # vectorization + similarity search
        context = "\n\n".join(
            [doc.page_content for doc in docs])  # Join chunks

        return {
            "context": context,
            "source_docs": docs,
            "num_results": len(docs)
        }

# Usage
extractor = PDFExtractor()
extracted_text = extractor.extract()
chunker = TextChunker()
chunks = chunker.chunk(extracted_text)
store = VectorStoreService()
print(store.create_vector_store(chunks))
print(store.retrieve_and_hydrate(query="What is the patient's name?"))