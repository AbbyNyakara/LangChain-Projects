"""
Docstring for src.vector_store.chroma

This module handles the Chroma DB storage and retrival
"""

from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

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
        Store Chunks in chroma DB   
        '''

        self.vector_store = Chroma.from_texts(
            texts=chunks,
            embedding=self.embeddings,
            collection_name=self.collection_name,
            persist_directory=self.persist_dir
        )

        # print(f"Created vector store with {len(chunks)} chunks")
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

    def retrieve_and_hydrate(self, query: str, k: int = 3) -> dict:

        retriever = self.as_retriever(k=k)

        # Invoke retriever (does vectorization + similarity search internally)
        docs = retriever.invoke(query)

        # Join all chunk content into single context string
        context = "\n\n".join([doc.page_content for doc in docs])

        return {
            "context": context,
            "source_docs": docs,
            "num_results": len(docs)
        }
