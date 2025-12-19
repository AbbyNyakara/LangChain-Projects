from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


class EmbeddingService:
    def __init__(self, model: str = "text-embedding-3-small"):
        """
        Initialize embedding service with selected model
        """
        self.model = model
        self.embeddings = OpenAIEmbeddings(model=model)

    def embed_chunks(self, chunks: list[str]) -> list[list[float]]:
        """
        Embed multiple text chunks.
        Args:chunks: List of text strings to embed.  
        Returns:List of embedding vectors
        p:s - this is directly handled in the create_vector_store(embedds and stores in chroma db)
        """

        vectors = self.embeddings.embed_documents(chunks)
        return vectors

    def embed_user_query(self, query: str):
        """
        Embed a user's single query
        Args: query - The user query passed in through as a prompt
        Returns: vector single embedded vector
        """
        return self.embeddings.embed_query(query)
