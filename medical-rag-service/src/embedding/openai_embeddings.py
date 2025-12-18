from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

load_dotenv()


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectors = embeddings.embed_documents(chunks)
    return vectors
