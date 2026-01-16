from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain_openai import OpenAIEmbeddings

load_dotenv()

query = "what is the patient's name?"
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectors = embeddings.embed_query(query)

# pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
# index = pc.Index(name="medical-rag-index")
# response = index.query(
#     vector=vectors, 
#     top_k=3, 
#     include_metadata=True
# )

# for match in response["matches"]:
#     print("Score:", match["score"])
#     print("Text:", match["metadata"].get("text"))
#     print("-" * 50)


from pinecone import Pinecone
import os

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# indexes = pc.list_indexes()
# print(indexes)

index = pc.Index("medical-rag-index")
stats = index.describe_index_stats()
print(stats)