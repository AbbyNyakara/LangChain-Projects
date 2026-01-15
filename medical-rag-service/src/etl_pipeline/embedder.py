"""
This module handles the 
"""
import boto3
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from pinecone import Pinecone, ServerlessSpec
import sys
import os
from pathlib import Path
import json
import logging
from datetime import datetime
src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))


load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    embedding_dimension: int = 1536
    batch_size: int = 32
    max_retries: int = 3
    retry_delay: float = 1.0


@dataclass
class PineconeConfig:
    """Configuration for Pinecone connection"""
    api_key: str = os.environ["PINECONE_API_KEY"]
    environment: str = "us-east-1"
    index_name: str = "medical-rag-index"  # the vector db name of this project
    metric: str = "cosine"
    dimension: int = 1536
    spec: Optional[Dict] = None  # For serverless config


class EmbeddingPipeline:
    def __init__(self, embedding_config: EmbeddingConfig, pinecone_config: PineconeConfig, s3_bucket: str):
        self.s3 = boto3.client('s3')
        self.bucket = s3_bucket

        # OpenAI Embeddings
        self.embeddings = OpenAIEmbeddings(model=embedding_config.model)

        # Pinecone
        self.pc = Pinecone(api_key=pinecone_config.api_key)
        self._setup_pinecone_index(pinecone_config)
        self.index = self.pc.Index(pinecone_config.index_name)

        self.config = embedding_config

    def _setup_pinecone_index(self, config=PineconeConfig):
        """Create Pinecone Index(database), if it doesnt exist"""
        indexes = self.pc.list_indexes()
        if config.index_name not in [idx.name for idx in indexes.indexes]:
            self.pc.create_index(
                name=config.index_name,
                dimension=config.metric,
                metric=config.metric,
                spec=ServerlessSpec(cloud="aws", region=config.environment)
            )

    def load_chunks(self, s3_key: str) -> List[Dict]:
        """Load chunks JSON from S3"""
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            data = json.loads(response['Body'].read().decode('utf-8'))
            chunks = data.get('chunks', [])
            return chunks
        except Exception as e:
            logger.error("Failed to load chunks:, %s", e)
            raise

    def embed_chunks(self, chunks: List[Dict]) -> List[tuple]:
        """
        Generate embeddings for chunks
        Returns: [(chunk_id, embedding, metadata), ...]
        """
        results = []

        # Process in batches
        for i in range(0, len(chunks), self.config.batch_size):
            batch = chunks[i:i + self.config.batch_size]
            texts = [c['text'] for c in batch]

            try:
                embeddings = self.embeddings.embed_documents(texts)

                # Combine with metadata
                for chunk, embedding in zip(batch, embeddings):
                    chunk_id = chunk['metadata']['chunk_id']
                    results.append((chunk_id, embedding, chunk['metadata']))

                logger.info("Embedded batch :%s", i //
                            self.config.batch_size + 1)
            except Exception as e:
                logger.error("Embedding failed: %s", e)
                raise

        return results

    def store_embeddings(self, vectors: List[tuple]) -> Dict:
        """
        Store embeddings in Pinecone
        Args: [(id, embedding, metadata), ...]
        """
        try:
            # Format for Pinecone
            pinecone_vectors = [
                {"id": vid, "values": emb, "metadata": meta}
                for vid, emb, meta in vectors
            ]

            # Upsert in batches
            for i in range(0, len(pinecone_vectors), 100):
                batch = pinecone_vectors[i:i + 100]
                self.index.upsert(vectors=batch)

            logger.info("Stored vectors in Pinecone")
            return {"success": True, "count": len(vectors)}
        except Exception as e:
            logger.error("Storage failed: %s", e)
            raise

    def process_document(self, chunks_s3_key: str) -> Dict:
        """
        Complete pipeline: Load → Embed → Store
        """
        try:
            # step1: Load chunks from s3
            chunks = self.load_chunks(s3_key=chunks_s3_key)
            # Step 2: Generate the embeddings
            vectors = self.embed_chunks(chunks)
            # TODO 3- Store in pinecone:
            result = self.store_embeddings(vectors)
            return {
                'success': True,
                'chunks': len(chunks),
                'stored': result['count'],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error("Pipeline failed:, %s", e)
            return {'success': False, 'error': str(e)}

    def search(self, query: str, top_k: int = 3) -> List[Dict]:
        """Embedd user query and similarity Search for similar chunks"""
        try:
            # Embed query
            query_embedding = self.embeddings.embed_query(query)

            # Search Pinecone
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )

            # Format results
            return [
                {
                    'id': match.id,
                    'score': match.score,
                    'metadata': match.metadata
                }
                for match in results.matches
            ]
        except Exception as e:
            logger.error("Search failed %s", e)
            raise
