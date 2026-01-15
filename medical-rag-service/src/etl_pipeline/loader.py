from src.etl_pipeline.embedder import EmbeddingPipeline, EmbeddingConfig, PineconeConfig
from pinecone import Pinecone
from pathlib import Path
import sys
from dataclasses import dataclass
import boto3
from typing import Dict, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


src_dir = Path(__file__).parent.parent
sys.path.insert(0, str(src_dir))


class LoadingPipeline:
    """
    Complete end-to-end loading pipeline
    Orchestrates extraction → chunking → embedding → loading
    """
    
    def __init__(
        self,
        embedding_config: EmbeddingConfig,
        pinecone_config: PineconeConfig,
        metadata_config: MetadataStoreConfig,
        s3_bucket: str
    ):
        self.embedding_pipeline = EmbeddingPipeline(
            embedding_config, pinecone_config, s3_bucket
        )
        self.metadata_store = MetadataStore(metadata_config)
        self.document_loader = DocumentLoader(
            self.embedding_pipeline,
            self.metadata_store,
            s3_bucket
        )
        self.vector_db_manager = VectorDatabaseManager(self.embedding_pipeline)
    
    def load_and_index(
        self,
        chunks_s3_key: str,
        original_filename: str,
        document_id: Optional[str] = None
    ) -> Dict:
        """
        Complete loading pipeline: Load → Embed → Store
        
        Args:
            chunks_s3_key: S3 key from DocumentChunkingPipeline
            original_filename: Original document filename
            document_id: Optional document ID
        
        Returns:
            Complete loading result
        """
        return self.document_loader.load_document(
            chunks_s3_key=chunks_s3_key,
            original_filename=original_filename,
            document_id=document_id
        )
    
    def search_documents(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict]:
        """Search across all loaded documents"""
        return self.vector_db_manager.search(query, top_k)
    
    def get_stats(self) -> Dict:
        """Get pipeline statistics"""
        return {
            'vector_db_stats': self.vector_db_manager.get_index_stats(),
            'timestamp': datetime.now().isoformat()
        }

