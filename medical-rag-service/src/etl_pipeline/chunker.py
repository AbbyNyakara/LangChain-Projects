"""
Document Chunking Pipeline with Metadata Storage
Chunks extracted text and stores in S3 + DynamoDB
"""

import boto3
import uuid
import logging
from typing import Dict, List, Optional
from langchain_text_splitters import RecursiveCharacterTextSplitter
from datetime import datetime
from pathlib import Path
import json
import sys
from dataclasses import dataclass

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

from src.etl_pipeline.extractor import DocumentOCRExtractor

@dataclass
class ChunkingConfig:
    """Configuration for chunking and metadata storage"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    s3_bucket: str = "medical-rag-docs-abigael-2026"
    documents_table: str = "medical-rag-documents"  
    chunks_table: str = "medical-rag-chunks"        
    region: str = "us-east-1"


class DocumentChunkingPipeline:
    """
    Complete pipeline: Extract → Chunk → Store in S3 → Store metadata in DynamoDB
    """

    def __init__(self, config: ChunkingConfig):
        self.config = config
        self.s3 = boto3.client('s3', region_name=config.region)
        self.dynamodb = boto3.resource('dynamodb', region_name=config.region)
        # self.table = self.dynamodb.Table(config.dynamodb_table)
        self.documents_table = self.dynamodb.Table(config.documents_table)  # ← New
        self.chunks_table = self.dynamodb.Table(config.chunks_table) 

    # ============ S3 Operations ============
    def fetch_extracted_text(self, s3_key: str) -> str:
        """Fetch extracted text from S3"""
        try:
            response = self.s3.get_object(Bucket=self.config.s3_bucket, Key=s3_key)
            text = response['Body'].read().decode('utf-8')
            logger.info("Fetched text from S3: %s", s3_key)
            return text
        except Exception as e:
            logger.error(f"Failed to fetch text: {e}")
            raise

    def save_chunks_to_s3(self, chunks: List[str], metadata_list: List[Dict], original_filename: str) -> str:
        """Save chunks and metadata to S3"""
        try:
            base_name = Path(original_filename).stem
            unique_id = uuid.uuid4().hex[:8]
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            s3_key = f"chunks/{timestamp}/{unique_id}/{base_name}_chunks.json"

            chunks_data = {
                'document_name': original_filename,
                'processing_date': datetime.now().isoformat(),
                'total_chunks': len(chunks),
                'chunks': [
                    {'text': chunk, 'metadata': meta}
                    for chunk, meta in zip(chunks, metadata_list)
                ]
            }

            self.s3.put_object(
                Bucket=self.config.s3_bucket,
                Key=s3_key,
                Body=json.dumps(chunks_data, indent=2).encode('utf-8'),
                ContentType='application/json',
                ServerSideEncryption='AES256'
            )
            logger.info(f"Saved chunks to S3: {s3_key}")
            return s3_key
        except Exception as e:
            logger.error(f"Failed to save chunks: {e}")
            raise

    def fetch_chunks_from_s3(self, chunks_s3_key: str) -> List[Dict]:
        """Fetch chunks from S3 for vectorization"""
        try:
            response = self.s3.get_object(Bucket=self.config.s3_bucket, Key=chunks_s3_key)
            chunks_data = json.loads(response['Body'].read().decode('utf-8'))
            return chunks_data['chunks']
        except Exception as e:
            logger.error(f"Failed to fetch chunks: {e}")
            raise

    # ============ Chunking Operations ============

    def chunk_text(self, text: str) -> List[str]:
        """Chunk text using recursive character splitter"""
        try:
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=self.config.chunk_size,
                chunk_overlap=self.config.chunk_overlap,
                length_function=len,
                is_separator_regex=False,
                add_start_index=True
            )
            chunks = splitter.split_text(text)
            logger.info(f"Created {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error(f"Chunking failed: {e}")
            raise

    def create_chunk_metadata(self, chunks: List[str], original_filename: str, text_s3_key: str, document_id: Optional[str] = None) -> List[Dict]:
        """Create metadata for each chunk"""
        if document_id is None:
            document_id = str(uuid.uuid4())

        metadata_list = []
        for idx, chunk in enumerate(chunks):
            metadata = {
                'chunk_id': f"{document_id}_chunk_{idx:04d}",
                'document_id': document_id,
                'original_filename': original_filename,
                'source_s3_key': text_s3_key,
                'chunk_index': idx,
                'total_chunks': len(chunks),
                'chunk_size': len(chunk),
                'created_at': datetime.now().isoformat()
            }
            metadata_list.append(metadata)

        return metadata_list, document_id

    # ============ DynamoDB Operations ============

    def store_document_metadata(self, document_id: str, filename: str, chunks_count: int, s3_key: str) -> bool:
        """Store document-level metadata in DynamoDB"""
        try:
            self.documents_table.put_item(Item={
                'document_id': document_id,
                'filename': filename,
                'chunks_count': chunks_count,
                's3_key': s3_key,
                'created_at': datetime.now().isoformat()
            })
            logger.info("Stored document metadata: %s", document_id)
            return True
        except Exception as e:
            logger.error(f"Failed to store document metadata: {e}")
            raise

    def store_chunk_metadata(self, chunk_metadata_list: List[Dict]) -> bool:
        """Store chunk-level metadata in DynamoDB"""
        try:
            with self.chunks_table.batch_writer() as batch:  # ← Remove batch_size
                for chunk_meta in chunk_metadata_list:
                    batch.put_item(Item=chunk_meta)
            logger.info(f"Stored {len(chunk_metadata_list)} chunk metadata items")
            return True
        except Exception as e:
            logger.error(f"Failed to store chunk metadata: {e}")
            raise


    def get_document_metadata(self, document_id: str) -> Dict:
        """Retrieve document metadata"""
        try:
            response = self.documents_table.get_item(Key={'document_id': document_id})
            return response.get('Item', {})
        except Exception as e:
            logger.error(f"Failed to get document metadata: {e}")
            raise

    def query_chunks_by_document(self, document_id: str) -> List[Dict]:
        """Query all chunks for a document"""
        try:
            response = self.chunks_table.query(
                KeyConditionExpression='document_id = :doc_id',
                ExpressionAttributeValues={':doc_id': document_id}
            )
            return response.get('Items', [])
        except Exception as e:
            logger.error(f"Failed to query chunks: {e}")
            raise

    # ============ Main Pipeline ============

    def process_document(self, text_s3_key: str, original_filename: str, document_id: Optional[str] = None) -> Dict:
        """
        Complete pipeline: Fetch → Chunk → Store in S3 → Store in DynamoDB
        
        Args:
            text_s3_key: S3 key of extracted text
            original_filename: Original document filename
            document_id: Optional document ID
        
        Returns:
            Processing result dictionary
        """
        try:
            logger.info("Starting processing: %s", original_filename)

            # Step 1: Fetch extracted text
            text = self.fetch_extracted_text(text_s3_key)

            # Step 2: Chunk text
            chunks = self.chunk_text(text)

            # Step 3: Create metadata
            chunk_metadata_list, doc_id = self.create_chunk_metadata(
                chunks, original_filename, text_s3_key, document_id
            )

            # Step 4: Save chunks to S3
            chunks_s3_key = self.save_chunks_to_s3(chunks, chunk_metadata_list, original_filename)

            # Step 5: Store document metadata in DynamoDB
            self.store_document_metadata(
                document_id=doc_id,
                filename=original_filename,
                chunks_count=len(chunks),
                s3_key=chunks_s3_key
            )

            # Step 6: Store chunk metadata in DynamoDB
            self.store_chunk_metadata(chunk_metadata_list)

            result = {
                'success': True,
                'document_id': doc_id,
                'original_file': original_filename,
                'chunks_s3_key': chunks_s3_key,
                'total_chunks': len(chunks),
                'total_characters': sum(len(c) for c in chunks),
                'avg_chunk_size': sum(len(c) for c in chunks) / len(chunks) if chunks else 0,
                'timestamp': datetime.now().isoformat()
            }

            logger.info("Processing complete chunks created")
            return result

        except Exception as e:
            logger.error("Document processing failed: %s", e)
            return {
                'success': False,
                'error': str(e),
                'original_file': original_filename
            }

# test_file = "/Users/abigaelmogusu/projects/LangChain-Projects/medical-rag-service/data/fake-aps.pdf"

# extractor = DocumentOCRExtractor(bucket="medical-rag-docs-abigael-2026", region="us-east-1")
# print("TESTING DOCUMENT CHUNKING PIPELINE")

# extraction_result = extractor.process_document(test_file)
# print(f"  - Uploaded to S3: {extraction_result['uploaded_to']}")
# extracted_text_s3_key = extraction_result['saved_text_to']
# original_filename = extraction_result['original_file']

# ## Chunk text
# chunker = DocumentChunkingPipeline(ChunkingConfig())  # ← Add ()
# chunking_result = chunker.process_document(text_s3_key=extracted_text_s3_key, original_filename=original_filename)

# if not chunking_result['success']:
#     print(f"✗ Chunking failed: {chunking_result['error']}")
   
# print(f"✓ Chunking successful!")
# print(f"  - Document ID: {chunking_result['document_id']}")
# print(f"  - Total chunks: {chunking_result['total_chunks']}")
# print(f"  - Total characters: {chunking_result['total_characters']}")
# print(f"  - Average chunk size: {chunking_result['avg_chunk_size']:.0f} chars")
# print(f"  - Chunks S3 key: {chunking_result['chunks_s3_key']}\n")

# document_id = chunking_result['document_id']
# chunks_s3_key = chunking_result['chunks_s3_key']

# ### S3 STORAGE:
# chunks_data = chunker.fetch_chunks_from_s3(chunks_s3_key)
# print(f"✓ Chunks retrieved from S3: {len(chunks_data)} items\n")

# # Show first chunk
# first_chunk = chunks_data[0]
# print(f"  First chunk preview:")
# print(f"    - Chunk ID: {first_chunk['metadata']['chunk_id']}")
# print(f"    - Chunk size: {first_chunk['metadata']['chunk_size']} chars")
# print(f"    - Text preview: {first_chunk['text'][:100]}...\n")

# ## DynamoDB 
# doc_metadata = chunker.get_document_metadata(document_id)
            
# if doc_metadata:
#     print(f"✓ Document metadata stored in DynamoDB!")
#     print(f"  - Document ID: {doc_metadata.get('document_id')}")
#     print(f"  - Filename: {doc_metadata.get('filename')}")
#     print(f"  - Chunks count: {doc_metadata.get('chunks_count')}")
#     print(f"  - Created at: {doc_metadata.get('created_at')}\n")
# else:
#     print(f"✗ Document metadata not found in DynamoDB\n")
    
# chunk_metadata_list = chunker.query_chunks_by_document(document_id)
            
# print(f"✓ Chunk metadata stored in DynamoDB: {len(chunk_metadata_list)}")
