import boto3
import uuid
import logging
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
import json


import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent  # Goes up 3 levels to medical-rag-service/
sys.path.insert(0, str(project_root))
#from extractor.document_extractor import DocumentOCRExtractor
from src.extractor.document_extractor import DocumentOCRExtractor


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DocumentChunkingPipeline:
    '''
    Complete Pipeline:
    Fetch Extracted Data from S3 Bucket-> Chunk Semantically -> Store Chunks in S3
    '''

    def __init__(self, bucket: str, region='us-easy-1'):
        self.bucket = bucket
        self.region = region
        self.s3 = boto3.client('s3', region_name=region)

    def fetch_extracted_text(self, s3_key: str):
        """
        Fetch the extracted text file from S3.
        This is the output from DocumentOCRExtractor.save_extracted_text()

        Args:
            text_s3_key: S3 key of the extracted text file (e.g 'extracted/20260108_abc123_fake-aps.txt')

        Returns:
            Text content as string
        """

        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=s3_key)
            text = response['Body'].read().decode('utf-8')
            return text
        except Exception as e:
            logging.error("Failed to process user: %s", e)
            raise

    def chunk_text_semantically(self, text:str, chunk_size: int = 400,chunk_overlap: int = 200,separators: Optional[List[str]] = None):
        '''
        Chunks text semantically using recursive character text splitting
        
        - Read more on how best to maintain context?
        Returns List of text chunks
        '''
        try:
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=separators,
                length_function=len,
                is_separator_regex=False,
                add_start_index=True # to track where chunks come from in original text:
            )
            chunks = text_splitter.split_text(text)
            return chunks
        except Exception as e:
            logger.error("Error Chunking data: %s", e)
            raise

    def create_chunk_metadata(self, chunks: List[str],original_filename: str,text_s3_key: str,document_id: Optional[str] = None):
        '''
        Create metadata for each chunk to preserve context for vectorization.
        This metadata will be stored with the chunks for later retrieval and filtering.
        Returns:
            List of metadata dictionaries, one per chunk
        '''
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
                'created_at': datetime.now().isoformat(),
                'processing_stage': 'chunked'  # Tracking pipeline stage
            }
            metadata_list.append(metadata)

        return metadata_list
    
    def save_chunks_to_s3(self, chunks: List[str], metadata_list, original_filename:str):
        """
        Saves chunks and their metadata to an s3 bucket. Creates single file that contains
        chunks and their metadata for easy retrieval
        Returns:
            Dictionary with S3 key(location)
        """
        base_name = Path(original_filename).stem
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        chunks_s3_key = f"chunks/{timestamp}/{unique_id}/{base_name}_chunks.json"

        # combine chunks + metadata
        chunks_data = {
            'document_name': original_filename,
            'processing_date': datetime.now().isoformat(),
            'total_chunks': len(chunks),
            'chunks': [
                {
                    'text': chunk,
                    'metadata': meta
                }
                for chunk, meta in zip(chunks, metadata_list)
            ]
        }

        try:
            self.s3.put_object(
                Bucket = self.bucket, 
                Key = chunks_s3_key, 
                Body=json.dumps(chunks_data, indent=2).encode('utf-8'),
                ContentType='application/json',
                ServerSideEncryption='AES256'
            )

            return {
                'chunks_s3_key': chunks_s3_key,
                'total_chunks': len(chunks),
                'total_characters': sum(len(c) for c in chunks)
            }
        
        except Exception as e:
            logger.error("Failed to save Chunks %s", e)
            raise

    def process_document(self, text_s3_key, original_filename, chunk_size:int=1000, chunk_overlap:int = 200):
        """
        Complete pipeline: Fetch → Chunk → Save
        
        Args:
            text_s3_key: S3 key of extracted text from DocumentOCRExtractor
            original_filename: Original document filename
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between chunks
        
        Returns:
            Processing results dictionary
        """
        try:
            # TODO 1: Fetch extracted text from S3
            text = self.fetch_extracted_text(text_s3_key)

            chunks = self.chunk_text_semantically(
                text,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )

            metadata_list = self.create_chunk_metadata(
                chunks,
                original_filename,
                text_s3_key
            )
            save_result = self.save_chunks_to_s3(
                chunks,
                metadata_list,
                original_filename
            )

            return {
                'success': True,
                'original_file': original_filename,
                'source_text_key': text_s3_key,
                'chunks_s3_key': save_result['chunks_s3_key'],
                'total_chunks': save_result['total_chunks'],
                'total_characters': save_result['total_characters'],
                'avg_chunk_size': save_result['total_characters'] / save_result['total_chunks'],
                'processing_date': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Document processing failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'original_file': original_filename
            }
        
    def fetch_chunks_for_vectorization(self, chunks_s3_key: str) -> List[Dict]:
        '''
        Fetch chunks and metadata from S3 for vectorization.
        Called by the vectorization/embedding step.
        
        Args:
            chunks_s3_key: S3 key of the chunks JSON file
        
        Returns:
            List of dicts with 'text' and 'metadata' keys
        '''
        try:
            response = self.s3.get_object(Bucket=self.bucket, Key=chunks_s3_key)
            chunks_data = json.loads(response['Body'].read().decode('utf-8'))
            return chunks_data['chunks']
        
        except Exception as e:
            logger.error("Failed to fetch chunks %s", e)
            raise



    



