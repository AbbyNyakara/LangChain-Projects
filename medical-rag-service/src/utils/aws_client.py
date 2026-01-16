from src.config.aws import AWSConfig
from typing import List
import uuid
from datetime import datetime
from pathlib import Path

class S3Handler:
    """Handles all S3 operations"""
    
    def __init__(self, config: AWSConfig):
        self.config = config
        self.s3 = config.get_s3_client()
        self.bucket = config.s3_bucket
        
    def upload_file(self, file_path: str, prefix: str = "uploads/") -> str:
        """Upload file to S3 with timestamp and UUID"""
        file_name = Path(file_path).name
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"{prefix}{timestamp}/{unique_id}/{file_name}"
        
        self.s3.upload_file(file_path, self.bucket, s3_key)
        return s3_key
    
    def save_text(self, text: str, prefix: str = "extracted/", filename: str = "document") -> str:
        """Save text content to S3"""
        unique_id = uuid.uuid4().hex[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        s3_key = f"{prefix}{timestamp}_{unique_id}_{filename}.txt"
        
        self.s3.put_object(
            Bucket=self.bucket,
            Key=s3_key,
            Body=text.encode('utf-8'),
            ContentType='text/plain'
        )
        return s3_key