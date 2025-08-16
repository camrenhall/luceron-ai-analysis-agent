"""
S3 service for document storage operations.
"""

import logging
import boto3
from botocore.exceptions import NoCredentialsError, ClientError

from config import settings

logger = logging.getLogger(__name__)


class S3Service:
    """Service for interacting with Amazon S3."""
    
    def __init__(self):
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
            region_name=settings.AWS_REGION
        )
        self.bucket_name = settings.S3_BUCKET_NAME
        logger.info("S3 service initialized")
    
    async def download_document(self, s3_key: str) -> bytes:
        """Download document from S3"""
        if not s3_key:
            raise ValueError("No S3 key provided")
        
        logger.info(f"📥 Downloading document from S3: {s3_key}")
        
        try:
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=s3_key)
            image_data = response['Body'].read()
            
            logger.info(f"📥 Downloaded {len(image_data)} bytes from S3")
            return image_data
            
        except NoCredentialsError:
            raise ValueError("AWS credentials not configured")
        except ClientError as e:
            raise ValueError(f"S3 download failed: {e}")


# Global S3 service instance
s3_service = S3Service()