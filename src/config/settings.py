"""
Configuration settings for the document analysis system.
"""

import os
from typing import Optional


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        # API Keys
        self.ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        self.OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
        
        # Backend Configuration
        self.BACKEND_URL: Optional[str] = os.getenv("BACKEND_URL")
        
        # AWS Configuration
        self.AWS_ACCESS_KEY_ID: Optional[str] = os.getenv("AWS_ACCESS_KEY_ID")
        self.AWS_SECRET_ACCESS_KEY: Optional[str] = os.getenv("AWS_SECRET_ACCESS_KEY")
        self.AWS_REGION: str = os.getenv("AWS_REGION", "us-east-1")
        self.S3_BUCKET_NAME: Optional[str] = os.getenv("S3_BUCKET_NAME")
        
        # Application Configuration
        self.PORT: int = int(os.getenv("PORT", 8080))
        
        # Validate required settings
        self._validate_required_settings()
    
    def _validate_required_settings(self) -> None:
        """Validate that all required environment variables are set."""
        required_settings = [
            ("ANTHROPIC_API_KEY", self.ANTHROPIC_API_KEY),
            ("OPENAI_API_KEY", self.OPENAI_API_KEY),
            ("BACKEND_URL", self.BACKEND_URL),
            ("AWS_ACCESS_KEY_ID", self.AWS_ACCESS_KEY_ID),
            ("AWS_SECRET_ACCESS_KEY", self.AWS_SECRET_ACCESS_KEY),
            ("S3_BUCKET_NAME", self.S3_BUCKET_NAME),
        ]
        
        for setting_name, setting_value in required_settings:
            if not setting_value:
                raise ValueError(f"{setting_name} environment variable is required")


# Global settings instance
settings = Settings()