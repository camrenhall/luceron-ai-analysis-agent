"""
Configuration settings for the document analysis system.
"""

import os
from typing import Optional, Dict, Any


class Settings:
    """Application settings loaded from environment variables."""
    
    def __init__(self):
        # API Keys
        self.ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
        
        # Backend Configuration
        self.BACKEND_URL: Optional[str] = os.getenv("BACKEND_URL")
        
        # OAuth2 configuration - private key from environment, service details static
        self.ANALYSIS_AGENT_PRIVATE_KEY: Optional[str] = os.getenv("ANALYSIS_AGENT_PRIVATE_KEY")
        
        # Static Luceron service configuration
        self.LUCERON_SERVICE_ID: str = "luceron_ai_analysis_agent"
        
        # Communications Agent Configuration
        self.COMMUNICATIONS_AGENT_URL: Optional[str] = os.getenv("COMMUNICATIONS_AGENT_URL")
        
        # Application Configuration
        self.PORT: int = int(os.getenv("PORT", 8080))
        
        # Validate required settings
        self._validate_required_settings()
    
    def _validate_required_settings(self) -> None:
        """Validate that all required environment variables are set."""
        required_settings = [
            ("ANTHROPIC_API_KEY", self.ANTHROPIC_API_KEY),
            ("BACKEND_URL", self.BACKEND_URL),
        ]
        
        for setting_name, setting_value in required_settings:
            if not setting_value:
                raise ValueError(f"{setting_name} environment variable is required")


def get_luceron_config() -> Optional[Dict[str, Any]]:
    """
    Get Luceron OAuth2 configuration with private key from environment
    
    Returns:
        Configuration dictionary or None if private key not available
    """
    if not settings.ANALYSIS_AGENT_PRIVATE_KEY:
        return None
        
    return {
        'service_id': settings.LUCERON_SERVICE_ID,
        'private_key': settings.ANALYSIS_AGENT_PRIVATE_KEY,
        'base_url': settings.BACKEND_URL
    }


# Global settings instance
settings = Settings()