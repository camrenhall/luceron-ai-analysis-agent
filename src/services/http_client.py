"""
HTTP client management service.
"""

import logging
import httpx

from config.settings import get_luceron_config
from services.oauth2_client import LuceronClient

logger = logging.getLogger(__name__)


class HTTPClientService:
    """Manages the global HTTP client for the application."""
    
    def __init__(self):
        self._client: httpx.AsyncClient = None
        self._luceron_client: LuceronClient = None
    
    async def init_client(self) -> None:
        """Initialize HTTP client with OAuth2 authentication"""
        # Initialize OAuth2 client
        config = get_luceron_config()
        if config:
            self._luceron_client = LuceronClient(
                service_id=config['service_id'],
                private_key_pem=config['private_key'],
                base_url=config['base_url']
            )
            logger.info("OAuth2 client initialized")
        else:
            logger.warning("OAuth2 configuration not available")
        
        # Initialize HTTP client without static auth header (we'll add tokens per request)
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info("HTTP client initialized")
    
    async def close_client(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            logger.info("HTTP client closed")
    
    async def get_auth_headers(self) -> dict:
        """Get authorization headers for API requests"""
        if self._luceron_client:
            try:
                token = self._luceron_client._get_access_token()
                return {"Authorization": f"Bearer {token}"}
            except Exception as e:
                logger.error(f"Failed to get access token: {e}")
                return {}
        return {}
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client instance."""
        if not self._client:
            raise RuntimeError("HTTP client not initialized. Call init_client() first.")
        return self._client
    
    @property
    def luceron_client(self) -> LuceronClient:
        """Get the Luceron OAuth2 client instance."""
        if not self._luceron_client:
            raise RuntimeError("Luceron client not initialized. Check OAuth2 configuration.")
        return self._luceron_client


# Global HTTP client service instance
http_client_service = HTTPClientService()