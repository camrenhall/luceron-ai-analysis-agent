"""
HTTP client management service.
"""

import logging
import httpx

logger = logging.getLogger(__name__)


class HTTPClientService:
    """Manages the global HTTP client for the application."""
    
    def __init__(self):
        self._client: httpx.AsyncClient = None
    
    async def init_client(self) -> None:
        """Initialize HTTP client"""
        self._client = httpx.AsyncClient(timeout=60.0)
        logger.info("HTTP client initialized")
    
    async def close_client(self) -> None:
        """Close HTTP client"""
        if self._client:
            await self._client.aclose()
            logger.info("HTTP client closed")
    
    @property
    def client(self) -> httpx.AsyncClient:
        """Get the HTTP client instance."""
        if not self._client:
            raise RuntimeError("HTTP client not initialized. Call init_client() first.")
        return self._client


# Global HTTP client service instance
http_client_service = HTTPClientService()