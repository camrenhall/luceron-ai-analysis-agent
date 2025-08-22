"""
Simplified OAuth2-authenticated client for Luceron Agent API
"""
import requests
import jwt
from datetime import datetime, timedelta
from cryptography.hazmat.primitives import serialization
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class LuceronClient:
    """Simplified OAuth2-authenticated client for Luceron Agent API"""
    
    def __init__(self, service_id: str, private_key_pem: str, base_url: str):
        """
        Initialize client with service configuration
        
        Args:
            service_id: Service identifier (e.g., "luceron_ai_communications_agent")
            private_key_pem: RSA private key in PEM format
            base_url: Luceron API base URL
        """
        self.service_id = service_id
        self.base_url = base_url
        
        # Load private key
        try:
            self.private_key = serialization.load_pem_private_key(
                private_key_pem.encode(),
                password=None
            )
        except Exception as e:
            raise ValueError(f"Failed to load private key: {e}")
        
        # Token caching
        self._access_token: Optional[str] = None
        self._token_expires_at: Optional[datetime] = None
        
        logger.info(f"Initialized LuceronClient for service: {self.service_id}")
    
    def _create_service_jwt(self) -> str:
        """Create service authentication JWT"""
        now = datetime.utcnow()
        payload = {
            'iss': self.service_id,
            'sub': self.service_id,
            'aud': 'luceron-auth-server',
            'iat': int(now.timestamp()),
            'exp': int((now + timedelta(minutes=5)).timestamp())
        }
        return jwt.encode(payload, self.private_key, algorithm='RS256')
    
    def _get_access_token(self) -> str:
        """Get valid access token (cached or fresh)"""
        # Use cached token if still valid
        if (self._access_token and self._token_expires_at and
            datetime.utcnow() < self._token_expires_at):
            return self._access_token

        # Request new token
        service_jwt = self._create_service_jwt()
        
        response = requests.post(f"{self.base_url}/oauth2/token", data={
            'grant_type': 'client_credentials',
            'client_assertion_type': 'urn:ietf:params:oauth:client-assertion-type:jwt-bearer',
            'client_assertion': service_jwt
        })

        if response.status_code == 200:
            data = response.json()
            self._access_token = data['access_token']
            # Cache with 60 second buffer to avoid expiration edge cases
            self._token_expires_at = datetime.utcnow() + timedelta(seconds=data['expires_in'] - 60)
            logger.info(f"Access token obtained, expires in {data['expires_in']} seconds")
            return self._access_token
        else:
            raise Exception(f"Token request failed: {response.status_code} - {response.text}")

    def query(self, natural_language: str) -> dict:
        """
        Query Luceron's agent database with natural language
        
        Args:
            natural_language: Natural language query
            
        Returns:
            Database query response
        """
        token = self._get_access_token()

        response = requests.post(
            f"{self.base_url}/agent/db",
            headers={
                'Authorization': f'Bearer {token}',
                'Content-Type': 'application/json'
            },
            json={'natural_language': natural_language}
        )

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Query failed: {response.status_code} - {response.text}")
    
    def health_check(self) -> bool:
        """Test authentication and connectivity"""
        try:
            result = self.query("Show me system status")
            return result.get('ok', False)
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return False


class LuceronClientWithRetry(LuceronClient):
    """Enhanced client with retry logic"""
    
    def query_with_retry(self, natural_language: str, max_retries: int = 3) -> Optional[dict]:
        """Query with automatic retry on failure"""
        for attempt in range(max_retries + 1):
            try:
                return self.query(natural_language)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(f"Query failed after {max_retries} retries: {e}")
                    return None
                
                # Handle token expiration
                if "401" in str(e):
                    logger.warning(f"Token expired, clearing cache (attempt {attempt + 1})")
                    self._access_token = None  # Force token refresh
                elif "403" in str(e):
                    logger.error(f"Insufficient permissions: {e}")
                    return None  # Don't retry permission errors
                
                logger.warning(f"Query failed, retrying... (attempt {attempt + 1})")
        
        return None