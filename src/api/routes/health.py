"""
Health check API routes.
"""

from datetime import datetime
from fastapi import APIRouter, HTTPException

from config.settings import settings
from services import http_client_service

router = APIRouter()


@router.get("/")
async def health_check():
    """Service health check"""
    try:
        auth_headers = await http_client_service.get_auth_headers()
        response = await http_client_service.client.get(f"{settings.BACKEND_URL}/", headers=auth_headers)
        response.raise_for_status()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "backend": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")