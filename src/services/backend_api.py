"""
Backend API service for data management.
"""

import logging
from datetime import datetime
from typing import Dict, Optional
import json

from config import settings
from .http_client import http_client_service
import httpx

logger = logging.getLogger(__name__)


def _log_api_error(operation: str, url: str, request_data: Optional[Dict] = None, response: Optional[httpx.Response] = None, exception: Optional[Exception] = None):
    """Enhanced error logging for API operations"""
    error_context = {
        "operation": operation,
        "url": url,
        "timestamp": datetime.now().isoformat()
    }
    
    if request_data:
        error_context["request_data"] = request_data
    
    if response:
        error_context.update({
            "status_code": response.status_code,
            "response_headers": dict(response.headers),
            "response_text": response.text[:1000] if response.text else None  # Limit to 1000 chars
        })
        
        try:
            response_json = response.json()
            error_context["response_json"] = response_json
        except:
            pass
    
    if exception:
        error_context["exception_type"] = type(exception).__name__
        error_context["exception_message"] = str(exception)
    
    logger.error(f"🚨 Backend API Error - {operation}")
    logger.error(f"Error Details: {json.dumps(error_context, indent=2, default=str)}")


class BackendAPIService:
    """Service for interacting with the backend API."""
    
    def __init__(self):
        self.backend_url = settings.BACKEND_URL
    

    async def get_document_metadata(self, document_id: str) -> dict:
        """Get document metadata from backend"""
        url = f"{self.backend_url}/api/documents/{document_id}"
        try:
            response = await http_client_service.client.get(url)
            if response.status_code == 404:
                raise ValueError(f"Document {document_id} not found")
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                raise ValueError(f"Document {document_id} not found")
            _log_api_error("get_document_metadata", url, response=e.response, exception=e)
            raise
        except Exception as e:
            _log_api_error("get_document_metadata", url, exception=e)
            raise
    

    async def get_case_context(self, case_id: str) -> dict:
        """Retrieve case details and context"""
        logger.info(f"🔍 Retrieving context for case {case_id}")
        url = f"{self.backend_url}/api/cases/{case_id}"
        
        try:
            response = await http_client_service.client.get(url)
            
            if response.status_code == 404:
                # Return basic context if case not found in detail
                case_context = {
                    "case_id": case_id,
                    "case_type": "family_law_financial_discovery",
                    "priority": "standard",
                    "focus_areas": ["asset_identification", "income_verification", "expense_analysis"],
                    "document_requirements": ["Financial statements", "Tax returns", "Bank records"]
                }
            else:
                response.raise_for_status()
                case_context = response.json()
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                # Return basic context if case not found in detail
                case_context = {
                    "case_id": case_id,
                    "case_type": "family_law_financial_discovery",
                    "priority": "standard",
                    "focus_areas": ["asset_identification", "income_verification", "expense_analysis"],
                    "document_requirements": ["Financial statements", "Tax returns", "Bank records"]
                }
            else:
                _log_api_error("get_case_context", url, response=e.response, exception=e)
                raise
        except Exception as e:
            _log_api_error("get_case_context", url, exception=e)
            raise
        
        logger.info(f"🔍 Retrieved context for case {case_id}")
        return case_context
    

    async def get_requested_documents(self, case_id: str) -> Optional[Dict]:
        """Get requested documents for a case"""
        url = f"{self.backend_url}/api/cases/{case_id}"
        logger.info(f"📋 Fetching requested documents for case {case_id}")
        
        try:
            response = await http_client_service.client.get(url)
            if response.status_code == 404:
                logger.warning(f"Case {case_id} not found")
                return None
            response.raise_for_status()
            
            case_data = response.json()
            logger.info(f"✅ Retrieved {len(case_data.get('requested_documents', []))} requested documents for case {case_id}")
            return case_data
            
        except httpx.HTTPStatusError as e:
            _log_api_error("get_requested_documents", url, response=e.response, exception=e)
            return None
        except Exception as e:
            _log_api_error("get_requested_documents", url, exception=e)
            return None

    async def update_document_status(
        self, 
        requested_doc_id: str, 
        is_completed: Optional[bool] = None,
        is_flagged_for_review: Optional[bool] = None,
        notes: Optional[str] = None
    ) -> Optional[Dict]:
        """Update the status of a requested document"""
        url = f"{self.backend_url}/api/cases/documents/{requested_doc_id}"
        
        update_data = {}
        if is_completed is not None:
            update_data["is_completed"] = is_completed
        if is_flagged_for_review is not None:
            update_data["is_flagged_for_review"] = is_flagged_for_review
        if notes is not None:
            update_data["notes"] = notes
        
        if not update_data:
            logger.warning("No data to update for document")
            return None
        
        logger.info(f"📝 Updating document {requested_doc_id} with fields: {list(update_data.keys())}")
        
        try:
            response = await http_client_service.client.put(url, json=update_data)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Successfully updated document {requested_doc_id}")
            return result
            
        except httpx.HTTPStatusError as e:
            _log_api_error("update_document_status", url, update_data, e.response, e)
            return None
        except Exception as e:
            _log_api_error("update_document_status", url, update_data, exception=e)
            return None


# Global backend API service instance
backend_api_service = BackendAPIService()