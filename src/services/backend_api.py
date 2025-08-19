"""
Backend API service for workflow and data management.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Union
import json

from config import settings
from models import WorkflowStatus
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
    
    async def load_workflow_state(self, workflow_id: str) -> Optional[Dict]:
        """Load workflow state from backend"""
        url = f"{self.backend_url}/api/workflows/{workflow_id}"
        try:
            response = await http_client_service.client.get(url)
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            _log_api_error("load_workflow_state", url, response=e.response, exception=e)
            return None
        except Exception as e:
            _log_api_error("load_workflow_state", url, exception=e)
            return None

    async def get_workflow_status(self, workflow_id: str) -> Optional[str]:
        """Get workflow status from backend"""
        workflow = await self.load_workflow_state(workflow_id)
        if workflow:
            return workflow.get("status")
        return None
    
    async def update_workflow_status(self, workflow_id: str, status: Union[str, WorkflowStatus]) -> None:
        """Update workflow status via backend"""
        # Handle both string and enum status
        if isinstance(status, WorkflowStatus):
            backend_status = status.value
        else:
            backend_status = status
        
        url = f"{self.backend_url}/api/workflows/{workflow_id}/status"
        request_data = {"status": backend_status}
        
        try:
            response = await http_client_service.client.put(url, json=request_data)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            _log_api_error("update_workflow_status", url, request_data, e.response, e)
            raise
        except Exception as e:
            _log_api_error("update_workflow_status", url, request_data, exception=e)
            raise

    async def add_reasoning_step(
        self, 
        workflow_id: str, 
        thought: str, 
        action: str = None,
        action_input: Dict = None, 
        action_output: str = None
    ) -> None:
        """Add reasoning step to workflow"""
        step = {
            "timestamp": datetime.now().isoformat(),
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "action_output": action_output
        }
        url = f"{self.backend_url}/api/workflows/{workflow_id}/reasoning-step"
        
        try:
            response = await http_client_service.client.post(url, json=step)
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            _log_api_error("add_reasoning_step", url, step, e.response, e)
        except Exception as e:
            _log_api_error("add_reasoning_step", url, step, exception=e)

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
    
    async def update_workflow(
        self,
        workflow_id: str,
        status: Optional[Union[str, WorkflowStatus]] = None,
        reasoning_chain: Optional[list] = None,
        final_response: Optional[str] = None
    ) -> None:
        """Update workflow with optional status, reasoning chain, and final response"""
        update_data = {}
        
        if status is not None:
            # Handle both string and enum status
            if isinstance(status, WorkflowStatus):
                update_data["status"] = status.value
            else:
                update_data["status"] = status
        
        if reasoning_chain is not None:
            update_data["reasoning_chain"] = reasoning_chain
        
        if final_response is not None:
            update_data["final_response"] = final_response
        
        if not update_data:
            logger.warning("No data to update in workflow")
            return
        
        url = f"{self.backend_url}/api/workflows/{workflow_id}"
        logger.info(f"📝 Updating workflow {workflow_id} with fields: {list(update_data.keys())}")
        
        try:
            response = await http_client_service.client.put(url, json=update_data)
            response.raise_for_status()
            logger.info(f"✅ Successfully updated workflow {workflow_id}")
            
        except httpx.HTTPStatusError as e:
            _log_api_error("update_workflow", url, update_data, e.response, e)
            raise
        except Exception as e:
            _log_api_error("update_workflow", url, update_data, exception=e)
            raise


# Global backend API service instance
backend_api_service = BackendAPIService()