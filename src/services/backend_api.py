"""
Backend API service for workflow and data management.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, Union

from config import settings
from models import WorkflowStatus
from .http_client import http_client_service

logger = logging.getLogger(__name__)


class BackendAPIService:
    """Service for interacting with the backend API."""
    
    def __init__(self):
        self.backend_url = settings.BACKEND_URL
    
    async def load_workflow_state(self, workflow_id: str) -> Optional[Dict]:
        """Load workflow state from backend"""
        try:
            response = await http_client_service.client.get(
                f"{self.backend_url}/api/workflows/{workflow_id}"
            )
            if response.status_code == 404:
                return None
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to load workflow state: {e}")
            return None

    async def get_workflow_status(self, workflow_id: str) -> Optional[str]:
        """Get workflow status from backend"""
        workflow = await self.load_workflow_state(workflow_id)
        if workflow:
            return workflow.get("status")
        return None
    
    async def update_workflow_status(self, workflow_id: str, status: Union[str, WorkflowStatus]) -> None:
        """Update workflow status via backend"""
        try:
            # Handle both string and enum status
            if isinstance(status, WorkflowStatus):
                backend_status = status.value
            else:
                backend_status = status
            
            response = await http_client_service.client.put(
                f"{self.backend_url}/api/workflows/{workflow_id}/status",
                json={"status": backend_status}
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to update workflow status: {e}")
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
        try:
            step = {
                "timestamp": datetime.now().isoformat(),
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "action_output": action_output
            }
            response = await http_client_service.client.post(
                f"{self.backend_url}/api/workflows/{workflow_id}/reasoning-step",
                json=step
            )
            response.raise_for_status()
        except Exception as e:
            logger.error(f"Failed to add reasoning step: {e}")

    async def get_document_metadata(self, document_id: str) -> dict:
        """Get document metadata from backend"""
        response = await http_client_service.client.get(
            f"{self.backend_url}/api/documents/{document_id}"
        )
        if response.status_code == 404:
            raise ValueError(f"Document {document_id} not found")
        response.raise_for_status()
        return response.json()
    
    # NOTE: store_analysis_results method removed - AWS handles all document analysis storage directly

    async def get_case_context(self, case_id: str) -> dict:
        """Retrieve case details and context"""
        logger.info(f"🔍 Retrieving context for case {case_id}")
        
        response = await http_client_service.client.get(f"{self.backend_url}/api/cases/{case_id}")
        
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
        
        logger.info(f"🔍 Retrieved context for case {case_id}")
        return case_context


# Global backend API service instance
backend_api_service = BackendAPIService()