"""
Backend API service for stateful agent management and data operations.
"""

import logging
from datetime import datetime
from typing import Dict, Optional, List, Any
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

    # =================================================================
    # STATEFUL AGENT MANAGEMENT API METHODS
    # =================================================================

    # Agent Conversations API
    async def create_conversation(
        self,
        case_id: str,
        agent_type: str,
        status: str = "ACTIVE"
    ) -> Dict:
        """Create a new agent conversation session"""
        url = f"{self.backend_url}/api/agent/conversations"
        request_data = {
            "case_id": case_id,
            "agent_type": agent_type,
            "status": status
        }
        
        try:
            response = await http_client_service.client.post(url, json=request_data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"🗣️ Created conversation {result.get('conversation_id')} for {agent_type} on case {case_id}")
            return result
        except httpx.HTTPStatusError as e:
            _log_api_error("create_conversation", url, request_data, e.response, e)
            raise
        except Exception as e:
            _log_api_error("create_conversation", url, request_data, exception=e)
            raise

    async def get_conversation_with_full_history(
        self,
        conversation_id: str,
        include_summaries: bool = True,
        include_function_calls: bool = True
    ) -> Dict:
        """Get conversation with complete message history and summaries"""
        url = f"{self.backend_url}/api/agent/conversations/{conversation_id}/full"
        params = {
            "include_summaries": include_summaries,
            "include_function_calls": include_function_calls
        }
        
        try:
            response = await http_client_service.client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            message_count = len(result.get("messages", []))
            summary_count = len(result.get("summaries", []))
            logger.info(f"🗣️ Retrieved conversation {conversation_id} with {message_count} messages and {summary_count} summaries")
            return result
        except httpx.HTTPStatusError as e:
            _log_api_error("get_conversation_with_full_history", url, params, e.response, e)
            raise
        except Exception as e:
            _log_api_error("get_conversation_with_full_history", url, params, exception=e)
            raise

    async def get_or_create_conversation(
        self,
        case_id: str,
        agent_type: str
    ) -> str:
        """Helper method to get existing active conversation or create new one"""
        # Try to get existing active conversation for this case+agent
        try:
            url = f"{self.backend_url}/api/agent/conversations"
            params = {
                "case_id": case_id,
                "agent_type": agent_type,
                "status": "ACTIVE",
                "limit": 1
            }
            
            response = await http_client_service.client.get(url, params=params)
            response.raise_for_status()
            conversations = response.json()
            
            if conversations and len(conversations) > 0:
                conversation_id = conversations[0]["conversation_id"]
                logger.info(f"🔄 Reusing existing conversation {conversation_id} for {agent_type} on case {case_id}")
                return conversation_id
                
        except Exception as e:
            logger.warning(f"Could not retrieve existing conversations: {e}, creating new one")
        
        # Create new conversation if none found
        conversation = await self.create_conversation(case_id, agent_type)
        return conversation["conversation_id"]

    # Agent Messages API
    async def add_message(
        self,
        conversation_id: str,
        role: str,
        content: Dict[str, Any],
        total_tokens: Optional[int] = None,
        model_used: Optional[str] = None,
        function_name: Optional[str] = None,
        function_arguments: Optional[Dict] = None,
        function_response: Optional[Dict] = None
    ) -> Dict:
        """Add a message to an agent conversation"""
        url = f"{self.backend_url}/api/agent/messages"
        request_data = {
            "conversation_id": conversation_id,
            "role": role,
            "content": content
        }
        
        if total_tokens is not None:
            request_data["total_tokens"] = total_tokens
        if model_used is not None:
            request_data["model_used"] = model_used
        if function_name is not None:
            request_data["function_name"] = function_name
        if function_arguments is not None:
            request_data["function_arguments"] = function_arguments
        if function_response is not None:
            request_data["function_response"] = function_response
        
        try:
            response = await http_client_service.client.post(url, json=request_data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"💬 Added {role} message to conversation {conversation_id}")
            return result
        except httpx.HTTPStatusError as e:
            _log_api_error("add_message", url, request_data, e.response, e)
            raise
        except Exception as e:
            _log_api_error("add_message", url, request_data, exception=e)
            raise

    async def get_conversation_history(
        self,
        conversation_id: str,
        limit: int = 50,
        include_function_calls: bool = True
    ) -> List[Dict]:
        """Get recent conversation history"""
        url = f"{self.backend_url}/api/agent/messages/conversation/{conversation_id}/history"
        params = {
            "limit": limit,
            "include_function_calls": include_function_calls
        }
        
        try:
            response = await http_client_service.client.get(url, params=params)
            response.raise_for_status()
            messages = response.json()
            logger.info(f"📜 Retrieved {len(messages)} messages from conversation {conversation_id}")
            return messages
        except httpx.HTTPStatusError as e:
            _log_api_error("get_conversation_history", url, params, e.response, e)
            raise
        except Exception as e:
            _log_api_error("get_conversation_history", url, params, exception=e)
            raise

    # Agent Context API
    async def store_context(
        self,
        case_id: str,
        agent_type: str,
        context_key: str,
        context_value: Dict[str, Any],
        expires_at: Optional[str] = None
    ) -> Dict:
        """Store persistent agent context"""
        url = f"{self.backend_url}/api/agent/context"
        request_data = {
            "case_id": case_id,
            "agent_type": agent_type,
            "context_key": context_key,
            "context_value": context_value
        }
        
        if expires_at is not None:
            request_data["expires_at"] = expires_at
        
        try:
            response = await http_client_service.client.post(url, json=request_data)
            response.raise_for_status()
            result = response.json()
            logger.info(f"🧠 Stored context '{context_key}' for {agent_type} on case {case_id}")
            return result
        except httpx.HTTPStatusError as e:
            _log_api_error("store_context", url, request_data, e.response, e)
            raise
        except Exception as e:
            _log_api_error("store_context", url, request_data, exception=e)
            raise

    async def get_case_agent_context(
        self,
        case_id: str,
        agent_type: str
    ) -> Dict[str, Any]:
        """Retrieve all context for a case and agent type"""
        url = f"{self.backend_url}/api/agent/context/case/{case_id}/agent/{agent_type}"
        
        try:
            response = await http_client_service.client.get(url)
            if response.status_code == 404:
                logger.info(f"🧠 No existing context found for {agent_type} on case {case_id}")
                return {}
            response.raise_for_status()
            
            context = response.json()
            context_keys = list(context.keys()) if context else []
            logger.info(f"🧠 Retrieved {len(context_keys)} context items for {agent_type} on case {case_id}: {context_keys}")
            return context
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return {}
            _log_api_error("get_case_agent_context", url, None, e.response, e)
            raise
        except Exception as e:
            _log_api_error("get_case_agent_context", url, exception=e)
            raise

    # Agent Summaries API
    async def create_auto_summary(
        self,
        conversation_id: str,
        messages_to_summarize: int = 15
    ) -> Dict:
        """Create automatic summary of conversation messages"""
        url = f"{self.backend_url}/api/agent/summaries/conversation/{conversation_id}/auto-summary"
        params = {"messages_to_summarize": messages_to_summarize}
        
        try:
            response = await http_client_service.client.post(url, params=params)
            response.raise_for_status()
            result = response.json()
            logger.info(f"📊 Created summary for conversation {conversation_id}, summarized {messages_to_summarize} messages")
            return result
        except httpx.HTTPStatusError as e:
            _log_api_error("create_auto_summary", url, params, e.response, e)
            raise
        except Exception as e:
            _log_api_error("create_auto_summary", url, params, exception=e)
            raise

    async def get_latest_summary(
        self,
        conversation_id: str
    ) -> Optional[Dict]:
        """Get the most recent conversation summary"""
        url = f"{self.backend_url}/api/agent/summaries/conversation/{conversation_id}/latest"
        
        try:
            response = await http_client_service.client.get(url)
            if response.status_code == 404:
                logger.info(f"📊 No summaries found for conversation {conversation_id}")
                return None
            response.raise_for_status()
            
            summary = response.json()
            logger.info(f"📊 Retrieved latest summary for conversation {conversation_id}")
            return summary
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 404:
                return None
            _log_api_error("get_latest_summary", url, None, e.response, e)
            raise
        except Exception as e:
            _log_api_error("get_latest_summary", url, exception=e)
            raise

    # Conversation Management Helpers
    async def get_message_count(self, conversation_id: str) -> int:
        """Get the total number of messages in a conversation"""
        try:
            messages = await self.get_conversation_history(conversation_id, limit=1000)
            return len(messages)
        except Exception as e:
            logger.warning(f"Could not get message count for conversation {conversation_id}: {e}")
            return 0

    async def should_create_summary(self, conversation_id: str, threshold: int = 20) -> bool:
        """Check if conversation should be summarized based on message count"""
        message_count = await self.get_message_count(conversation_id)
        return message_count > threshold


# Global backend API service instance
backend_api_service = BackendAPIService()