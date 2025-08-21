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

    async def create_general_conversation(
        self,
        agent_type: str,
        conversation_purpose: str = "case_discovery"
    ) -> str:
        """Create a general conversation without a specific case_id for case discovery purposes"""
        conversation_data = {
            "agent_type": agent_type,
            "case_id": None,  # No specific case yet
            "conversation_purpose": conversation_purpose,
            "metadata": {
                "supports_case_discovery": True,
                "created_for": "natural_language_case_search"
            }
        }
        
        try:
            url = f"{self.backend_url}/api/agent/conversations"
            response = await http_client_service.client.post(url, json=conversation_data)
            response.raise_for_status()
            result = response.json()
            conversation_id = result.get("conversation_id")
            logger.info(f"🗣️ Created general conversation {conversation_id} for {agent_type} (purpose: {conversation_purpose})")
            return conversation_id
        except httpx.HTTPStatusError as e:
            _log_api_error("create_general_conversation", url, conversation_data, e.response, e)
            raise
        except Exception as e:
            _log_api_error("create_general_conversation", url, conversation_data, exception=e)
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
            messages = await self.get_conversation_history(conversation_id, limit=500)
            return len(messages)
        except Exception as e:
            logger.warning(f"Could not get message count for conversation {conversation_id}: {e}")
            return 0

    async def should_create_summary(self, conversation_id: str, threshold: int = 20) -> bool:
        """Check if conversation should be summarized based on message count"""
        message_count = await self.get_message_count(conversation_id)
        return message_count > threshold

    # Case Search and Discovery Methods
    async def search_cases(self, search_query: Dict[str, Any]) -> Dict[str, Any]:
        """Advanced case search with flexible filtering and fuzzy matching"""
        url = f"{self.backend_url}/api/cases/search"
        
        try:
            response = await http_client_service.client.post(url, json=search_query)
            response.raise_for_status()
            result = response.json()
            
            case_count = len(result.get("cases", []))
            total_count = result.get("total_count", 0)
            logger.info(f"🔍 Case search returned {case_count}/{total_count} cases")
            
            return result
        except httpx.HTTPStatusError as e:
            _log_api_error("search_cases", url, search_query, e.response, e)
            raise
        except Exception as e:
            _log_api_error("search_cases", url, search_query, exception=e)
            raise

    async def search_cases_by_name(self, name: str, use_fuzzy: bool = True, threshold: float = 0.3, limit: int = 10) -> Dict[str, Any]:
        """Search cases by client name with optional fuzzy matching"""
        search_query = {
            "client_name": name,
            "use_fuzzy_matching": use_fuzzy,
            "fuzzy_threshold": threshold,
            "limit": limit
        }
        
        logger.info(f"🔍 Searching cases by name: '{name}' (fuzzy: {use_fuzzy})")
        return await self.search_cases(search_query)

    async def search_cases_by_email(self, email: str, use_fuzzy: bool = False, limit: int = 10) -> Dict[str, Any]:
        """Search cases by client email address"""
        search_query = {
            "client_email": email,
            "use_fuzzy_matching": use_fuzzy,
            "limit": limit
        }
        
        logger.info(f"📧 Searching cases by email: '{email}'")
        return await self.search_cases(search_query)

    async def search_cases_by_phone(self, phone: str, limit: int = 10) -> Dict[str, Any]:
        """Search cases by client phone number"""
        search_query = {
            "client_phone": phone,
            "limit": limit
        }
        
        logger.info(f"📞 Searching cases by phone: '{phone}'")
        return await self.search_cases(search_query)

    async def search_cases_multi_field(self, name: Optional[str] = None, email: Optional[str] = None, 
                                     phone: Optional[str] = None, status: Optional[str] = None,
                                     use_fuzzy: bool = True, threshold: float = 0.3, limit: int = 20) -> Dict[str, Any]:
        """Multi-field case search with flexible matching"""
        search_query = {
            "use_fuzzy_matching": use_fuzzy,
            "fuzzy_threshold": threshold,
            "limit": limit
        }
        
        # Add non-null search fields
        if name:
            search_query["client_name"] = name
        if email:
            search_query["client_email"] = email
        if phone:
            search_query["client_phone"] = phone
        if status:
            search_query["status"] = status
            
        search_terms = [f"{k}={v}" for k, v in search_query.items() if k not in ["use_fuzzy_matching", "fuzzy_threshold", "limit"]]
        logger.info(f"🔍 Multi-field case search: {', '.join(search_terms)}")
        
        return await self.search_cases(search_query)

    async def list_all_cases(self, limit: int = 50, offset: int = 0, status: Optional[str] = None) -> Dict[str, Any]:
        """List all cases with pagination"""
        url = f"{self.backend_url}/api/cases"
        params = {
            "limit": limit,
            "offset": offset
        }
        if status:
            params["status"] = status
        
        try:
            response = await http_client_service.client.get(url, params=params)
            response.raise_for_status()
            result = response.json()
            
            case_count = len(result.get("cases", []))
            total_count = result.get("total_count", 0)
            logger.info(f"📋 Listed {case_count}/{total_count} cases (offset: {offset})")
            
            return result
        except httpx.HTTPStatusError as e:
            _log_api_error("list_all_cases", url, params, e.response, e)
            raise
        except Exception as e:
            _log_api_error("list_all_cases", url, params, exception=e)
            raise

    async def progressive_case_search(self, search_term: str, search_type: str = "name") -> Dict[str, Any]:
        """Progressive search: exact -> partial -> fuzzy matching"""
        logger.info(f"🎯 Starting progressive search for '{search_term}' (type: {search_type})")
        
        results = {"exact": None, "partial": None, "fuzzy": None, "best_match": None}
        
        try:
            # Step 1: Exact match (no fuzzy, higher precision)
            if search_type == "name":
                exact_results = await self.search_cases_by_name(search_term, use_fuzzy=False, limit=5)
            elif search_type == "email":
                exact_results = await self.search_cases_by_email(search_term, use_fuzzy=False, limit=5)
            elif search_type == "phone":
                exact_results = await self.search_cases_by_phone(search_term, limit=5)
            else:
                exact_results = {"cases": [], "total_count": 0}
                
            results["exact"] = exact_results
            
            # If exact match found, use it as best match
            if exact_results.get("total_count", 0) > 0:
                results["best_match"] = exact_results
                logger.info(f"✅ Exact match found: {exact_results['total_count']} cases")
                return results
            
            # Step 2: Fuzzy matching for names
            if search_type == "name":
                fuzzy_results = await self.search_cases_by_name(search_term, use_fuzzy=True, threshold=0.3, limit=10)
                results["fuzzy"] = fuzzy_results
                
                if fuzzy_results.get("total_count", 0) > 0:
                    results["best_match"] = fuzzy_results
                    logger.info(f"🎯 Fuzzy match found: {fuzzy_results['total_count']} cases")
                    return results
            
            # Step 3: Multi-field search as fallback
            multi_results = await self.search_cases_multi_field(
                name=search_term if search_type == "name" else None,
                email=search_term if search_type == "email" else None,
                phone=search_term if search_type == "phone" else None,
                use_fuzzy=True,
                threshold=0.2,  # Lower threshold for broader search
                limit=15
            )
            results["partial"] = multi_results
            results["best_match"] = multi_results
            
            logger.info(f"🔍 Progressive search completed. Best match: {multi_results.get('total_count', 0)} cases")
            return results
            
        except Exception as e:
            logger.error(f"❌ Progressive case search failed for '{search_term}': {e}")
            raise


# Global backend API service instance
backend_api_service = BackendAPIService()