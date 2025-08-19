"""
Domain models for the document analysis and stateful agent management system.
"""

from typing import Dict, List, Optional, Any
from pydantic import BaseModel
from datetime import datetime


class AnalysisTask(BaseModel):
    """Model representing a single analysis task within a workflow."""
    task_id: int
    name: str
    document_ids: List[str]
    analysis_type: str
    status: str = "PENDING"  # PENDING, PROCESSING, COMPLETED, FAILED
    depends_on: List[int] = []
    results: Optional[Dict] = None


class TaskGraph(BaseModel):
    """Model representing a complete task execution graph."""
    tasks: List[AnalysisTask]
    execution_plan: str


# =================================================================
# STATEFUL AGENT MANAGEMENT MODELS
# =================================================================

class AgentConversation(BaseModel):
    """Model representing an agent conversation session."""
    conversation_id: str
    case_id: str
    agent_type: str  # "AnalysisAgent", "CommunicationsAgent", etc.
    status: str = "ACTIVE"  # ACTIVE, COMPLETED, ARCHIVED
    total_tokens_used: int = 0
    created_at: datetime
    updated_at: datetime


class AgentMessage(BaseModel):
    """Model representing a single message in an agent conversation."""
    message_id: str
    conversation_id: str
    role: str  # "system", "user", "assistant", "function"
    content: Dict[str, Any]  # Flexible content structure
    sequence_number: int
    total_tokens: Optional[int] = None
    model_used: Optional[str] = None
    function_name: Optional[str] = None
    function_arguments: Optional[Dict[str, Any]] = None
    function_response: Optional[Dict[str, Any]] = None
    created_at: datetime


class AgentContext(BaseModel):
    """Model representing persistent agent context/memory."""
    context_id: str
    case_id: str
    agent_type: str
    context_key: str
    context_value: Dict[str, Any]
    expires_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime


class AgentSummary(BaseModel):
    """Model representing a conversation summary."""
    summary_id: str
    conversation_id: str
    summary_content: str
    messages_summarized: int
    created_at: datetime


class ConversationWithHistory(BaseModel):
    """Model representing a conversation with its complete message history."""
    conversation: AgentConversation
    messages: List[AgentMessage] = []
    summaries: List[AgentSummary] = []


class AgentMemoryContext(BaseModel):
    """Model for organizing agent memory and context."""
    case_id: str
    agent_type: str
    conversation_id: Optional[str] = None
    persistent_context: Dict[str, Any] = {}  # Long-term memory
    session_context: Dict[str, Any] = {}     # Short-term working memory
    conversation_history: List[AgentMessage] = []
    summaries: List[AgentSummary] = []