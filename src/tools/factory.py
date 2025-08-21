"""
Factory for creating document analysis tools.
"""

import logging
from typing import List
from langchain.tools import BaseTool

from .planning import PlanAnalysisTasksTool
from .storage import StoreEvaluationResultsTool, RetrieveAgentContextTool
from .conversation_summary import ManageConversationSummaryTool, OptimizeConversationMemoryTool
from .context import GetCaseContextTool
from .case_analysis_retrieval import GetAllCaseAnalysesTool
from .document_requirements import GetRequestedDocumentsTool
from .document_satisfaction import DocumentSatisfactionTool
from .document_completion_manager import DocumentCompletionManagerTool
from .communications import SendAnalysisFindingsTool, SendCustomAnalysisMessageTool
from .case_search import case_search_tool, case_search_by_name_tool, case_search_by_email_tool, case_search_by_phone_tool

logger = logging.getLogger(__name__)


class DocumentAnalysisToolFactory:
    """Factory to create tools with proper dependencies"""
    
    def __init__(self):
        logger.info("Tool factory initialized")
    
    def create_all_tools(self) -> List[BaseTool]:
        """Create all tools for senior-partner-level reasoning and evaluation with persistent memory, conversation management, and intelligent case discovery"""
        return [
            # Case Discovery and Search Tools
            case_search_tool,  # Universal case search with auto-detection
            case_search_by_name_tool,  # Specialized name search with fuzzy matching
            case_search_by_email_tool,  # Email-based case search
            case_search_by_phone_tool,  # Phone number case search
            
            # Document Analysis and Review Tools
            GetAllCaseAnalysesTool(),  # Primary tool for comprehensive case review
            GetCaseContextTool(),  # Get case context and metadata
            GetRequestedDocumentsTool(),  # Get requested documents for a case
            
            # Document Evaluation and Completion Tools
            DocumentSatisfactionTool(),   # Evaluate document satisfaction and mark as completed
            DocumentCompletionManagerTool(),  # Comprehensive document completion workflow
            
            # Memory and Context Management Tools
            RetrieveAgentContextTool(),  # Retrieve persistent agent memory and context
            StoreEvaluationResultsTool(),  # Store senior partner evaluations in persistent context
            ManageConversationSummaryTool(),  # Manage conversation summaries for token optimization
            OptimizeConversationMemoryTool(),  # Comprehensive memory optimization
            
            # Planning and Communication Tools
            PlanAnalysisTasksTool(),  # Plan complex analysis tasks
            SendAnalysisFindingsTool(),  # Send analysis findings to Communications Agent
            SendCustomAnalysisMessageTool()  # Send custom analysis messages
        ]


# Global factory instance
tool_factory = DocumentAnalysisToolFactory()