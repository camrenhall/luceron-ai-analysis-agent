"""
Factory for creating document analysis tools.
"""

import logging
from typing import List
from langchain.tools import BaseTool

from .planning import PlanAnalysisTasksTool
from .storage import StoreEvaluationResultsTool
from .context import GetCaseContextTool
from .case_analysis_retrieval import GetAllCaseAnalysesTool

logger = logging.getLogger(__name__)


class DocumentAnalysisToolFactory:
    """Factory to create tools with proper dependencies"""
    
    def __init__(self):
        logger.info("Tool factory initialized")
    
    def create_all_tools(self) -> List[BaseTool]:
        """Create all tools for senior-partner-level reasoning and evaluation"""
        return [
            GetAllCaseAnalysesTool(),  # Primary tool for comprehensive case review
            StoreEvaluationResultsTool(),  # Store senior partner evaluations (NOT document analysis)
            PlanAnalysisTasksTool(),
            GetCaseContextTool()
        ]


# Global factory instance
tool_factory = DocumentAnalysisToolFactory()