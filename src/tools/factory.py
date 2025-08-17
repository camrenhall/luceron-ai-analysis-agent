"""
Factory for creating document analysis tools.
"""

import logging
from typing import List
from langchain.tools import BaseTool

from .planning import PlanAnalysisTasksTool
from .storage import StoreAnalysisResultsTool
from .context import GetCaseContextTool

logger = logging.getLogger(__name__)


class DocumentAnalysisToolFactory:
    """Factory to create tools with proper dependencies"""
    
    def __init__(self):
        logger.info("Tool factory initialized")
    
    def create_all_tools(self) -> List[BaseTool]:
        """Create all document analysis tools (analysis now handled by Step Functions)"""
        return [
            PlanAnalysisTasksTool(),
            StoreAnalysisResultsTool(),
            GetCaseContextTool()
        ]


# Global factory instance
tool_factory = DocumentAnalysisToolFactory()