"""
Factory for creating document analysis tools.
"""

import logging
from typing import List
from langchain.tools import BaseTool

from .planning import PlanAnalysisTasksTool
from .analysis import OpenAIDocumentAnalysisTool
from .storage import StoreAnalysisResultsTool
from .context import GetCaseContextTool

logger = logging.getLogger(__name__)


class DocumentAnalysisToolFactory:
    """Factory to create tools with proper dependencies"""
    
    def __init__(self):
        logger.info("Tool factory initialized")
    
    def create_analysis_tool(self) -> OpenAIDocumentAnalysisTool:
        """Create OpenAI analysis tool"""
        return OpenAIDocumentAnalysisTool()
    
    def create_all_tools(self) -> List[BaseTool]:
        """Create all document analysis tools"""
        return [
            PlanAnalysisTasksTool(),
            self.create_analysis_tool(),
            StoreAnalysisResultsTool(),
            GetCaseContextTool()
        ]


# Global factory instance
tool_factory = DocumentAnalysisToolFactory()