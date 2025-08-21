"""
Factory for creating case discovery tools.
"""

import logging
from typing import List
from langchain.tools import BaseTool

from .case_search_consolidated import consolidated_case_search_tool

logger = logging.getLogger(__name__)


class CaseDiscoveryToolFactory:
    """Factory to create tools with proper dependencies"""
    
    def __init__(self):
        logger.info("Tool factory initialized")
    
    def create_all_tools(self) -> List[BaseTool]:
        """Create minimal toolset focused solely on intelligent case discovery"""
        return [
            # Case Discovery and Search - The Only Tool
            consolidated_case_search_tool  # Universal case search with intelligent auto-detection and dynamic endpoint selection
        ]


# Global factory instance
tool_factory = CaseDiscoveryToolFactory()