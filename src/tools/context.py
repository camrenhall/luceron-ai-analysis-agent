"""
Context retrieval tools for document analysis.
"""

import json
import logging
from langchain.tools import BaseTool

from services import backend_api_service

logger = logging.getLogger(__name__)


class GetCaseContextTool(BaseTool):
    """Tool to retrieve case context for informed analysis"""
    name: str = "get_case_context"
    description: str = "Retrieve case details and context. Input: case_id"
    
    def _run(self, case_id: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, case_id: str) -> str:
        try:
            case_context = await backend_api_service.get_case_context(case_id)
            return json.dumps(case_context, indent=2)
            
        except Exception as e:
            error_msg = f"Context retrieval failed: {str(e)}"
            logger.error(f"🔍 Context ERROR: {error_msg}")
            return json.dumps({"error": error_msg})