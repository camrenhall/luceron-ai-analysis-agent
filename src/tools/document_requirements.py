"""
Tool for retrieving and managing document requirements for cases.
"""

import json
import logging
from langchain.tools import BaseTool

from services import backend_api_service

logger = logging.getLogger(__name__)


class GetRequestedDocumentsTool(BaseTool):
    """Tool to retrieve requested documents for a case"""
    name: str = "get_requested_documents"
    description: str = """Retrieve all requested documents for a case. This shows what documents are needed, 
    their current completion status, and any notes. Input: case_id. Output: JSON with requested documents list."""
    
    def _run(self, case_id: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, case_id: str) -> str:
        try:
            logger.info(f"📋 Retrieving requested documents for case {case_id}")
            
            case_data = await backend_api_service.get_requested_documents(case_id)
            
            if not case_data:
                return json.dumps({
                    "error": f"Case {case_id} not found or no data available"
                })
            
            requested_documents = case_data.get("requested_documents", [])
            
            # Structure the response for the agent
            result = {
                "case_id": case_id,
                "total_requested_documents": len(requested_documents),
                "requested_documents": requested_documents,
                "case_details": {
                    "case_type": case_data.get("case_type"),
                    "priority": case_data.get("priority"),
                    "created_at": case_data.get("created_at"),
                    "updated_at": case_data.get("updated_at")
                },
                "completion_summary": {
                    "completed": len([doc for doc in requested_documents if doc.get("is_completed", False)]),
                    "pending": len([doc for doc in requested_documents if not doc.get("is_completed", False)]),
                    "flagged_for_review": len([doc for doc in requested_documents if doc.get("is_flagged_for_review", False)])
                }
            }
            
            logger.info(f"✅ Retrieved {len(requested_documents)} requested documents for case {case_id}")
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to retrieve requested documents: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})