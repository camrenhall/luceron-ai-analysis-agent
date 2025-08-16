"""
Storage tools for document analysis results.
"""

import json
import logging
from langchain.tools import BaseTool

from ..services import backend_api_service

logger = logging.getLogger(__name__)


class StoreAnalysisResultsTool(BaseTool):
    """Tool to store analysis results back to backend database"""
    name: str = "store_analysis_results"
    description: str = "Store document analysis results in database. Input: JSON with document_id, case_id, analysis_content"
    
    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, **kwargs) -> str:
        try:
            logger.info(f"💾 DEBUG: _arun called with kwargs keys: {list(kwargs.keys())}")
            
            # FIX: Handle the nested kwargs structure that LangChain is using
            if 'kwargs' in kwargs:
                # LangChain is passing data as {'kwargs': {actual_data}}
                data = kwargs['kwargs']
                logger.info("💾 Found data in nested kwargs structure")
            else:
                # Fallback to direct kwargs
                data = kwargs
                logger.info("💾 Using direct kwargs structure")
            
            logger.info(f"💾 Data keys: {list(data.keys())}")
            
            document_id = data.get("document_id")
            if not document_id:
                raise ValueError("document_id is required")
            
            logger.info(f"💾 Storing analysis results for document {document_id}")
            
            # Ensure analysis_content is a string (the agent is passing it as a dict)
            if isinstance(data.get("analysis_content"), dict):
                # Convert the dict to a JSON string
                data["analysis_content"] = json.dumps(data["analysis_content"])
                logger.info("💾 Converted analysis_content dict to JSON string")
            
            # Ensure we have all required fields for the backend API
            analysis_payload = {
                "document_id": data.get("document_id"),
                "case_id": data.get("case_id"),
                "workflow_id": data.get("workflow_id"),
                "analysis_content": data.get("analysis_content", ""),
                "model_used": data.get("model_used", "o3"),
                "tokens_used": data.get("tokens_used"),
                "analysis_status": data.get("analysis_status", "completed")
            }
            
            logger.info(f"💾 Sending payload with keys: {list(analysis_payload.keys())}")
            logger.info(f"💾 Storing analysis with {analysis_payload.get('tokens_used')} tokens used")
            
            # Call backend API service to store results
            result = await backend_api_service.store_analysis_results(analysis_payload)
            analysis_id = result.get("analysis_id")
            
            logger.info(f"💾 Analysis results stored successfully: {analysis_id}")
            
            return json.dumps({
                "status": "stored",
                "analysis_id": analysis_id,
                "document_id": document_id
            })
            
        except Exception as e:
            error_msg = f"Failed to store analysis results: {str(e)}"
            logger.error(f"💾 Storage ERROR: {error_msg}")
            return json.dumps({"error": error_msg})