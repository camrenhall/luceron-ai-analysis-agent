"""
Storage tools for senior partner evaluation results.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any
from langchain.tools import BaseTool

from services import backend_api_service

logger = logging.getLogger(__name__)


class StoreEvaluationResultsTool(BaseTool):
    """Tool to store senior partner evaluation and reasoning results"""
    name: str = "store_evaluation_results"
    description: str = """Store senior partner evaluation results after reviewing case analyses. 
    Input: JSON with case_id, evaluation_type, findings, recommendations, and risk_assessment"""
    
    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, **kwargs) -> str:
        """
        Store senior partner evaluation results.
        
        This is NOT for storing document analysis (AWS does that).
        This is for storing the senior partner's evaluation, reasoning, and recommendations
        after reviewing all the case documents.
        """
        try:
            logger.info(f"📝 Storing senior partner evaluation")
            
            # Handle LangChain's nested kwargs structure
            if 'kwargs' in kwargs:
                data = kwargs['kwargs']
                logger.info("📝 Found data in nested kwargs structure")
            else:
                data = kwargs
                logger.info("📝 Using direct kwargs structure")
            
            logger.info(f"📝 Data keys: {list(data.keys())}")
            
            case_id = data.get("case_id")
            if not case_id:
                raise ValueError("case_id is required for evaluation storage")
            
            # Structure the evaluation payload
            evaluation_payload = {
                "case_id": case_id,
                "workflow_id": data.get("workflow_id"),
                "evaluation_type": data.get("evaluation_type", "comprehensive_review"),
                "evaluation_timestamp": datetime.now().isoformat(),
                "findings": data.get("findings", {}),
                "patterns_identified": data.get("patterns_identified", []),
                "red_flags": data.get("red_flags", []),
                "recommendations": data.get("recommendations", []),
                "risk_assessment": data.get("risk_assessment", {
                    "level": "medium",
                    "factors": []
                }),
                "confidence_score": data.get("confidence_score", 0.0),
                "documents_reviewed_count": data.get("documents_reviewed_count", 0),
                "evaluator_notes": data.get("evaluator_notes", ""),
                "next_steps": data.get("next_steps", [])
            }
            
            logger.info(f"📝 Storing evaluation for case {case_id}")
            
            # Store as a reasoning step in the workflow
            if evaluation_payload.get("workflow_id"):
                await backend_api_service.add_reasoning_step(
                    workflow_id=evaluation_payload["workflow_id"],
                    thought="Senior partner evaluation completed",
                    action="store_evaluation",
                    action_input=evaluation_payload,
                    action_output=json.dumps({
                        "evaluation_stored": True,
                        "case_id": case_id,
                        "timestamp": evaluation_payload["evaluation_timestamp"]
                    })
                )
            
            logger.info(f"📝 Senior partner evaluation stored successfully for case {case_id}")
            
            return json.dumps({
                "status": "evaluation_stored",
                "case_id": case_id,
                "evaluation_type": evaluation_payload["evaluation_type"],
                "timestamp": evaluation_payload["evaluation_timestamp"]
            })
            
        except Exception as e:
            error_msg = f"Failed to store evaluation results: {str(e)}"
            logger.error(f"📝 Storage ERROR: {error_msg}")
            return json.dumps({"error": error_msg})