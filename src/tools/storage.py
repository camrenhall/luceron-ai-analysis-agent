"""
Storage tools for persistent agent memory and evaluation results.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any
from langchain.tools import BaseTool

from services import backend_api_service

logger = logging.getLogger(__name__)


class StoreEvaluationResultsTool(BaseTool):
    """Tool to store senior partner evaluation results in persistent agent context"""
    name: str = "store_evaluation_results"
    description: str = """Store senior partner evaluation results in persistent context for future reference.
    Input: JSON with case_id, evaluation_type, findings, recommendations, risk_assessment, and other evaluation data.
    This stores results in the agent's long-term memory system for continuity across conversations."""
    
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
            
            logger.info(f"📝 Storing evaluation in persistent context for case {case_id}")
            
            # Store in persistent agent context with structured key
            context_key = f"evaluation_{evaluation_payload['evaluation_type']}_{datetime.now().strftime('%Y%m%d_%H%M')}"
            
            # Store the evaluation in persistent context
            await backend_api_service.store_context(
                case_id=case_id,
                agent_type="AnalysisAgent",
                context_key=context_key,
                context_value=evaluation_payload,
                expires_at=None  # Keep evaluation results indefinitely
            )
            
            # Also store a summary in a well-known key for quick access
            evaluation_summary = {
                "last_evaluation_date": evaluation_payload["evaluation_timestamp"],
                "evaluation_type": evaluation_payload["evaluation_type"],
                "key_findings_count": len(evaluation_payload.get("findings", {})),
                "risk_level": evaluation_payload.get("risk_assessment", {}).get("level", "unknown"),
                "confidence_score": evaluation_payload.get("confidence_score", 0.0),
                "documents_reviewed": evaluation_payload.get("documents_reviewed_count", 0),
                "has_recommendations": len(evaluation_payload.get("recommendations", [])) > 0,
                "has_red_flags": len(evaluation_payload.get("red_flags", [])) > 0
            }
            
            await backend_api_service.store_context(
                case_id=case_id,
                agent_type="AnalysisAgent",
                context_key="latest_evaluation_summary",
                context_value=evaluation_summary,
                expires_at=None
            )
            
            logger.info(f"✅ Senior partner evaluation stored in persistent context for case {case_id}: {evaluation_payload['evaluation_type']}")
            
            return json.dumps({
                "status": "evaluation_stored_in_context",
                "case_id": case_id,
                "evaluation_type": evaluation_payload["evaluation_type"],
                "timestamp": evaluation_payload["evaluation_timestamp"],
                "context_key": context_key,
                "persistent_storage": True,
                "agent_memory_updated": True
            })
            
        except Exception as e:
            error_msg = f"Failed to store evaluation results: {str(e)}"
            logger.error(f"📝 Storage ERROR: {error_msg}")
            return json.dumps({"error": error_msg})


class RetrieveAgentContextTool(BaseTool):
    """Tool to retrieve persistent agent context and memory"""
    name: str = "retrieve_agent_context"
    description: str = """Retrieve persistent agent context and memory for a case.
    Input: JSON with case_id and optionally context_key for specific context.
    Returns all stored context/memory for the agent on this case, including previous evaluations, findings, and insights."""
    
    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, **kwargs) -> str:
        """
        Retrieve persistent agent context for continuity across conversations.
        """
        try:
            logger.info("🧠 Retrieving agent context from persistent memory")
            
            # Handle LangChain's nested kwargs structure
            if 'kwargs' in kwargs:
                data = kwargs['kwargs']
            else:
                data = kwargs
            
            case_id = data.get("case_id")
            if not case_id:
                raise ValueError("case_id is required to retrieve context")
            
            context_key = data.get("context_key")  # Optional: retrieve specific context
            
            logger.info(f"🧠 Retrieving context for case {case_id}")
            
            # Retrieve all context for this agent/case
            all_context = await backend_api_service.get_case_agent_context(
                case_id=case_id,
                agent_type="AnalysisAgent"
            )
            
            if not all_context:
                logger.info(f"🧠 No persistent context found for case {case_id}")
                return json.dumps({
                    "status": "no_context_found",
                    "case_id": case_id,
                    "message": "No previous context or memory found for this case"
                })
            
            # If specific context key requested, return just that
            if context_key:
                if context_key in all_context:
                    specific_context = all_context[context_key]
                    logger.info(f"🧠 Retrieved specific context '{context_key}' for case {case_id}")
                    return json.dumps({
                        "status": "specific_context_retrieved",
                        "case_id": case_id,
                        "context_key": context_key,
                        "context": specific_context
                    })
                else:
                    return json.dumps({
                        "status": "context_key_not_found",
                        "case_id": case_id,
                        "context_key": context_key,
                        "available_keys": list(all_context.keys())
                    })
            
            # Return organized summary of all context
            context_summary = {
                "case_id": case_id,
                "total_context_items": len(all_context),
                "context_keys": list(all_context.keys()),
                "recent_evaluations": [],
                "key_insights": [],
                "other_context": {}
            }
            
            # Organize context by type
            for key, value in all_context.items():
                if "evaluation" in key.lower():
                    context_summary["recent_evaluations"].append({
                        "key": key,
                        "type": value.get("evaluation_type", "unknown"),
                        "timestamp": value.get("evaluation_timestamp", "unknown"),
                        "summary": value
                    })
                elif "insight" in key.lower() or "finding" in key.lower():
                    context_summary["key_insights"].append({
                        "key": key,
                        "summary": value.get("insight_summary", str(value)[:200] + "..." if len(str(value)) > 200 else str(value))
                    })
                else:
                    context_summary["other_context"][key] = value
            
            logger.info(f"🧠 Retrieved {len(all_context)} context items for case {case_id}")
            
            return json.dumps({
                "status": "context_retrieved",
                "case_id": case_id,
                "context_summary": context_summary,
                "full_context": all_context
            })
            
        except Exception as e:
            error_msg = f"Failed to retrieve agent context: {str(e)}"
            logger.error(f"🧠 Context retrieval ERROR: {error_msg}")
            return json.dumps({"error": error_msg})