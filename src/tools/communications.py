"""
Communications tool for sending analysis findings to the Communications Agent.
"""

import json
import logging
from typing import Dict
from langchain.tools import BaseTool

from services.communications_agent import communications_agent_service

logger = logging.getLogger(__name__)


class SendAnalysisFindingsTool(BaseTool):
    """Tool to send analysis findings to the Communications Agent for client communication"""
    name: str = "send_analysis_findings"
    description: str = """Send analysis findings to the Communications Agent for client communication. 
    Input: JSON with case_id, finding_type, and analysis_details. 
    Finding types: document_type_mismatch, year_mismatch, duplicate_document, missing_documents, 
    document_quality_issues, document_satisfied.
    This tool sends structured analysis findings to the Communications Agent, which handles all client messaging."""
    
    def _run(self, input_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, input_data: str) -> str:
        try:
            # Parse input data
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON input"})
            
            case_id = data.get("case_id")
            finding_type = data.get("finding_type")
            analysis_details = data.get("analysis_details", {})
            
            if not case_id:
                return json.dumps({"error": "case_id is required"})
            
            if not finding_type:
                return json.dumps({"error": "finding_type is required"})
            
            logger.info(f"📤 Sending {finding_type} analysis finding for case {case_id}")
            
            # Check if Communications Agent is available
            if not communications_agent_service.is_available():
                logger.warning("Communications Agent not configured - analysis finding not sent")
                return json.dumps({
                    "success": False,
                    "error": "Communications Agent not configured",
                    "case_id": case_id,
                    "finding_type": finding_type
                })
            
            # Send analysis finding
            result = await communications_agent_service.send_document_analysis_finding(
                case_id=case_id,
                finding_type=finding_type,
                analysis_details=analysis_details
            )
            
            logger.info(f"📊 Analysis finding transmission result: {result.get('success', False)}")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to send analysis findings: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})


class SendCustomAnalysisMessageTool(BaseTool):
    """Tool to send custom analysis messages to the Communications Agent"""
    name: str = "send_custom_analysis_message"
    description: str = """Send a custom analysis message to the Communications Agent. 
    Input: JSON with case_id and analysis_message (free-form text describing analysis findings).
    Use this for complex findings that don't fit standard finding types."""
    
    def _run(self, input_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, input_data: str) -> str:
        try:
            # Parse input data
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON input"})
            
            case_id = data.get("case_id")
            analysis_message = data.get("analysis_message")
            
            if not case_id:
                return json.dumps({"error": "case_id is required"})
            
            if not analysis_message:
                return json.dumps({"error": "analysis_message is required"})
            
            logger.info(f"📤 Sending custom analysis message for case {case_id}")
            
            # Check if Communications Agent is available
            if not communications_agent_service.is_available():
                logger.warning("Communications Agent not configured - analysis message not sent")
                return json.dumps({
                    "success": False,
                    "error": "Communications Agent not configured",
                    "case_id": case_id
                })
            
            # Send custom analysis message
            result = await communications_agent_service.send_analysis_finding(
                analysis_message=analysis_message,
                case_id=case_id
            )
            
            logger.info(f"📊 Custom analysis message transmission result: {result.get('success', False)}")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to send custom analysis message: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})