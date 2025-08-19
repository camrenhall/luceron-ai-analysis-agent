"""
Communications Agent service for sending client communications.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Optional, AsyncGenerator
import httpx

from config import settings
from .http_client import http_client_service

logger = logging.getLogger(__name__)


class CommunicationsAgentService:
    """Service for interacting with the Communications Agent."""
    
    def __init__(self):
        self.communications_url = settings.COMMUNICATIONS_AGENT_URL
        
    def is_available(self) -> bool:
        """Check if Communications Agent is configured and available"""
        return bool(self.communications_url)
    
    async def send_analysis_finding(self, analysis_message: str, case_id: Optional[str] = None) -> Dict:
        """
        Send analysis findings to the Communications Agent for client communication.
        
        Args:
            analysis_message: Structured analysis findings for the Communications Agent to process
            case_id: Optional case ID for logging/tracking purposes
            
        Returns:
            Dict with communication results including success status and events
        """
        if not self.is_available():
            logger.warning("Communications Agent URL not configured - message not sent")
            return {
                "success": False,
                "error": "Communications Agent not configured",
                "analysis_message": analysis_message,
                "case_id": case_id
            }
        
        url = f"{self.communications_url}/chat"
        request_data = {"message": analysis_message}
        
        communication_log = {
            "timestamp": datetime.now().isoformat(),
            "case_id": case_id,
            "analysis_message": analysis_message,
            "url": url
        }
        
        logger.info(f"📊 Sending analysis findings to Communications Agent for case {case_id}")
        logger.debug(f"Analysis findings: {analysis_message[:100]}...")
        
        try:
            # Send POST request and handle SSE response
            events = []
            
            async with http_client_service.client.stream(
                "POST", 
                url, 
                json=request_data,
                headers={"Accept": "text/event-stream"},
                timeout=30.0
            ) as response:
                
                if response.status_code != 200:
                    error_msg = f"Communications Agent returned status {response.status_code}"
                    logger.error(f"❌ Analysis findings transmission failed: {error_msg}")
                    
                    communication_log.update({
                        "success": False,
                        "error": error_msg,
                        "status_code": response.status_code
                    })
                    
                    return communication_log
                
                # Parse SSE events from the stream
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line or line.startswith(':'):
                        continue
                    
                    if line.startswith('data: '):
                        event_data = line[6:]  # Remove 'data: ' prefix
                        try:
                            event = json.loads(event_data)
                            events.append(event)
                            logger.debug(f"Received SSE event: {event}")
                            
                            # Log important events
                            if event.get("type") == "started":
                                logger.info("📤 Communications Agent workflow started")
                            elif event.get("type") == "completed":
                                logger.info("✅ Communications Agent workflow completed")
                            elif event.get("type") == "error":
                                logger.error(f"❌ Communications Agent workflow error: {event.get('message')}")
                                
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse SSE event: {event_data}")
            
            # Determine overall success based on events
            has_completed = any(event.get("type") == "completed" for event in events)
            has_error = any(event.get("type") == "error" for event in events)
            
            success = has_completed and not has_error
            
            communication_log.update({
                "success": success,
                "events": events,
                "event_count": len(events)
            })
            
            if success:
                logger.info(f"✅ Successfully sent analysis findings to Communications Agent for case {case_id}")
            else:
                logger.warning(f"⚠️ Analysis findings transmission may not have completed successfully for case {case_id}")
            
            return communication_log
            
        except httpx.TimeoutException:
            error_msg = "Analysis findings transmission timed out"
            logger.error(f"❌ {error_msg}")
            communication_log.update({
                "success": False,
                "error": error_msg
            })
            return communication_log
            
        except httpx.RequestError as e:
            error_msg = f"Analysis findings transmission failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            communication_log.update({
                "success": False,
                "error": error_msg
            })
            return communication_log
            
        except Exception as e:
            error_msg = f"Unexpected error during analysis findings transmission: {str(e)}"
            logger.error(f"❌ {error_msg}")
            communication_log.update({
                "success": False,
                "error": error_msg
            })
            return communication_log
    
    async def send_document_analysis_finding(
        self, 
        case_id: str,
        finding_type: str,
        analysis_details: Dict
    ) -> Dict:
        """
        Send structured document analysis findings to the Communications Agent.
        
        Args:
            case_id: The case ID
            finding_type: Type of finding (e.g., 'document_type_mismatch', 'year_mismatch', 'missing_documents')
            analysis_details: Structured details about the analysis finding
            
        Returns:
            Dict with communication results
        """
        
        # Format analysis finding message
        analysis_message = self._format_analysis_finding(finding_type, case_id, analysis_details)
        
        if not analysis_message:
            logger.error(f"❌ Could not generate analysis finding for type: {finding_type}")
            return {
                "success": False,
                "error": f"Unknown finding type: {finding_type}",
                "case_id": case_id
            }
        
        # Log the analysis finding being sent
        logger.info(f"📊 Sending {finding_type} analysis finding for case {case_id}")
        logger.debug(f"Analysis details: {analysis_details}")
        
        # Send the analysis finding
        result = await self.send_analysis_finding(analysis_message, case_id)
        
        # Add finding context to result
        result.update({
            "finding_type": finding_type,
            "analysis_details": analysis_details
        })
        
        return result
    
    def _format_analysis_finding(self, finding_type: str, case_id: str, details: Dict) -> Optional[str]:
        """
        Format an analysis finding message using structured templates.
        
        Args:
            finding_type: The type of analysis finding
            case_id: The case ID
            details: Analysis details dictionary
            
        Returns:
            Formatted analysis finding message or None if template not found
        """
        
        # Base analysis finding format
        base_format = f"ANALYSIS FINDING: {finding_type.replace('_', ' ').title()}\nCase ID: {case_id}\n"
        
        # Specific finding formats
        if finding_type == "document_type_mismatch":
            return base_format + (
                f"Expected Document: {details.get('expected_type', 'Unknown')}\n"
                f"Received Document: {details.get('received_type', 'Unknown')}\n"
                f"Document ID: {details.get('document_id', 'Unknown')}\n"
                f"Confidence: {details.get('confidence', 'Unknown')}\n"
                f"Analysis Details: {details.get('analysis_details', 'No details provided')}\n"
                f"Recommendation: Request correct document type from client."
            )
        
        elif finding_type == "year_mismatch":
            return base_format + (
                f"Expected Year: {details.get('expected_year', 'Unknown')}\n"
                f"Received Year: {details.get('received_year', 'Unknown')}\n"
                f"Document Type: {details.get('document_type', 'Unknown')}\n"
                f"Document ID: {details.get('document_id', 'Unknown')}\n"
                f"Analysis Details: {details.get('analysis_details', 'No details provided')}\n"
                f"Recommendation: Request document for correct year from client."
            )
        
        elif finding_type == "duplicate_document":
            return base_format + (
                f"Document Type: {details.get('document_type', 'Unknown')}\n"
                f"Original Document ID: {details.get('original_document_id', 'Unknown')}\n"
                f"Duplicate Document ID: {details.get('duplicate_document_id', 'Unknown')}\n"
                f"Similarity Score: {details.get('similarity_score', 'Unknown')}\n"
                f"Still Needed: {details.get('remaining_documents', 'Unknown')}\n"
                f"Recommendation: Acknowledge receipt and request remaining documents."
            )
        
        elif finding_type == "missing_documents":
            return base_format + (
                f"Completion Status: {details.get('completed_count', 0)}/{details.get('total_required', 0)} documents satisfied\n"
                f"Missing Documents: {details.get('missing_documents', 'Unknown')}\n"
                f"Priority: {details.get('priority', 'Standard')}\n"
                f"Analysis Details: {details.get('analysis_details', 'No details provided')}\n"
                f"Recommendation: Request missing documents from client."
            )
        
        elif finding_type == "document_quality_issues":
            return base_format + (
                f"Document Type: {details.get('document_type', 'Unknown')}\n"
                f"Document ID: {details.get('document_id', 'Unknown')}\n"
                f"Quality Issues: {details.get('quality_issues', 'Unknown')}\n"
                f"Impact Assessment: {details.get('impact_assessment', 'Unknown')}\n"
                f"Recommendation: Request higher quality version of document."
            )
        
        elif finding_type == "document_satisfied":
            return base_format + (
                f"Document Type: {details.get('document_type', 'Unknown')}\n"
                f"Document ID: {details.get('document_id', 'Unknown')}\n"
                f"Satisfaction Details: {details.get('satisfaction_details', 'No details provided')}\n"
                f"Remaining Requirements: {details.get('remaining_count', 0)} documents still needed\n"
                f"Recommendation: {details.get('recommendation', 'Continue with remaining requirements.')}"
            )
        
        else:
            logger.error(f"Unknown analysis finding type: {finding_type}")
            return None


# Global communications service instance
communications_agent_service = CommunicationsAgentService()