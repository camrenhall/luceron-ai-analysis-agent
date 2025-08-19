"""
Comprehensive document completion management tool that integrates all document satisfaction functionality.
"""

import json
import logging
from typing import Dict, List, Optional
from datetime import datetime
from langchain.tools import BaseTool

from services import backend_api_service
from .document_satisfaction import DocumentSatisfactionTool
from .document_requirements import GetRequestedDocumentsTool
from .case_analysis_retrieval import GetAllCaseAnalysesTool

logger = logging.getLogger(__name__)


class DocumentCompletionManagerTool(BaseTool):
    """
    Comprehensive tool for managing document completion workflow.
    This tool orchestrates the entire process of checking document requirements,
    retrieving analyzed documents, evaluating satisfaction, and updating completion status.
    """
    name: str = "manage_document_completion"
    description: str = """Complete workflow tool for document completion management. 
    Input: case_id or JSON with case_id and optional send_communications (boolean). This tool will:
    1. Get all requested documents for the case
    2. Get all analyzed documents for the case  
    3. Evaluate which requested documents are satisfied by analyzed documents
    4. Mark satisfied documents as completed
    5. Optionally send communications to Communications Agent for document issues
    6. Provide comprehensive report of completion status
    """
    
    def __init__(self):
        super().__init__()
        self.satisfaction_tool = DocumentSatisfactionTool()
        self.requirements_tool = GetRequestedDocumentsTool()
        self.analysis_tool = GetAllCaseAnalysesTool()
    
    def _run(self, input_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, input_data: str) -> str:
        try:
            # Parse input - can be just case_id string or JSON with options
            try:
                data = json.loads(input_data)
                case_id = data.get("case_id")
                send_communications = data.get("send_communications", False)
            except json.JSONDecodeError:
                # Assume it's just a case_id string
                case_id = input_data
                send_communications = False
            
            if not case_id:
                return json.dumps({"error": "case_id is required"})
            
            logger.info(f"🚀 Starting comprehensive document completion workflow for case {case_id}")
            if send_communications:
                logger.info("📤 Communications will be sent for document issues")
            
            # Step 1: Get requested documents
            logger.info("📋 Step 1: Retrieving requested documents...")
            requirements_result = await self.requirements_tool._arun(case_id)
            requirements_data = json.loads(requirements_result)
            
            if "error" in requirements_data:
                return json.dumps({
                    "error": f"Failed to get requested documents: {requirements_data['error']}"
                })
            
            # Step 2: Get all analyzed documents
            logger.info("📊 Step 2: Retrieving analyzed documents...")
            analysis_result = await self.analysis_tool._arun(case_id)
            analysis_data = json.loads(analysis_result)
            
            if "error" in analysis_data:
                return json.dumps({
                    "error": f"Failed to get analyzed documents: {analysis_data['error']}"
                })
            
            # Step 3: Prepare document analysis results for satisfaction evaluation
            document_analyses = analysis_data.get("document_analyses", [])
            
            # Step 4: Evaluate document satisfaction
            logger.info("🔍 Step 3: Evaluating document satisfaction...")
            satisfaction_input = {
                "case_id": case_id,
                "document_analysis_results": document_analyses,
                "send_communications": send_communications
            }
            
            satisfaction_result = await self.satisfaction_tool._arun(json.dumps(satisfaction_input))
            satisfaction_data = json.loads(satisfaction_result)
            
            if "error" in satisfaction_data:
                return json.dumps({
                    "error": f"Failed to evaluate satisfaction: {satisfaction_data['error']}"
                })
            
            # Step 5: Generate comprehensive completion report
            logger.info("📈 Step 4: Generating completion report...")
            
            completion_report = self._generate_completion_report(
                case_id, requirements_data, analysis_data, satisfaction_data
            )
            
            logger.info(f"✅ Document completion workflow completed for case {case_id}")
            return json.dumps(completion_report, indent=2)
            
        except Exception as e:
            error_msg = f"Document completion workflow failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})
    
    def _generate_completion_report(
        self, 
        case_id: str, 
        requirements_data: Dict, 
        analysis_data: Dict, 
        satisfaction_data: Dict
    ) -> Dict:
        """Generate a comprehensive completion report"""
        
        # Extract key metrics
        total_requested = requirements_data.get("total_requested_documents", 0)
        total_analyzed = analysis_data.get("summary", {}).get("total_documents_analyzed", 0)
        newly_completed = satisfaction_data.get("summary", {}).get("newly_completed", 0)
        still_pending = satisfaction_data.get("summary", {}).get("still_pending", 0)
        
        # Calculate completion percentages
        completion_percentage = 0
        if total_requested > 0:
            completed_count = total_requested - still_pending
            completion_percentage = (completed_count / total_requested) * 100
        
        # Identify gaps and next steps
        next_steps = self._identify_next_steps(requirements_data, satisfaction_data)
        
        # Extract document details
        completed_documents = []
        pending_documents = []
        
        for result in satisfaction_data.get("satisfaction_results", []):
            if result.get("is_satisfied"):
                completed_documents.append({
                    "document_name": result.get("requested_document_name"),
                    "satisfied_by": len(result.get("matched_documents", [])),
                    "reason": result.get("reason")
                })
            else:
                pending_documents.append({
                    "document_name": result.get("requested_document_name"),
                    "description": result.get("requested_description"),
                    "reason_not_satisfied": result.get("reason")
                })
        
        return {
            "case_id": case_id,
            "workflow_completed_at": datetime.now().isoformat(),
            "summary": {
                "total_requested_documents": total_requested,
                "total_analyzed_documents": total_analyzed,
                "completion_percentage": round(completion_percentage, 1),
                "newly_completed_in_this_run": newly_completed,
                "still_pending": still_pending
            },
            "completion_details": {
                "completed_documents": completed_documents,
                "pending_documents": pending_documents
            },
            "workflow_steps": {
                "requirements_retrieved": "success",
                "analyses_retrieved": "success",
                "satisfaction_evaluated": "success",
                "documents_updated": satisfaction_data.get("summary", {}).get("newly_completed", 0) > 0
            },
            "next_steps": next_steps,
            "raw_data": {
                "requirements": requirements_data,
                "satisfaction_evaluation": satisfaction_data
            }
        }
    
    def _identify_next_steps(self, requirements_data: Dict, satisfaction_data: Dict) -> List[str]:
        """Identify recommended next steps based on completion status"""
        
        next_steps = []
        
        # Check completion status
        still_pending = satisfaction_data.get("summary", {}).get("still_pending", 0)
        
        if still_pending > 0:
            next_steps.append(f"Request {still_pending} additional documents from client")
            
            # Identify specific document types needed
            pending_docs = []
            for result in satisfaction_data.get("satisfaction_results", []):
                if not result.get("is_satisfied"):
                    doc_name = result.get("requested_document_name")
                    if doc_name:
                        pending_docs.append(doc_name)
            
            if pending_docs:
                next_steps.append(f"Specifically needed: {', '.join(pending_docs)}")
        
        # Check for flagged documents
        flagged_count = requirements_data.get("completion_summary", {}).get("flagged_for_review", 0)
        if flagged_count > 0:
            next_steps.append(f"Review {flagged_count} documents flagged for manual review")
        
        # Check analysis quality
        total_analyzed = len(satisfaction_data.get("satisfaction_results", []))
        satisfied_count = len([r for r in satisfaction_data.get("satisfaction_results", []) if r.get("is_satisfied")])
        
        if total_analyzed > 0 and satisfied_count == 0:
            next_steps.append("Consider reviewing document analysis quality - no documents were automatically satisfied")
        
        if not next_steps:
            next_steps.append("All requested documents appear to be satisfied. Case may be ready for final review.")
        
        return next_steps