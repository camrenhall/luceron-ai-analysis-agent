"""
Workflow management API routes.
"""

import uuid
from fastapi import APIRouter, HTTPException, BackgroundTasks

from models import TriggerDocumentAnalysisRequest, DocumentAnalysisResponse, DocumentAnalysisStatus
from services import http_client_service, backend_api_service
from core import execute_analysis_workflow
from config import settings

router = APIRouter()


@router.post("/trigger-analysis", response_model=DocumentAnalysisResponse)
async def trigger_document_analysis(request: TriggerDocumentAnalysisRequest, background_tasks: BackgroundTasks):
    """Trigger document analysis workflow"""
    
    # Use workflow_id from request if provided, otherwise generate one
    workflow_id = request.workflow_id or f"wf_analysis_{uuid.uuid4().hex[:12]}"
    
    # Create workflow state in backend - FIX: Use backend's WorkflowStatus enum values
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": "PENDING",  # Use backend's WorkflowStatus enum value instead of DocumentAnalysisStatus
        "initial_prompt": f"Analyze documents for case {request.case_id}: {request.document_ids}"
    }
    
    try:
        response = await http_client_service.client.post(
            f"{settings.BACKEND_URL}/api/workflows", 
            json=workflow_data
        )
        response.raise_for_status()
        
        # Trigger background processing
        background_tasks.add_task(
            execute_analysis_workflow, 
            workflow_id, 
            request.case_id,
            request.document_ids, 
            request.case_context or ""
        )
        
        return DocumentAnalysisResponse(
            workflow_id=workflow_id,
            status=DocumentAnalysisStatus.PENDING_PLANNING,
            message=f"Document analysis workflow {workflow_id} triggered"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/{workflow_id}/status")
async def get_analysis_status(workflow_id: str):
    """Get document analysis workflow status"""
    state = await backend_api_service.load_workflow_state(workflow_id)
    if not state:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return state