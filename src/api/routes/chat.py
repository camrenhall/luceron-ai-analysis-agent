"""
Chat interface API routes.
"""

import json
import uuid
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models import ChatRequest, AWSAnalysisResult, WorkflowStatus
from services import http_client_service, backend_api_service
from agents import DocumentAnalysisCallbackHandler, create_document_analysis_agent
from config import settings

router = APIRouter()


@router.post("/chat")
async def chat_with_analysis_agent(request: ChatRequest):
    """Interactive chat interface for document analysis"""
    
    workflow_id = f"wf_chat_{uuid.uuid4().hex[:8]}"
    
    # Create workflow for interactive session
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": WorkflowStatus.PENDING.value,
        "initial_prompt": request.message,
        "document_ids": request.document_ids or [],
        "priority": "interactive"
    }
    
    try:
        response = await http_client_service.client.post(
            f"{settings.BACKEND_URL}/api/workflows", 
            json=workflow_data
        )
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {e}")
    
    async def generate_stream():
        try:
            yield f"data: {json.dumps({'type': 'workflow_started', 'workflow_id': workflow_id})}\n\n"
            
            # Execute agent
            callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
            agent = create_document_analysis_agent(workflow_id)
            
            result = await agent.ainvoke(
                {"input": request.message},
                config={"callbacks": [callback_handler]}
            )
            
            # Update final status
            await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.COMPLETED)
            
            # Stream final result
            output = result.get("output", "Analysis completed")
            yield f"data: {json.dumps({'type': 'final_response', 'response': output})}\n\n"
            yield f"data: {json.dumps({'type': 'workflow_complete', 'workflow_id': workflow_id})}\n\n"
            
        except Exception as e:
            await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.FAILED)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


@router.post("/aws-analysis-result")
async def receive_aws_analysis(request: AWSAnalysisResult):
    """Endpoint for AWS Lambda to POST document analysis results for agent reasoning."""
    
    # Create or use existing workflow
    workflow_id = request.workflow_id or f"wf_aws_{uuid.uuid4().hex[:8]}"
    
    # Create workflow for tracking reasoning chain
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_type": "ReasoningAgent",
        "case_id": request.case_id,
        "status": WorkflowStatus.REASONING.value,
        "initial_prompt": f"Evaluate analysis results for case {request.case_id}",
        "document_ids": request.document_ids,
        "priority": "aws_batch"
    }
    
    try:
        # Create workflow in backend
        response = await http_client_service.client.post(
            f"{settings.BACKEND_URL}/api/workflows", 
            json=workflow_data
        )
        response.raise_for_status()
        
        # Update workflow status to indicate reasoning phase
        await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.REASONING)
        
        # Initialize agent for reasoning
        callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
        agent = create_document_analysis_agent(workflow_id)
        
        # Construct prompt for agent to reason over AWS results
        reasoning_prompt = f"""
        I have received the following document analysis results from AWS processing:
        
        Case ID: {request.case_id}
        Document IDs: {request.document_ids}
        
        Analysis Data:
        {json.dumps(request.analysis_data, indent=2)}
        
        Please evaluate these results and provide:
        1. Key findings and patterns
        2. Financial red flags or concerns
        3. Recommendations for further investigation
        4. Overall assessment of document completeness and authenticity
        """
        
        # Execute reasoning
        result = await agent.ainvoke(
            {"input": reasoning_prompt},
            config={"callbacks": [callback_handler]}
        )
        
        # Update status to evaluating
        await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.EVALUATING)
        
        # Store reasoning results
        await backend_api_service.add_reasoning_step(
            workflow_id=workflow_id,
            thought="Completed evaluation of AWS analysis results",
            action="store_evaluation",
            action_output=result.get("output", "")
        )
        
        # Mark as completed
        await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.COMPLETED)
        
        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "evaluation": result.get("output", "Evaluation completed")
        }
        
    except Exception as e:
        await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.FAILED)
        raise HTTPException(status_code=500, detail=f"Failed to process AWS analysis results: {e}")