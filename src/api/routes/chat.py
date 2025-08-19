"""
Chat interface API routes.
"""

import asyncio
import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models import ChatRequest, WorkflowStatus
from services import http_client_service, backend_api_service
from agents import DocumentAnalysisCallbackHandler, create_document_analysis_agent
from config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


def _extract_text_from_llm_response(raw_output):
    """Extract text content from LLM response, handling various response formats"""
    if isinstance(raw_output, list) and len(raw_output) > 0:
        if isinstance(raw_output[0], dict) and 'text' in raw_output[0]:
            return raw_output[0]['text']
        else:
            return str(raw_output[0])
    elif isinstance(raw_output, dict) and 'text' in raw_output:
        return raw_output['text']
    else:
        return str(raw_output)


async def process_analysis_background(workflow_id: str, request: ChatRequest):
    """Background task for processing analysis"""
    try:
        logger.info(f"Starting background analysis for workflow {workflow_id}")
        
        # Update status to processing
        await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.PROCESSING)
        
        # Execute agent with enhanced context
        callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
        agent = create_document_analysis_agent(workflow_id)
        
        # Enhance the message with case context instruction
        enhanced_message = f"""Case ID: {request.case_id}

User Query: {request.message}

Instructions: As a senior legal partner, use the get_all_case_analyses tool to retrieve and review ALL document analyses for this case before responding. 
Analyze patterns, identify inconsistencies, and provide comprehensive insights based on the complete documentation."""
        
        result = await agent.ainvoke(
            {"input": enhanced_message},
            config={"callbacks": [callback_handler]}
        )
        
        # Extract final response
        raw_output = result.get("output", "Analysis completed")
        output = _extract_text_from_llm_response(raw_output)
        
        # Store final response and update status
        await backend_api_service.update_workflow(
            workflow_id=workflow_id,
            status=WorkflowStatus.COMPLETED,
            final_response=output
        )
        
        logger.info(f"✅ Background analysis completed for workflow {workflow_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed for workflow {workflow_id}: {e}")
        await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.FAILED)


@router.post("/notify")
async def notify_analysis_work(request: ChatRequest):
    """Fire-and-forget notification for analysis work"""
    
    # Create workflow in database
    workflow_data = {
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": WorkflowStatus.PENDING.value,  # Note: PENDING, not PROCESSING
        "initial_prompt": request.message
    }
    
    try:
        response = await http_client_service.client.post(
            f"{settings.BACKEND_URL}/api/workflows", 
            json=workflow_data
        )
        response.raise_for_status()
        workflow_response = response.json()
        workflow_id = workflow_response.get("workflow_id")
        if not workflow_id:
            raise ValueError("Backend did not return workflow_id")
        
        # Schedule background processing (don't await!)
        asyncio.create_task(process_analysis_background(workflow_id, request))
        
        # Return immediately
        return {
            "status": "accepted",
            "workflow_id": workflow_id,
            "message": "Analysis work scheduled"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule analysis: {e}")


@router.post("/chat")
async def chat_with_analysis_agent(request: ChatRequest):
    """Interactive chat interface for document analysis"""
    
    # Create workflow for interactive session (backend generates the ID)
    workflow_data = {
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": WorkflowStatus.PROCESSING.value,
        "initial_prompt": request.message
    }
    
    try:
        response = await http_client_service.client.post(
            f"{settings.BACKEND_URL}/api/workflows", 
            json=workflow_data
        )
        response.raise_for_status()
        workflow_response = response.json()
        workflow_id = workflow_response.get("workflow_id")
        if not workflow_id:
            raise ValueError("Backend did not return workflow_id")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {e}")
    
    async def generate_stream():
        try:
            yield f"data: {json.dumps({'type': 'workflow_started', 'workflow_id': workflow_id})}\n\n"
            
            # Execute agent with enhanced context
            callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
            agent = create_document_analysis_agent(workflow_id)
            
            # Enhance the message with case context instruction
            enhanced_message = f"""Case ID: {request.case_id}

User Query: {request.message}

Instructions: As a senior legal partner, use the get_all_case_analyses tool to retrieve and review ALL document analyses for this case before responding. 
Analyze patterns, identify inconsistencies, and provide comprehensive insights based on the complete documentation."""
            
            result = await agent.ainvoke(
                {"input": enhanced_message},
                config={"callbacks": [callback_handler]}
            )
            
            # Extract final response
            raw_output = result.get("output", "Analysis completed")
            output = _extract_text_from_llm_response(raw_output)
            
            # Store final response and update status
            await backend_api_service.update_workflow(
                workflow_id=workflow_id,
                status=WorkflowStatus.COMPLETED,
                final_response=output
            )
            
            logger.info(f"✅ Stored final response for workflow {workflow_id}")
            
            # Stream final result
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


