"""
Chat interface API routes.
"""

import json
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
    
    # Create workflow for interactive session (backend generates the ID)
    workflow_data = {
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": WorkflowStatus.PENDING.value,
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
    
    # Use existing workflow or create new one
    workflow_id = request.workflow_id
    
    if not workflow_id:
        # Create new workflow (backend generates the ID)
        workflow_data = {
            "agent_type": "ReasoningAgent",
            "case_id": request.case_id,
            "status": WorkflowStatus.PROCESSING.value,  # Changed from REASONING
            "initial_prompt": f"Evaluate analysis results for case {request.case_id}"
            # Removed document_ids and priority - not supported by backend
        }
        
        try:
            # Create workflow in backend
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
    else:
        # Update existing workflow status to indicate reasoning phase
        try:
            await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.PROCESSING)
        except Exception as e:
            logger.error(f"Failed to update workflow status: {e}")
    
    # Process the analysis with agent
    try:
        # Initialize agent for reasoning
        callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
        agent = create_document_analysis_agent(workflow_id)
        
        # Construct prompt for agent to reason over AWS results with comprehensive context
        reasoning_prompt = f"""
        New document analysis results have been received from AWS processing.
        
        Case ID: {request.case_id}
        Document IDs: {request.document_ids}
        
        New Analysis Data:
        {json.dumps(request.analysis_data, indent=2)}
        
        As a senior legal partner reviewing this case:
        1. First, use the get_all_case_analyses tool to retrieve ALL existing analyses for case {request.case_id}
        2. Compare this new analysis with existing documents to identify:
           - Consistency with previously analyzed documents
           - New patterns or contradictions that emerge
           - How this fits into the overall financial picture
        3. Provide comprehensive evaluation including:
           - Key findings specific to these new documents
           - How these documents relate to the broader case
           - Updated risk assessment based on complete documentation
           - Recommendations for case strategy
        
        Think systematically and provide senior-partner-level insights based on the COMPLETE case documentation.
        """
        
        # Execute reasoning
        result = await agent.ainvoke(
            {"input": reasoning_prompt},
            config={"callbacks": [callback_handler]}
        )
        
        # Update status to synthesizing results
        await backend_api_service.update_workflow_status(workflow_id, WorkflowStatus.SYNTHESIZING_RESULTS)
        
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