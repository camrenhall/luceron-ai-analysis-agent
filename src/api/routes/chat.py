"""
Chat interface API routes (stateless).
"""

import asyncio
import json
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models import ChatRequest
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


async def process_analysis_background(request: ChatRequest):
    """Background task for processing analysis (stateless)"""
    try:
        logger.info(f"Starting background analysis for case {request.case_id}")
        
        # Execute agent with enhanced context (no workflow tracking)
        callback_handler = DocumentAnalysisCallbackHandler(None)  # No workflow ID
        agent = create_document_analysis_agent(None)  # No workflow ID
        
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
        
        logger.info(f"✅ Background analysis completed for case {request.case_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed for case {request.case_id}: {e}")


@router.post("/notify")
async def notify_analysis_work(request: ChatRequest):
    """Fire-and-forget notification for analysis work (stateless)"""
    
    try:
        # Schedule background processing (don't await!) - no workflow tracking
        asyncio.create_task(process_analysis_background(request))
        
        # Return immediately
        return {
            "status": "accepted",
            "case_id": request.case_id,
            "message": "Analysis work scheduled"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule analysis: {e}")


@router.post("/chat")
async def chat_with_analysis_agent(request: ChatRequest):
    """Interactive chat interface for document analysis (stateless)"""
    
    async def generate_stream():
        try:
            yield f"data: {json.dumps({'type': 'analysis_started', 'case_id': request.case_id})}\n\n"
            
            # Execute agent with enhanced context (no workflow tracking)
            callback_handler = DocumentAnalysisCallbackHandler(None)  # No workflow ID
            agent = create_document_analysis_agent(None)  # No workflow ID
            
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
            
            logger.info(f"✅ Analysis completed for case {request.case_id}")
            
            # Stream final result
            yield f"data: {json.dumps({'type': 'final_response', 'response': output})}\n\n"
            yield f"data: {json.dumps({'type': 'analysis_complete', 'case_id': request.case_id})}\n\n"
            
        except Exception as e:
            logger.error(f"❌ Analysis failed for case {request.case_id}: {e}")
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )


