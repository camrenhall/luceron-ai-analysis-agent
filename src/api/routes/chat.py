"""
Chat interface API routes with stateful conversation management.
"""

import asyncio
import json
import logging
from datetime import datetime
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
    """Background task for processing analysis with conversation management"""
    conversation_id = None
    try:
        logger.info(f"🚀 Starting stateful background analysis for case {request.case_id}")
        
        # Step 1: Get or create conversation for this case and agent type
        conversation_id = await backend_api_service.get_or_create_conversation(
            case_id=request.case_id,
            agent_type="DocumentAnalysisAgent"
        )
        
        # Step 2: Load existing context from previous interactions
        existing_context = await backend_api_service.get_case_agent_context(
            case_id=request.case_id,
            agent_type="DocumentAnalysisAgent"
        )
        
        # Step 3: Add user message to conversation
        await backend_api_service.add_message(
            conversation_id=conversation_id,
            role="user",
            content={
                "text": request.message,
                "case_id": request.case_id,
                "request_type": "background_analysis"
            },
            model_used="gpt-4-turbo"
        )
        
        # Step 4: Check if conversation needs summarization
        if await backend_api_service.should_create_summary(conversation_id):
            logger.info(f"📊 Creating summary for long conversation {conversation_id}")
            await backend_api_service.create_auto_summary(conversation_id)
        
        # Step 5: Get recent conversation history for context
        conversation_history = await backend_api_service.get_conversation_history(
            conversation_id=conversation_id,
            limit=10  # Recent messages only to save tokens
        )
        
        # Step 6: Create context-aware system message
        context_prompt = f"""You are a senior legal document analysis partner with access to case context.
        
Case ID: {request.case_id}
Conversation: {conversation_id}
        
Previous Context: {existing_context if existing_context else 'No previous context'}
        
Instructions: Use the get_all_case_analyses tool to retrieve and review ALL document analyses for this case. 
Analyze patterns, identify inconsistencies, and provide comprehensive insights. Store important findings in persistent context for future reference.
        
User Query: {request.message}"""
        
        # Step 7: Execute agent with conversation context
        callback_handler = DocumentAnalysisCallbackHandler(conversation_id)
        agent = create_document_analysis_agent(conversation_id)
        
        result = await agent.ainvoke(
            {"input": context_prompt},
            config={"callbacks": [callback_handler]}
        )
        
        # Step 8: Extract and store response
        raw_output = result.get("output", "Analysis completed")
        output = _extract_text_from_llm_response(raw_output)
        
        # Step 9: Add assistant response to conversation
        await backend_api_service.add_message(
            conversation_id=conversation_id,
            role="assistant", 
            content={
                "text": output,
                "analysis_type": "comprehensive_review",
                "completion_status": "completed"
            },
            model_used="gpt-4-turbo"
        )
        
        # Step 10: Store important findings in persistent context
        if "findings" in output.lower() or "important" in output.lower():
            context_key = f"analysis_result_{datetime.now().strftime('%Y%m%d_%H%M')}"
            await backend_api_service.store_context(
                case_id=request.case_id,
                agent_type="DocumentAnalysisAgent",
                context_key=context_key,
                context_value={
                    "analysis_summary": output[:500] + "..." if len(output) > 500 else output,
                    "conversation_id": conversation_id,
                    "completed_at": datetime.now().isoformat(),
                    "request_type": "background_analysis"
                },
                expires_at=None  # Persist indefinitely
            )
        
        logger.info(f"✅ Stateful background analysis completed for case {request.case_id}")
        
    except Exception as e:
        logger.error(f"❌ Background analysis failed for case {request.case_id}: {e}")
        if conversation_id:
            # Log error to conversation
            try:
                await backend_api_service.add_message(
                    conversation_id=conversation_id,
                    role="system",
                    content={"error": str(e), "error_type": "background_analysis_failure"},
                    model_used="system"
                )
            except:
                pass  # Don't fail if we can't log the error


@router.post("/notify")
async def notify_analysis_work(request: ChatRequest):
    """Fire-and-forget notification for analysis work with conversation tracking"""
    
    try:
        # Get or create conversation for tracking
        conversation_id = await backend_api_service.get_or_create_conversation(
            case_id=request.case_id,
            agent_type="DocumentAnalysisAgent"
        )
        
        # Schedule background processing (don't await!)
        asyncio.create_task(process_analysis_background(request))
        
        # Return immediately with conversation tracking info
        return {
            "status": "accepted",
            "case_id": request.case_id,
            "conversation_id": conversation_id,
            "message": "Analysis work scheduled with conversation tracking"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to schedule analysis: {e}")


@router.post("/chat")
async def chat_with_analysis_agent(request: ChatRequest):
    """Interactive chat interface with stateful conversation management and streaming"""
    
    async def generate_stream():
        conversation_id = None
        try:
            # Step 1: Get or create conversation
            conversation_id = await backend_api_service.get_or_create_conversation(
                case_id=request.case_id,
                agent_type="DocumentAnalysisAgent"
            )
            
            yield f"data: {json.dumps({'type': 'conversation_ready', 'conversation_id': conversation_id, 'case_id': request.case_id})}\\n\\n"
            
            # Step 2: Load existing context and conversation history
            existing_context = await backend_api_service.get_case_agent_context(
                case_id=request.case_id,
                agent_type="DocumentAnalysisAgent"
            )
            
            # Step 3: Check if conversation needs summarization first
            if await backend_api_service.should_create_summary(conversation_id):
                yield f"data: {json.dumps({'type': 'creating_summary', 'message': 'Managing conversation length...'})}\\n\\n"
                await backend_api_service.create_auto_summary(conversation_id)
            
            # Step 4: Add user message to conversation
            await backend_api_service.add_message(
                conversation_id=conversation_id,
                role="user",
                content={
                    "text": request.message,
                    "case_id": request.case_id,
                    "request_type": "interactive_chat"
                },
                model_used="gpt-4-turbo"
            )
            
            yield f"data: {json.dumps({'type': 'processing_started', 'message': 'Agent is analyzing with full conversation context...'})}\\n\\n"
            
            # Step 5: Get conversation history for context
            conversation_history = await backend_api_service.get_conversation_history(
                conversation_id=conversation_id,
                limit=10
            )
            
            # Step 6: Get latest summary if available
            latest_summary = await backend_api_service.get_latest_summary(conversation_id)
            
            # Step 7: Create rich context-aware system message
            context_elements = []
            if existing_context:
                context_elements.append(f"Previous Context: {json.dumps(existing_context, indent=2)}")
            if latest_summary:
                context_elements.append(f"Previous Conversation Summary: {latest_summary['summary_content']}")
            if conversation_history:
                recent_messages = [f"{msg['role']}: {msg['content'].get('text', '')[:100]}..." for msg in conversation_history[-3:]]
                context_elements.append(f"Recent Messages: {'; '.join(recent_messages)}")
            
            context_prompt = f"""You are a senior legal document analysis partner with comprehensive case context and conversation history.

Case ID: {request.case_id}
Conversation ID: {conversation_id}

{chr(10).join(context_elements) if context_elements else 'Starting fresh conversation'}

Current User Query: {request.message}

Instructions: 
1. Use the get_all_case_analyses tool to retrieve and review ALL document analyses for this case
2. Consider the conversation history and previous context when formulating your response
3. Analyze patterns, identify inconsistencies, and provide comprehensive insights
4. Be specific and reference previous findings when relevant
5. Update persistent context with any important new findings"""
            
            # Step 8: Execute agent with full context
            callback_handler = DocumentAnalysisCallbackHandler(conversation_id)
            agent = create_document_analysis_agent(conversation_id)
            
            result = await agent.ainvoke(
                {"input": context_prompt},
                config={"callbacks": [callback_handler]}
            )
            
            # Step 9: Extract and process response
            raw_output = result.get("output", "Analysis completed")
            output = _extract_text_from_llm_response(raw_output)
            
            # Step 10: Add assistant response to conversation
            await backend_api_service.add_message(
                conversation_id=conversation_id,
                role="assistant",
                content={
                    "text": output,
                    "analysis_type": "interactive_analysis",
                    "conversation_context_used": bool(existing_context or latest_summary)
                },
                model_used="gpt-4-turbo"
            )
            
            # Step 11: Store important findings in persistent context
            if any(keyword in output.lower() for keyword in ["finding", "important", "key", "significant", "concern"]):
                context_key = f"chat_insight_{datetime.now().strftime('%Y%m%d_%H%M')}"
                await backend_api_service.store_context(
                    case_id=request.case_id,
                    agent_type="DocumentAnalysisAgent",
                    context_key=context_key,
                    context_value={
                        "insight_summary": output[:300] + "..." if len(output) > 300 else output,
                        "conversation_id": conversation_id,
                        "derived_from": "interactive_chat",
                        "timestamp": datetime.now().isoformat()
                    },
                    expires_at=None  # Keep indefinitely
                )
            
            # Step 12: Stream final results
            yield f"data: {json.dumps({'type': 'final_response', 'response': output, 'conversation_id': conversation_id})}\\n\\n"
            yield f"data: {json.dumps({'type': 'conversation_updated', 'case_id': request.case_id, 'conversation_id': conversation_id})}\\n\\n"
            
        except Exception as e:
            logger.error(f"❌ Stateful chat analysis failed for case {request.case_id}: {e}")
            
            # Log error to conversation if we have one
            if conversation_id:
                try:
                    await backend_api_service.add_message(
                        conversation_id=conversation_id,
                        role="system",
                        content={"error": str(e), "error_type": "chat_analysis_failure"},
                        model_used="system"
                    )
                except:
                    pass
            
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\\n\\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )