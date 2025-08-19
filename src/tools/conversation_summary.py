"""
Conversation summary management tools for token optimization and memory management.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any
from langchain.tools import BaseTool

from services import backend_api_service

logger = logging.getLogger(__name__)


class ManageConversationSummaryTool(BaseTool):
    """Tool for managing conversation summaries and memory optimization"""
    name: str = "manage_conversation_summary"
    description: str = """Manage conversation summaries for memory and token optimization.
    Input: JSON with action (create_summary, get_summary, check_needs_summary), conversation_id, and optional messages_to_summarize (default 15).
    Use this tool when conversations are getting long to create summaries that preserve important context while reducing token usage."""
    
    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, **kwargs) -> str:
        """
        Manage conversation summaries for efficient memory management.
        """
        try:
            logger.info("📊 Managing conversation summary")
            
            # Handle LangChain's nested kwargs structure
            if 'kwargs' in kwargs:
                data = kwargs['kwargs']
            else:
                data = kwargs
            
            action = data.get("action")
            conversation_id = data.get("conversation_id")
            
            if not action:
                raise ValueError("action is required (create_summary, get_summary, check_needs_summary)")
            
            if not conversation_id and action != "check_needs_summary":
                raise ValueError("conversation_id is required for this action")
            
            logger.info(f"📊 Executing action '{action}' for conversation {conversation_id}")
            
            if action == "create_summary":
                messages_to_summarize = data.get("messages_to_summarize", 15)
                
                # Create the summary
                summary_result = await backend_api_service.create_auto_summary(
                    conversation_id=conversation_id,
                    messages_to_summarize=messages_to_summarize
                )
                
                logger.info(f"📊 Created summary for conversation {conversation_id}: {messages_to_summarize} messages")
                
                return json.dumps({
                    "status": "summary_created",
                    "conversation_id": conversation_id,
                    "messages_summarized": messages_to_summarize,
                    "summary_id": summary_result.get("summary_id"),
                    "summary_content_preview": summary_result.get("summary_content", "")[:200] + "..." if len(summary_result.get("summary_content", "")) > 200 else summary_result.get("summary_content", ""),
                    "token_optimization": "Older messages compressed to summary format for efficient token usage"
                })
            
            elif action == "get_summary":
                # Get the latest summary
                latest_summary = await backend_api_service.get_latest_summary(conversation_id)
                
                if not latest_summary:
                    return json.dumps({
                        "status": "no_summary_found",
                        "conversation_id": conversation_id,
                        "message": "No summaries exist for this conversation yet"
                    })
                
                logger.info(f"📊 Retrieved latest summary for conversation {conversation_id}")
                
                return json.dumps({
                    "status": "summary_retrieved",
                    "conversation_id": conversation_id,
                    "summary": latest_summary,
                    "messages_summarized": latest_summary.get("messages_summarized", 0),
                    "created_at": latest_summary.get("created_at")
                })
            
            elif action == "check_needs_summary":
                # Check if conversation needs summarization
                threshold = data.get("threshold", 20)
                
                if not conversation_id:
                    return json.dumps({
                        "status": "error",
                        "message": "conversation_id required for check_needs_summary action"
                    })
                
                needs_summary = await backend_api_service.should_create_summary(
                    conversation_id=conversation_id,
                    threshold=threshold
                )
                
                message_count = await backend_api_service.get_message_count(conversation_id)
                
                logger.info(f"📊 Checked summary needs for conversation {conversation_id}: {message_count} messages, threshold {threshold}")
                
                return json.dumps({
                    "status": "summary_check_complete",
                    "conversation_id": conversation_id,
                    "needs_summary": needs_summary,
                    "current_message_count": message_count,
                    "threshold": threshold,
                    "recommendation": "Create summary to optimize token usage" if needs_summary else "No summary needed yet"
                })
            
            else:
                return json.dumps({
                    "status": "invalid_action",
                    "message": f"Unknown action '{action}'. Valid actions: create_summary, get_summary, check_needs_summary"
                })
            
        except Exception as e:
            error_msg = f"Failed to manage conversation summary: {str(e)}"
            logger.error(f"📊 Summary management ERROR: {error_msg}")
            return json.dumps({"error": error_msg})


class OptimizeConversationMemoryTool(BaseTool):
    """Tool for comprehensive conversation memory optimization"""
    name: str = "optimize_conversation_memory"
    description: str = """Optimize conversation memory by creating summaries and managing token usage.
    Input: JSON with conversation_id and optional optimization_strategy (aggressive, moderate, conservative).
    Automatically analyzes conversation length and creates appropriate summaries to balance context preservation with token efficiency."""
    
    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, **kwargs) -> str:
        """
        Perform comprehensive memory optimization for a conversation.
        """
        try:
            logger.info("🚀 Starting comprehensive memory optimization")
            
            # Handle LangChain's nested kwargs structure
            if 'kwargs' in kwargs:
                data = kwargs['kwargs']
            else:
                data = kwargs
            
            conversation_id = data.get("conversation_id")
            if not conversation_id:
                raise ValueError("conversation_id is required")
            
            strategy = data.get("optimization_strategy", "moderate")
            
            logger.info(f"🚀 Optimizing memory for conversation {conversation_id} with {strategy} strategy")
            
            # Get current conversation stats
            message_count = await backend_api_service.get_message_count(conversation_id)
            existing_summary = await backend_api_service.get_latest_summary(conversation_id)
            
            optimization_results = {
                "conversation_id": conversation_id,
                "strategy": strategy,
                "initial_message_count": message_count,
                "had_existing_summary": bool(existing_summary),
                "actions_taken": [],
                "final_status": {}
            }
            
            # Determine thresholds based on strategy
            if strategy == "aggressive":
                threshold = 15
                messages_to_summarize = 20
            elif strategy == "conservative":  
                threshold = 30
                messages_to_summarize = 10
            else:  # moderate
                threshold = 20
                messages_to_summarize = 15
            
            # Check if summarization is needed
            needs_summary = message_count > threshold
            
            if needs_summary:
                logger.info(f"🚀 Creating summary: {message_count} messages exceeds threshold of {threshold}")
                
                # Create summary
                summary_result = await backend_api_service.create_auto_summary(
                    conversation_id=conversation_id,
                    messages_to_summarize=messages_to_summarize
                )
                
                optimization_results["actions_taken"].append("created_summary")
                optimization_results["summary_created"] = {
                    "messages_summarized": messages_to_summarize,
                    "summary_id": summary_result.get("summary_id")
                }
                
                # Update final message count after summarization
                final_message_count = await backend_api_service.get_message_count(conversation_id)
                optimization_results["final_status"]["message_count"] = final_message_count
                optimization_results["final_status"]["tokens_saved_estimate"] = messages_to_summarize * 100  # Rough estimate
                
                logger.info(f"🚀 Memory optimization complete: {message_count} -> {final_message_count} messages")
            else:
                optimization_results["actions_taken"].append("no_action_needed")
                optimization_results["final_status"]["message_count"] = message_count
                optimization_results["final_status"]["reason"] = f"Message count ({message_count}) below threshold ({threshold})"
            
            # Calculate memory efficiency
            if existing_summary:
                optimization_results["final_status"]["memory_efficiency"] = "high"
                optimization_results["final_status"]["note"] = "Conversation has active memory management"
            elif needs_summary:
                optimization_results["final_status"]["memory_efficiency"] = "optimized"
                optimization_results["final_status"]["note"] = "Memory optimized during this operation"
            else:
                optimization_results["final_status"]["memory_efficiency"] = "good"
                optimization_results["final_status"]["note"] = "Conversation length within optimal range"
            
            return json.dumps({
                "status": "memory_optimization_complete",
                "optimization_results": optimization_results
            })
            
        except Exception as e:
            error_msg = f"Failed to optimize conversation memory: {str(e)}"
            logger.error(f"🚀 Memory optimization ERROR: {error_msg}")
            return json.dumps({"error": error_msg})