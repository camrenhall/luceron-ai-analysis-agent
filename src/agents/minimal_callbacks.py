"""
Minimal callback handler that only stores essential conversation messages.
"""

import logging
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentFinish
from typing import Optional

from services import backend_api_service

logger = logging.getLogger(__name__)


class MinimalConversationCallbackHandler(BaseCallbackHandler):
    """Minimal callback handler that only stores final natural language responses"""
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.final_response_stored = False
    
    async def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Store only the final natural language response from the agent"""
        if self.conversation_id and not self.final_response_stored:
            try:
                # Extract the final natural language output
                final_output = finish.return_values.get('output', 'Analysis completed')
                
                # Store only the clean natural language response
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="assistant",
                    content={
                        "text": final_output,
                        "analysis_type": "interactive_analysis",
                        "response_type": "natural_language"
                    },
                    model_used="claude-3-5-sonnet-20241022"
                )
                
                self.final_response_stored = True
                logger.info(f"💬 Stored final natural language response for conversation {self.conversation_id}")
                
            except Exception as e:
                logger.warning(f"Could not store final response message: {e}")
    
    # Override all other callback methods to do nothing (just log)
    async def on_llm_start(self, serialized, prompts, **kwargs):
        """LLM processing starts - no storage needed"""
        logger.debug("🧠 Agent starting analysis...")
    
    async def on_tool_start(self, serialized, input_str, **kwargs):
        """Tool execution starts - no storage needed"""
        tool_name = serialized.get('name', 'Unknown tool')
        logger.debug(f"🔧 Executing {tool_name}")
    
    async def on_tool_end(self, output, **kwargs):
        """Tool execution completes - no storage needed"""
        logger.debug("✅ Tool execution completed")
    
    async def on_agent_action(self, action, **kwargs):
        """Agent decision-making - no storage needed"""
        logger.debug(f"🤖 Agent decided: {action.tool}")
    
    async def on_tool_error(self, error, **kwargs):
        """Tool execution error - log but don't store"""
        logger.error(f"❌ Tool execution error: {error}")
    
    async def on_llm_error(self, error, **kwargs):
        """LLM processing error - log but don't store"""
        logger.error(f"❌ LLM processing error: {error}")
        
        # Only store critical errors that affect conversation state
        if self.conversation_id:
            try:
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="system",
                    content={
                        "error": str(error),
                        "error_type": "llm_processing_failure",
                        "timestamp": datetime.now().isoformat()
                    },
                    model_used="system"
                )
            except Exception as e:
                logger.warning(f"Could not store critical error message: {e}")