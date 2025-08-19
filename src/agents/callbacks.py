"""
Callback handlers for stateful document analysis agents.
"""

import logging
import json
from datetime import datetime
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from typing import Optional, Dict, Any

from services import backend_api_service

logger = logging.getLogger(__name__)


class DocumentAnalysisCallbackHandler(BaseCallbackHandler):
    """Callback handler for stateful document analysis conversation tracking"""
    
    def __init__(self, conversation_id: Optional[str] = None):
        self.conversation_id = conversation_id
        self.current_llm_run_id = None
        self.current_tool_name = None
        self.current_tool_input = None
    
    async def on_llm_start(self, serialized, prompts, **kwargs):
        """Record when LLM processing starts"""
        self.current_llm_run_id = kwargs.get('run_id')
        logger.info("🧠 Agent analyzing document analysis strategy...")
        
        if self.conversation_id:
            try:
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="system",
                    content={
                        "event": "llm_start",
                        "message": "Agent starting analysis",
                        "model_info": serialized.get('name', 'unknown'),
                        "timestamp": datetime.now().isoformat()
                    },
                    model_used=serialized.get('name', 'gpt-4-turbo')
                )
            except Exception as e:
                logger.warning(f"Could not store LLM start message: {e}")
    
    async def on_tool_start(self, serialized, input_str, **kwargs):
        """Record when tool execution starts"""
        tool_name = serialized.get('name', 'Unknown tool')
        self.current_tool_name = tool_name
        self.current_tool_input = input_str
        
        logger.info(f"🔧 Executing {tool_name}")
        
        if self.conversation_id:
            try:
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="assistant",
                    content={
                        "event": "tool_start",
                        "message": f"Executing tool: {tool_name}",
                        "tool_input_preview": str(input_str)[:200] + "..." if len(str(input_str)) > 200 else str(input_str),
                        "timestamp": datetime.now().isoformat()
                    },
                    function_name=tool_name,
                    function_arguments={"input": input_str} if input_str else None,
                    model_used="agent_system"
                )
            except Exception as e:
                logger.warning(f"Could not store tool start message: {e}")
    
    async def on_tool_end(self, output, **kwargs):
        """Record when tool execution completes"""
        logger.info("✅ Tool execution completed")
        
        if self.conversation_id and self.current_tool_name:
            try:
                # Safely convert output to string and truncate if needed
                output_str = str(output)
                output_preview = output_str[:500] + "..." if len(output_str) > 500 else output_str
                
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="function",
                    content={
                        "event": "tool_end",
                        "message": f"Tool {self.current_tool_name} completed",
                        "output_preview": output_preview,
                        "output_length": len(output_str),
                        "timestamp": datetime.now().isoformat()
                    },
                    function_name=self.current_tool_name,
                    function_arguments={"input": self.current_tool_input} if self.current_tool_input else None,
                    function_response={"output": output_preview, "full_length": len(output_str)},
                    model_used="agent_system"
                )
            except Exception as e:
                logger.warning(f"Could not store tool end message: {e}")
            finally:
                # Reset current tool tracking
                self.current_tool_name = None
                self.current_tool_input = None
    
    async def on_agent_action(self, action: AgentAction, **kwargs):
        """Record agent decision-making process"""
        logger.info(f"🤖 Agent decided: {action.tool}")
        
        if self.conversation_id:
            try:
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="assistant",
                    content={
                        "event": "agent_action",
                        "message": f"Agent decided to use tool: {action.tool}",
                        "tool_selected": action.tool,
                        "reasoning": action.log[:300] + "..." if len(action.log) > 300 else action.log,
                        "timestamp": datetime.now().isoformat()
                    },
                    function_name=action.tool,
                    function_arguments=action.tool_input,
                    model_used="agent_reasoning"
                )
            except Exception as e:
                logger.warning(f"Could not store agent action message: {e}")
    
    async def on_agent_finish(self, finish: AgentFinish, **kwargs):
        """Record when agent completes its work"""
        logger.info("✅ Agent analysis completed")
        
        if self.conversation_id:
            try:
                # Extract the final output
                final_output = finish.return_values.get('output', 'Analysis completed')
                output_preview = str(final_output)[:300] + "..." if len(str(final_output)) > 300 else str(final_output)
                
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="assistant",
                    content={
                        "event": "agent_finish",
                        "message": "Agent completed analysis",
                        "final_output_preview": output_preview,
                        "completion_status": "success",
                        "timestamp": datetime.now().isoformat()
                    },
                    model_used="agent_system"
                )
            except Exception as e:
                logger.warning(f"Could not store agent finish message: {e}")
    
    async def on_tool_error(self, error, **kwargs):
        """Record tool execution errors"""
        logger.error(f"❌ Tool execution error: {error}")
        
        if self.conversation_id and self.current_tool_name:
            try:
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="system",
                    content={
                        "event": "tool_error",
                        "message": f"Tool {self.current_tool_name} encountered an error",
                        "error_message": str(error),
                        "timestamp": datetime.now().isoformat()
                    },
                    function_name=self.current_tool_name,
                    function_response={"error": str(error)},
                    model_used="agent_system"
                )
            except Exception as e:
                logger.warning(f"Could not store tool error message: {e}")
    
    async def on_llm_error(self, error, **kwargs):
        """Record LLM processing errors"""
        logger.error(f"❌ LLM processing error: {error}")
        
        if self.conversation_id:
            try:
                await backend_api_service.add_message(
                    conversation_id=self.conversation_id,
                    role="system",
                    content={
                        "event": "llm_error",
                        "message": "LLM processing encountered an error",
                        "error_message": str(error),
                        "timestamp": datetime.now().isoformat()
                    },
                    model_used="agent_system"
                )
            except Exception as e:
                logger.warning(f"Could not store LLM error message: {e}")