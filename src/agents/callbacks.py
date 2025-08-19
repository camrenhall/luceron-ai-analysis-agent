"""
Callback handlers for document analysis agents.
"""

import logging
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish
from typing import Optional

logger = logging.getLogger(__name__)


class DocumentAnalysisCallbackHandler(BaseCallbackHandler):
    """Callback handler for document analysis logging (stateless)"""
    
    def __init__(self, workflow_id: Optional[str] = None):
        # workflow_id is ignored - keeping for compatibility
        pass
    
    async def on_llm_start(self, serialized, prompts, **kwargs):
        logger.info("🧠 Agent analyzing document analysis strategy...")
    
    async def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get('name', 'Unknown tool')
        logger.info(f"🔧 Executing {tool_name}")
    
    async def on_tool_end(self, output, **kwargs):
        logger.info("✅ Tool execution completed")
    
    async def on_agent_action(self, action: AgentAction, **kwargs):
        logger.info(f"🤖 Agent decided: {action.tool}")