"""
Callback handlers for document analysis agents.
"""

from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish

from services import backend_api_service


class DocumentAnalysisCallbackHandler(BaseCallbackHandler):
    """Callback handler for document analysis workflow state persistence"""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
    
    async def on_llm_start(self, serialized, prompts, **kwargs):
        await backend_api_service.add_reasoning_step(
            self.workflow_id, 
            "Agent analyzing document analysis strategy..."
        )
    
    async def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get('name', 'Unknown tool')
        await backend_api_service.add_reasoning_step(
            self.workflow_id,
            f"Executing {tool_name}",
            action=tool_name,
            action_input={"input": str(input_str)[:300]}
        )
    
    async def on_tool_end(self, output, **kwargs):
        await backend_api_service.add_reasoning_step(
            self.workflow_id,
            "Tool execution completed",
            action_output=str(output)[:500]
        )
    
    async def on_agent_action(self, action: AgentAction, **kwargs):
        await backend_api_service.add_reasoning_step(
            self.workflow_id,
            f"Agent decided: {action.tool}. Reasoning: {action.log[:300]}..."
        )