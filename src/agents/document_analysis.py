"""
Document analysis agent creation and configuration for stateful conversations.
"""

import logging
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from typing import Optional

from config import settings
from tools import tool_factory
from utils.prompts import load_system_prompt, load_conversation_context_prompt

logger = logging.getLogger(__name__)


def create_document_analysis_agent(conversation_id: Optional[str] = None) -> AgentExecutor:
    """
    Create document analysis agent with conversation awareness.
    
    Args:
        conversation_id: Optional conversation ID for stateful tracking.
                        If provided, enables conversation history and context awareness.
    
    Returns:
        Configured AgentExecutor for document analysis tasks.
    """
    
    # Log agent creation with conversation context
    if conversation_id:
        logger.info(f"🤖 Creating conversation-aware document analysis agent for conversation {conversation_id}")
    else:
        logger.info("🤖 Creating stateless document analysis agent")
    
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=settings.ANTHROPIC_API_KEY,
        temperature=0.1
    )
    
    tools = tool_factory.create_all_tools()
    
    # Load base system prompt
    base_system_prompt = load_system_prompt('document_analysis_agent_system_prompt.md')
    
    # Enhance system prompt for conversation-aware agents
    if conversation_id:
        conversation_context_template = load_conversation_context_prompt()
        enhanced_system_prompt = f"{base_system_prompt}\n\n{conversation_context_template.format(conversation_id=conversation_id)}"
        system_prompt = enhanced_system_prompt
    else:
        system_prompt = base_system_prompt
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    # Configure agent executor with conversation-specific settings
    executor_config = {
        "agent": agent,
        "tools": tools,
        "verbose": True,
        "handle_parsing_errors": True,
        "max_iterations": 25 if conversation_id else 20,  # More iterations for conversation-aware agents
        "return_intermediate_steps": True
    }
    
    executor = AgentExecutor(**executor_config)
    
    # Log successful creation
    mode = "conversation-aware" if conversation_id else "stateless"
    logger.info(f"✅ Successfully created {mode} document analysis agent")
    
    return executor