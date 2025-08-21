# Prompts Directory

This directory contains all natural language prompts and instructions used by the Document Analysis Agent.

## Files

### System Prompts
- `document_analysis_agent_system_prompt.md` - Main system prompt for the document analysis agent

### Chat Context Templates
- `chat_context_prompt.md` - Basic chat context template for case analysis
- `chat_context_case_specific_prompt.md` - Extended template for case-specific analysis
- `chat_context_discovery_prompt.md` - Template for case discovery and search mode

### Conversation Enhancement
- `conversation_context_enhancement.md` - Enhancement for stateful conversations with memory

### Criteria and Rules
- `document_satisfaction_criteria.md` - Rules for determining when documents satisfy requirements

## Usage

The prompts are loaded by utility functions in `src/utils/prompts.py`:
- `load_system_prompt(filename)` - Loads system prompts
- `load_prompt_template(filename)` - Loads templates that can be formatted with variables
- `load_conversation_context_prompt()` - Loads conversation context enhancement
- `load_chat_context_prompt()` - Loads basic chat context
- `load_chat_context_case_specific_prompt()` - Loads case-specific chat context
- `load_chat_context_discovery_detailed_prompt()` - Loads discovery mode context

## Template Variables

Chat templates support Python string formatting with these variables:
- `{case_id}` - The case identifier  
- `{conversation_id}` - The conversation identifier
- `{context_elements}` - Formatted context elements
- `{user_message}` - The current user query
- `{existing_context}` - Previous context information