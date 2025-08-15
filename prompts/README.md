# Prompts Directory

This directory contains all natural language prompts and instructions used by the Document Analysis Agent.

## Files

### System Prompts
- `document_analysis_system_prompt.md` - System prompt for individual document analysis
- `document_analysis_agent_system_prompt.md` - System prompt for the workflow orchestration agent

### Workflow Templates
- `workflow_execution_prompt.md` - Template for workflow execution prompts with variables for case_id, document_ids, and case_context

## Usage

The prompts are loaded by utility functions in main.py:
- `load_system_prompt(filename)` - Loads system prompts
- `load_prompt_template(filename)` - Loads templates that can be formatted with variables

## Template Variables

Workflow templates support Python string formatting with these variables:
- `{case_id}` - The case identifier
- `{document_ids}` - List of document IDs to analyze
- `{case_context}` - Additional context about the case