# Prompts Directory

This directory contains the unified prompt foundation for the Case Discovery Agent.

## Unified Architecture

### Single Prompt File
- `agent_prompt.md` - **The complete foundation prompt** that defines:
  - Agent identity and role
  - Available tools and capabilities  
  - Instructions for all scenarios
  - Response patterns and examples
  - Behavioral guidelines

## Design Philosophy

This unified approach eliminates complexity by having **one source of truth** for the entire agent:
- No variable substitution or formatting
- No separate chat context files
- No conversation enhancement layers
- All behavior defined in one comprehensive prompt

## Usage

Load the unified prompt with:
```python
from utils.prompts import load_system_prompt
prompt = load_system_prompt('agent_prompt.md')
```

