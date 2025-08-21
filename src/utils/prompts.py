"""
Utility functions for loading and managing prompts.
"""

import os
import logging

logger = logging.getLogger(__name__)


def load_system_prompt(filename: str) -> str:
    """Load system prompt from markdown file in prompts directory - HARD FAILURE IF NOT FOUND"""
    prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts', filename)
    try:
        with open(prompts_path, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.critical(f"CRITICAL FAILURE: Prompt file {filename} is empty - this is a fatal error")
                raise RuntimeError(f"CRITICAL FAILURE: Required prompt file {filename} is empty")
            logger.info(f"Successfully loaded prompt: {filename}")
            return content
    except FileNotFoundError:
        logger.critical(f"CRITICAL FAILURE: System prompt file {filename} not found in prompts/ directory - this is a fatal error")
        raise RuntimeError(f"CRITICAL FAILURE: Required prompt file {filename} not found in prompts/ directory")


def load_prompt_template(filename: str) -> str:
    """Load prompt template from markdown file in prompts directory - HARD FAILURE IF NOT FOUND"""
    prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts', filename)
    try:
        with open(prompts_path, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.critical(f"CRITICAL FAILURE: Prompt template file {filename} is empty - this is a fatal error")
                raise RuntimeError(f"CRITICAL FAILURE: Required prompt template file {filename} is empty")
            logger.info(f"Successfully loaded prompt template: {filename}")
            return content
    except FileNotFoundError:
        logger.critical(f"CRITICAL FAILURE: Prompt template file {filename} not found in prompts/ directory - this is a fatal error")
        raise RuntimeError(f"CRITICAL FAILURE: Required prompt template file {filename} not found in prompts/ directory")


def load_conversation_context_prompt() -> str:
    """Load conversation context enhancement prompt template"""
    return load_prompt_template('conversation_context_enhancement.md')


def load_chat_context_prompt() -> str:
    """Load chat context prompt template"""
    return load_prompt_template('chat_context_prompt.md')




def load_chat_context_case_specific_prompt() -> str:
    """Load chat context case specific prompt template"""
    return load_prompt_template('chat_context_case_specific_prompt.md')


def load_chat_context_discovery_detailed_prompt() -> str:
    """Load chat context discovery detailed prompt template"""
    return load_prompt_template('chat_context_discovery_prompt.md')