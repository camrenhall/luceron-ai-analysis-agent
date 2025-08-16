"""
Utility functions for loading and managing prompts.
"""

import os
import logging

logger = logging.getLogger(__name__)


def load_system_prompt(filename: str) -> str:
    """Load system prompt from markdown file in prompts directory"""
    try:
        # Try loading from prompts directory first
        prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts', filename)
        with open(prompts_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        # Fallback to current directory
        try:
            with open(filename, 'r') as f:
                return f.read()
        except FileNotFoundError:
            logger.error(f"System prompt file {filename} not found in prompts/ directory or current directory")
            raise FileNotFoundError(f"Required prompt file {filename} not found")


def load_prompt_template(filename: str) -> str:
    """Load prompt template from markdown file in prompts directory"""
    try:
        prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts', filename)
        with open(prompts_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.error(f"Prompt template file {filename} not found in prompts/ directory")
        raise FileNotFoundError(f"Required prompt template file {filename} not found")