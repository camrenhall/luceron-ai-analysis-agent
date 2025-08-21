"""
Prompt loading utility for unified prompt architecture.
"""

import os
import logging

logger = logging.getLogger(__name__)


def load_system_prompt(filename: str) -> str:
    """Load system prompt from markdown file in prompts directory"""
    prompts_path = os.path.join(os.path.dirname(__file__), '..', '..', 'prompts', filename)
    try:
        with open(prompts_path, 'r') as f:
            content = f.read().strip()
            if not content:
                logger.critical(f"CRITICAL FAILURE: Prompt file {filename} is empty")
                raise RuntimeError(f"CRITICAL FAILURE: Required prompt file {filename} is empty")
            logger.info(f"Successfully loaded prompt: {filename}")
            return content
    except FileNotFoundError:
        logger.critical(f"CRITICAL FAILURE: Prompt file {filename} not found in prompts/ directory")
        raise RuntimeError(f"CRITICAL FAILURE: Required prompt file {filename} not found in prompts/ directory")