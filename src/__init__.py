"""
Document Analysis Agent - Modular enterprise structure.
AI-powered document intelligence orchestrator for Family Law financial discovery.
"""

from .api import create_app
from .config import settings

__version__ = "1.0.0"
__all__ = ["create_app", "settings"]