"""
Document Analysis Agent - Modular enterprise structure.
AI-powered document intelligence orchestrator for Family Law financial discovery.
"""

__version__ = "1.0.0"

# Import function to avoid circular dependencies at module level
def create_app():
    """Lazy import of create_app to avoid dependency issues"""
    from .api import create_app as _create_app
    return _create_app()

# Import settings directly since it has minimal dependencies
from .config import settings

__all__ = ["create_app", "settings"]