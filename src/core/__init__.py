"""
Core package for application lifecycle and background tasks.
"""

from .background_tasks import execute_analysis_workflow
from .lifecycle import lifespan

__all__ = [
    "execute_analysis_workflow",
    "lifespan"
]