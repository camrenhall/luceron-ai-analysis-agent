"""
Enumerations for the document analysis system.
"""

from enum import Enum


class WorkflowStatus(str, Enum):
    """Status enumeration for agent reasoning workflows."""
    ANALYZING = "analyzing"  # Initial and processing state
    COMPLETED = "completed"
    FAILED = "failed"