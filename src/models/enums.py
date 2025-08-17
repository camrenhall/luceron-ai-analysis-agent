"""
Enumerations for the document analysis system.
"""

from enum import Enum


class WorkflowStatus(str, Enum):
    """Status enumeration for agent reasoning workflows."""
    PENDING = "PENDING"
    REASONING = "REASONING"
    EVALUATING = "EVALUATING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"