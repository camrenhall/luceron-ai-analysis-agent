"""
Enumerations for the document analysis system.
"""

from enum import Enum


class WorkflowStatus(str, Enum):
    """Status enumeration for agent reasoning workflows."""
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"  # Maps to our internal "REASONING" state
    AWAITING_SCHEDULE = "AWAITING_SCHEDULE"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    PENDING_PLANNING = "PENDING_PLANNING"
    AWAITING_BATCH_COMPLETION = "AWAITING_BATCH_COMPLETION"
    SYNTHESIZING_RESULTS = "SYNTHESIZING_RESULTS"  # Maps to our internal "EVALUATING" state
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"