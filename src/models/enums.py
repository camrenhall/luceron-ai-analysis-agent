"""
Enumerations for the document analysis system.
"""

from enum import Enum


class WorkflowStatus(str, Enum):
    """Status enumeration for agent reasoning workflows."""
    PENDING = "pending"
    PROCESSING = "analyzing"  # Maps to backend's "analyzing" state
    AWAITING_SCHEDULE = "awaiting_schedule"
    COMPLETED = "completed"
    FAILED = "failed"
    PENDING_PLANNING = "pending_planning"
    AWAITING_BATCH_COMPLETION = "awaiting_batch_completion"
    SYNTHESIZING_RESULTS = "analyzing"  # Maps to backend's "analyzing" state
    NEEDS_HUMAN_REVIEW = "needs_human_review"