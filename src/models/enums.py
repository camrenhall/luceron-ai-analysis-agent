"""
Enumerations for the document analysis system.
"""

from enum import Enum


class DocumentAnalysisStatus(str, Enum):
    """Status enumeration for document analysis workflows."""
    PENDING_PLANNING = "PENDING_PLANNING"
    SYNTHESIZING_RESULTS = "SYNTHESIZING_RESULTS"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"