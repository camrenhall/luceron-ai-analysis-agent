"""
Enumerations for the document analysis system.
"""

from enum import Enum


class Status(str, Enum):
    """
    Unified status enum for all processing entities in the system.
    
    - PENDING: Initial state, entity created but not yet being processed
    - PROCESSING: Entity is currently being worked on
    - COMPLETED: Processing finished successfully
    - FAILED: Processing encountered an error and stopped
    """
    PENDING = "PENDING"
    PROCESSING = "PROCESSING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class CaseStatus(str, Enum):
    """
    Case status enum aligned with backend API migration.
    
    - OPEN: Case is active and accepting documents/communications
    - CLOSED: Case has been completed or terminated
    """
    OPEN = "OPEN"
    CLOSED = "CLOSED"


# Backward compatibility alias
WorkflowStatus = Status