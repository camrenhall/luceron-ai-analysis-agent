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


# Backward compatibility alias
WorkflowStatus = Status