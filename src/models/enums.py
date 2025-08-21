"""
Enumerations for the document analysis system.
"""

from enum import Enum



class CaseStatus(str, Enum):
    """
    Case status enum aligned with backend API migration.
    
    - OPEN: Case is active and accepting documents/communications
    - CLOSED: Case has been completed or terminated
    """
    OPEN = "OPEN"
    CLOSED = "CLOSED"


