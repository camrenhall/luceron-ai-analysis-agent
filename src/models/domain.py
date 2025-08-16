"""
Domain models for the document analysis system.
"""

from typing import Dict, List, Optional
from pydantic import BaseModel


class AnalysisTask(BaseModel):
    """Model representing a single analysis task within a workflow."""
    task_id: int
    name: str
    document_ids: List[str]
    analysis_type: str
    status: str = "PENDING"  # PENDING, SUBMITTED, COMPLETED, FAILED
    depends_on: List[int] = []
    results: Optional[Dict] = None


class TaskGraph(BaseModel):
    """Model representing a complete task execution graph."""
    tasks: List[AnalysisTask]
    execution_plan: str