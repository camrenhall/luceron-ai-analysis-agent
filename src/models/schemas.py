"""
Request and response schemas for the document analysis API.
"""

from typing import List, Optional
from pydantic import BaseModel

from .enums import DocumentAnalysisStatus


class TriggerDocumentAnalysisRequest(BaseModel):
    """Request model for triggering document analysis workflow."""
    case_id: str
    document_ids: List[str]
    case_context: Optional[str] = None
    workflow_id: Optional[str] = None


class DocumentAnalysisResponse(BaseModel):
    """Response model for document analysis operations."""
    workflow_id: str
    status: DocumentAnalysisStatus
    message: str


class ChatRequest(BaseModel):
    """Request model for interactive chat with the analysis agent."""
    message: str
    case_id: Optional[str] = None
    document_ids: Optional[List[str]] = None