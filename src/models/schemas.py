"""
Request and response schemas for the document analysis API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request model for interactive chat with the analysis agent."""
    message: str
    case_id: str


class AWSAnalysisResult(BaseModel):
    """Model for receiving analysis results from AWS Lambda."""
    workflow_id: str
    case_id: str
    document_ids: List[str]
    analysis_data: Dict[str, Any]  # Structured JSON from AWS processing
    metadata: Optional[Dict[str, Any]] = None