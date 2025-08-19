"""
Request and response schemas for the document analysis API.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class ChatRequest(BaseModel):
    """Request model for interactive chat with the analysis agent."""
    message: str
    case_id: str


