"""
Request and response schemas for the document analysis API.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    """Request model for interactive chat with the analysis agent."""
    message: str
    conversation_id: Optional[str] = None


class DateFilter(BaseModel):
    """Date filtering for case searches."""
    operator: str = Field(..., description="Comparison operator: eq, gte, lte, between")
    value: str = Field(..., description="ISO datetime string")
    end_value: Optional[str] = Field(None, description="End value for 'between' operator")


class CaseSearchQuery(BaseModel):
    """Request model for case search with flexible filtering."""
    client_name: Optional[str] = Field(None, description="Client name search term")
    client_email: Optional[str] = Field(None, description="Client email search term")
    client_phone: Optional[str] = Field(None, description="Client phone search term")
    status: Optional[str] = Field(None, description="Case status filter")
    use_fuzzy_matching: bool = Field(False, description="Enable fuzzy/similarity matching")
    fuzzy_threshold: float = Field(0.3, description="Fuzzy matching similarity threshold (0.0-1.0)")
    created_at: Optional[DateFilter] = Field(None, description="Case creation date filter")
    last_communication_date: Optional[DateFilter] = Field(None, description="Last communication date filter")
    limit: int = Field(50, description="Maximum results to return", ge=1, le=500)
    offset: int = Field(0, description="Offset for pagination", ge=0)


class CaseSearchResult(BaseModel):
    """Individual case result from search."""
    case_id: str
    client_name: str
    client_email: Optional[str] = None
    client_phone: Optional[str] = None
    status: str
    created_at: str
    last_communication_date: Optional[str] = None
    similarity_score: Optional[float] = Field(None, description="Fuzzy matching similarity score")


class CaseSearchResponse(BaseModel):
    """Response model for case search results."""
    total_count: int
    cases: List[CaseSearchResult]
    limit: int
    offset: int
    search_query: CaseSearchQuery


