"""
Consolidated case search tool for intelligent case discovery with dynamic endpoint selection.
"""

import re
import logging
from typing import Dict, Any, Optional, Type
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from services import backend_api_service

logger = logging.getLogger(__name__)


class CaseSearchInput(BaseModel):
    """Input schema for consolidated case search tool."""
    search_term: str = Field(..., description="Name, email, phone number, or case ID to search for")
    use_fuzzy_matching: bool = Field(True, description="Enable fuzzy matching for typos and variations")
    limit: int = Field(10, description="Maximum number of results to return")


class ConsolidatedCaseSearchTool(BaseTool):
    """
    Intelligent case search tool that automatically detects search type and uses optimal backend methods.
    
    This tool replaces multiple specialized search tools with one that dynamically determines:
    - Search type (name, email, phone, case_id)
    - Appropriate backend API method to call
    - Progressive matching strategy (exact -> fuzzy -> partial)
    """
    
    name: str = "search_cases"
    description: str = """
    Universal case search tool that automatically detects search type and finds relevant cases.
    
    Automatically handles:
    - Client names (with fuzzy matching for typos)
    - Email addresses (exact and partial matching)
    - Phone numbers (various formats)
    - Case IDs
    
    Uses progressive search strategy: exact match -> fuzzy match -> broader search.
    
    Examples:
    - search_cases("John Smith") - Find cases for John Smith with typo tolerance
    - search_cases("john@gmail.com") - Find cases by exact email match
    - search_cases("555-123-4567") - Find cases by phone number
    - search_cases("test_case_001") - Find case by ID
    
    Returns comprehensive case information with analysis recommendations.
    """
    args_schema: Type[BaseModel] = CaseSearchInput

    def _detect_search_type(self, search_term: str) -> str:
        """Auto-detect the type of search term provided with enhanced logic."""
        search_term = search_term.strip()
        
        # Case ID pattern detection (common patterns: test_case_001, CASE-123, etc.)
        case_id_patterns = [
            r'^[a-zA-Z]+_[a-zA-Z0-9_]+_\d+$',  # test_case_001
            r'^CASE-\d+$',  # CASE-123
            r'^[A-Z]{2,4}-\d{4,6}$',  # ABC-12345
        ]
        
        for pattern in case_id_patterns:
            if re.match(pattern, search_term):
                return "case_id"
        
        # Email pattern detection
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, search_term):
            return "email"
        
        # Phone pattern detection (various formats)
        phone_patterns = [
            r'^\+?1?[\s-.]?\(?(\d{3})\)?[\s-.]?(\d{3})[\s-.]?(\d{4})$',  # Various US formats
            r'^\d{10}$',  # 5551234567
            r'^\d{3}[\s.-]\d{3}[\s.-]\d{4}$',  # 555-123-4567, 555.123.4567, 555 123 4567
        ]
        
        for pattern in phone_patterns:
            if re.match(pattern, search_term):
                return "phone"
        
        # Partial phone (for broader searching)
        if search_term.isdigit() and len(search_term) >= 3:
            return "phone"
        
        # Default to name search
        return "name"

    def _format_search_results(self, results: Dict[str, Any], search_term: str, search_type: str) -> str:
        """Format search results for agent consumption with enhanced recommendations."""
        if not results or not results.get("cases"):
            return f"No cases found for {search_type} search: '{search_term}'"
        
        cases = results["cases"]
        total_count = results.get("total_count", len(cases))
        
        # Determine search quality
        search_quality = self._assess_search_quality(results)
        
        # Build formatted response
        response_lines = [
            f"Found {len(cases)} case(s) matching '{search_term}' (total: {total_count})",
            f"Search quality: {search_quality}",
            ""
        ]
        
        for i, case in enumerate(cases, 1):
            case_info = [
                f"{i}. {case['client_name']} (Case ID: {case['case_id']})",
                f"   Status: {case['status']}",
                f"   Created: {case['created_at']}"
            ]
            
            # Add contact information if available
            if case.get('client_email'):
                case_info.append(f"   Email: {case['client_email']}")
            if case.get('client_phone'):
                case_info.append(f"   Phone: {case['client_phone']}")
            if case.get('last_communication_date'):
                case_info.append(f"   Last Contact: {case['last_communication_date']}")
            
            # Add similarity score if this was a fuzzy match
            if case.get('similarity_score') is not None:
                case_info.append(f"   Match Score: {case['similarity_score']:.2f}")
            
            response_lines.extend(case_info)
            response_lines.append("")  # Empty line between cases
        
        # Add intelligent recommendations
        recommendations = self._generate_recommendations(cases, search_type, search_quality)
        response_lines.extend(recommendations)
        
        return "\n".join(response_lines)

    def _assess_search_quality(self, results: Dict[str, Any]) -> str:
        """Assess the quality of search results."""
        cases = results.get("cases", [])
        if not cases:
            return "No matches"
        
        # Check if we have high-confidence exact matches
        high_confidence_matches = [c for c in cases if c.get('similarity_score', 1.0) > 0.9]
        if high_confidence_matches:
            return "Exact match"
        
        # Check if we have decent fuzzy matches  
        decent_matches = [c for c in cases if c.get('similarity_score', 1.0) > 0.6]
        if decent_matches:
            return "Good match"
        
        # Otherwise it's a broad/fuzzy search
        return "Fuzzy match"

    def _generate_recommendations(self, cases: list, search_type: str, search_quality: str) -> list:
        """Generate intelligent next-step recommendations."""
        recommendations = []
        
        if len(cases) == 1:
            case_id = cases[0]['case_id']
            recommendations.extend([
                "💡 Single case found:",
                f"   • Case ID: {case_id}",
                f"   • Use this case ID for further operations in other systems"
            ])
        
        elif len(cases) > 1:
            if search_quality == "Fuzzy match":
                recommendations.extend([
                    "💡 Multiple fuzzy matches found:",
                    "   • Review cases above and select the correct one",
                    "   • Try a more specific search term if needed",
                    "   • Use case ID for exact identification once selected"
                ])
            else:
                recommendations.extend([
                    "💡 Multiple cases found:",
                    "   • Review cases above and select which one to analyze",
                    "   • All appear to be valid matches for your search"
                ])
        
        return recommendations

    async def _arun(self, search_term: str, use_fuzzy_matching: bool = True, limit: int = 10) -> str:
        """Execute intelligent case search with dynamic method selection."""
        try:
            logger.info(f"🔍 Universal case search: '{search_term}' (fuzzy: {use_fuzzy_matching})")
            
            # Auto-detect search type
            search_type = self._detect_search_type(search_term)
            logger.info(f"🎯 Detected search type: {search_type}")
            
            # Execute progressive search using the most appropriate backend method
            if search_type == "case_id":
                # For case IDs, try direct lookup first, then fallback to general search
                try:
                    # Attempt to get case context directly (this validates case existence)
                    case_context = await backend_api_service.get_case_context(search_term)
                    if case_context and case_context.get('case_id'):
                        # Convert to standard search result format
                        mock_result = {
                            "cases": [{
                                "case_id": case_context['case_id'],
                                "client_name": case_context.get('client_name', 'Unknown'),
                                "status": case_context.get('status', 'Unknown'),
                                "created_at": case_context.get('created_at', 'Unknown'),
                                "similarity_score": 1.0  # Perfect match for case ID
                            }],
                            "total_count": 1
                        }
                        results = mock_result
                    else:
                        # Fallback to progressive search
                        results = await backend_api_service.progressive_case_search(search_term, "name")
                except Exception:
                    # Case ID didn't work, try as general search
                    results = await backend_api_service.progressive_case_search(search_term, "name")
            else:
                # Use progressive search for all other types
                results = await backend_api_service.progressive_case_search(search_term, search_type)
            
            # Use best match results from progressive search
            best_results = results.get("best_match", {"cases": [], "total_count": 0})
            
            # Log search analytics
            case_count = len(best_results.get("cases", []))
            total_found = best_results.get("total_count", 0)
            logger.info(f"✅ Search completed: {case_count}/{total_found} cases for '{search_term}' (type: {search_type})")
            
            # Format and return results
            return self._format_search_results(best_results, search_term, search_type)
            
        except Exception as e:
            error_msg = f"❌ Case search failed for '{search_term}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _run(self, search_term: str, use_fuzzy_matching: bool = True, limit: int = 10) -> str:
        """Synchronous version (not implemented for async backend)."""
        raise NotImplementedError("This tool requires async execution. Use _arun method.")


# Create tool instance
consolidated_case_search_tool = ConsolidatedCaseSearchTool()