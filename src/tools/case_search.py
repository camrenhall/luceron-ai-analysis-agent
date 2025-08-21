"""
Case search and discovery tool for intelligent case lookup by name, email, or phone.
"""

import re
import logging
from typing import Dict, Any, List, Optional
from langchain.tools import BaseTool
from pydantic import BaseModel, Field

from services import backend_api_service

logger = logging.getLogger(__name__)


class CaseSearchInput(BaseModel):
    """Input schema for case search tool."""
    search_term: str = Field(..., description="Name, email, or phone number to search for")
    search_type: Optional[str] = Field("auto", description="Search type: 'name', 'email', 'phone', or 'auto' to detect")
    use_fuzzy_matching: bool = Field(True, description="Enable fuzzy matching for typos and variations")
    limit: int = Field(10, description="Maximum number of results to return")


class CaseSearchTool(BaseTool):
    """Tool for searching and discovering cases by client information."""
    
    name: str = "search_cases"
    description: str = """
    Search for cases by client name, email address, or phone number.
    
    This tool automatically detects the search type and uses progressive matching:
    1. Exact matching first
    2. Fuzzy matching for names (handles typos, variations)
    3. Partial matching for broader results
    
    Examples:
    - search_cases(search_term="John Smith") - Find cases for John Smith
    - search_cases(search_term="john@gmail.com") - Find cases by email
    - search_cases(search_term="555-1234") - Find cases by phone number
    - search_cases(search_term="Jon Smth", use_fuzzy_matching=True) - Handle typos
    
    Returns detailed case information including case_id, client details, status, and creation date.
    """
    args_schema = CaseSearchInput

    def _detect_search_type(self, search_term: str) -> str:
        """Auto-detect the type of search term provided."""
        search_term = search_term.strip()
        
        # Email pattern detection
        email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if re.match(email_pattern, search_term):
            return "email"
        
        # Phone pattern detection (various formats)
        phone_patterns = [
            r'^\d{10}$',  # 5551234567
            r'^\d{3}-\d{3}-\d{4}$',  # 555-123-4567
            r'^\(\d{3}\)\s?\d{3}-\d{4}$',  # (555) 123-4567
            r'^\+1\d{10}$',  # +15551234567
            r'^\d{3}\.\d{3}\.\d{4}$',  # 555.123.4567
            r'^\d{3}\s\d{3}\s\d{4}$',  # 555 123 4567
        ]
        
        for pattern in phone_patterns:
            if re.match(pattern, search_term):
                return "phone"
        
        # If it contains digits but doesn't match phone patterns, could be partial phone
        if any(c.isdigit() for c in search_term) and len(search_term) >= 3:
            return "phone"
        
        # Default to name search
        return "name"

    def _format_search_results(self, results: Dict[str, Any], search_term: str, search_type: str) -> str:
        """Format search results for agent consumption."""
        if not results or not results.get("cases"):
            return f"No cases found for {search_type} search: '{search_term}'"
        
        cases = results["cases"]
        total_count = results.get("total_count", len(cases))
        
        # Build formatted response
        response_lines = [
            f"Found {len(cases)} case(s) matching '{search_term}' (total: {total_count}):",
            ""
        ]
        
        for i, case in enumerate(cases, 1):
            case_info = [
                f"{i}. {case['client_name']} (Case ID: {case['case_id']})",
                f"   Status: {case['status']}",
                f"   Created: {case['created_at']}"
            ]
            
            if case.get('client_email'):
                case_info.append(f"   Email: {case['client_email']}")
            
            if case.get('client_phone'):
                case_info.append(f"   Phone: {case['client_phone']}")
            
            if case.get('last_communication_date'):
                case_info.append(f"   Last Contact: {case['last_communication_date']}")
            
            if case.get('similarity_score') is not None:
                case_info.append(f"   Match Score: {case['similarity_score']:.2f}")
            
            response_lines.extend(case_info)
            response_lines.append("")  # Empty line between cases
        
        # Add recommendation for next steps
        if len(cases) == 1:
            case_id = cases[0]['case_id']
            response_lines.extend([
                f"💡 To analyze documents for this case, use: get_all_case_analyses(case_id='{case_id}')",
                f"💡 To see requested documents, use: get_requested_documents(case_id='{case_id}')"
            ])
        elif len(cases) > 1:
            response_lines.append("💡 Multiple cases found. Please specify which case you'd like to analyze by using the Case ID.")
        
        return "\n".join(response_lines)

    async def _arun(self, search_term: str, search_type: str = "auto", 
                   use_fuzzy_matching: bool = True, limit: int = 10) -> str:
        """Execute case search with intelligent matching."""
        try:
            logger.info(f"🔍 Case search requested: '{search_term}' (type: {search_type})")
            
            # Auto-detect search type if needed
            if search_type == "auto":
                search_type = self._detect_search_type(search_term)
                logger.info(f"🎯 Auto-detected search type: {search_type}")
            
            # Execute progressive search
            results = await backend_api_service.progressive_case_search(
                search_term=search_term, 
                search_type=search_type
            )
            
            # Use the best match from progressive search
            best_results = results.get("best_match", {"cases": [], "total_count": 0})
            
            # Format results for agent
            formatted_response = self._format_search_results(best_results, search_term, search_type)
            
            # Log search success
            case_count = len(best_results.get("cases", []))
            logger.info(f"✅ Case search completed: {case_count} cases found for '{search_term}'")
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"❌ Case search failed for '{search_term}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _run(self, search_term: str, search_type: str = "auto", 
            use_fuzzy_matching: bool = True, limit: int = 10) -> str:
        """Synchronous version (not implemented for async backend)."""
        raise NotImplementedError("This tool requires async execution. Use _arun method.")


# Convenience functions for different search types
class CaseSearchByNameTool(BaseTool):
    """Specialized tool for searching cases by client name only."""
    
    name: str = "search_cases_by_name"
    description: str = """
    Search for cases specifically by client name with fuzzy matching support.
    
    Handles typos, name variations, and different name orders.
    Examples:
    - search_cases_by_name("John Smith")
    - search_cases_by_name("Smith, John") 
    - search_cases_by_name("Jon Smth") - handles typos
    """
    args_schema = CaseSearchInput

    async def _arun(self, search_term: str, use_fuzzy_matching: bool = True, limit: int = 10, **kwargs) -> str:
        """Execute name-specific case search."""
        try:
            logger.info(f"👤 Name search: '{search_term}' (fuzzy: {use_fuzzy_matching})")
            
            results = await backend_api_service.search_cases_by_name(
                name=search_term,
                use_fuzzy=use_fuzzy_matching,
                limit=limit
            )
            
            # Format results
            formatted_response = CaseSearchTool()._format_search_results(results, search_term, "name")
            
            case_count = len(results.get("cases", []))
            logger.info(f"✅ Name search completed: {case_count} cases found")
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"❌ Name search failed for '{search_term}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("This tool requires async execution. Use _arun method.")


class CaseSearchByEmailTool(BaseTool):
    """Specialized tool for searching cases by email address."""
    
    name: str = "search_cases_by_email"
    description: str = """
    Search for cases by client email address.
    
    Supports exact and partial email matching.
    Examples:
    - search_cases_by_email("john@gmail.com")
    - search_cases_by_email("@company.com") - find all cases from a domain
    """
    args_schema = CaseSearchInput

    async def _arun(self, search_term: str, limit: int = 10, **kwargs) -> str:
        """Execute email-specific case search."""
        try:
            logger.info(f"📧 Email search: '{search_term}'")
            
            results = await backend_api_service.search_cases_by_email(
                email=search_term,
                limit=limit
            )
            
            formatted_response = CaseSearchTool()._format_search_results(results, search_term, "email")
            
            case_count = len(results.get("cases", []))
            logger.info(f"✅ Email search completed: {case_count} cases found")
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"❌ Email search failed for '{search_term}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("This tool requires async execution. Use _arun method.")


class CaseSearchByPhoneTool(BaseTool):
    """Specialized tool for searching cases by phone number."""
    
    name: str = "search_cases_by_phone"
    description: str = """
    Search for cases by client phone number.
    
    Supports various phone formats and partial matching.
    Examples:
    - search_cases_by_phone("555-123-4567")
    - search_cases_by_phone("555") - find all cases with 555 area code
    - search_cases_by_phone("1234") - partial number search
    """
    args_schema = CaseSearchInput

    async def _arun(self, search_term: str, limit: int = 10, **kwargs) -> str:
        """Execute phone-specific case search."""
        try:
            logger.info(f"📞 Phone search: '{search_term}'")
            
            results = await backend_api_service.search_cases_by_phone(
                phone=search_term,
                limit=limit
            )
            
            formatted_response = CaseSearchTool()._format_search_results(results, search_term, "phone")
            
            case_count = len(results.get("cases", []))
            logger.info(f"✅ Phone search completed: {case_count} cases found")
            
            return formatted_response
            
        except Exception as e:
            error_msg = f"❌ Phone search failed for '{search_term}': {str(e)}"
            logger.error(error_msg)
            return error_msg

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("This tool requires async execution. Use _arun method.")


# Create tool instances
case_search_tool = CaseSearchTool()
case_search_by_name_tool = CaseSearchByNameTool()
case_search_by_email_tool = CaseSearchByEmailTool()
case_search_by_phone_tool = CaseSearchByPhoneTool()