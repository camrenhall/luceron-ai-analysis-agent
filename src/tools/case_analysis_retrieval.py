"""
Case analysis retrieval tool for comprehensive document review.
This tool enables the Analysis Agent to act as a senior partner reviewing all analyzed documents.
"""

import json
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from langchain.tools import BaseTool

from services import http_client_service
from config import settings

logger = logging.getLogger(__name__)


class GetAllCaseAnalysesTool(BaseTool):
    """
    Tool to retrieve ALL document analyses for a case.
    Designed to support senior-partner-level review of comprehensive case documentation.
    """
    name: str = "get_all_case_analyses"
    description: str = """Retrieve ALL analyzed documents for a case. This provides comprehensive access to all document analyses, 
    enabling pattern recognition, contradiction detection, and complete financial picture assessment. 
    Input: case_id (string). Output: Structured analysis with all documents, patterns, and insights."""
    
    def _run(self, case_id: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, case_id: str) -> str:
        """
        Retrieve and process all document analyses for comprehensive case review.
        
        This method:
        1. Fetches all analyses from backend
        2. Processes and structures the data
        3. Identifies patterns and key findings
        4. Prepares data for senior-partner-level reasoning
        """
        try:
            if not case_id:
                return json.dumps({"error": "case_id is required"})
            
            logger.info(f"📊 Retrieving all analyses for case {case_id}")
            
            # Call backend API to get all analyses
            response = await http_client_service.client.get(
                f"{settings.BACKEND_URL}/api/documents/analysis/case/{case_id}",
                params={"include_content": "true"}
            )
            
            if response.status_code == 404:
                logger.warning(f"No analyses found for case {case_id}")
                return json.dumps({
                    "case_id": case_id,
                    "status": "no_analyses_found",
                    "message": "No document analyses found for this case"
                })
            
            response.raise_for_status()
            data = response.json()
            
            # Process and structure the analyses for optimal reasoning
            processed_analyses = self._process_analyses(data)
            
            # Extract patterns and insights across all documents
            cross_document_insights = self._extract_cross_document_patterns(processed_analyses)
            
            # Build comprehensive response
            result = {
                "case_id": case_id,
                "retrieval_timestamp": datetime.now().isoformat(),
                "summary": {
                    "total_documents_analyzed": data.get("total_analyses", 0),
                    "total_tokens_used": data.get("total_tokens_used", 0),
                    "analysis_status": "complete"
                },
                "document_analyses": processed_analyses,
                "cross_document_insights": cross_document_insights,
                "metadata": {
                    "newest_analysis": self._get_newest_analysis_date(data),
                    "oldest_analysis": self._get_oldest_analysis_date(data),
                    "models_used": self._get_unique_models(data)
                }
            }
            
            logger.info(f"✅ Successfully retrieved {len(processed_analyses)} analyses for case {case_id}")
            
            return json.dumps(result, indent=2)
            
        except Exception as e:
            error_msg = f"Failed to retrieve case analyses: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})
    
    def _process_analyses(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Process raw analyses into structured format for reasoning.
        Extracts key information and organizes it for pattern recognition.
        """
        processed = []
        
        for analysis in data.get("analyses", []):
            # Parse analysis_content if it's a string
            content = analysis.get("analysis_content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except json.JSONDecodeError:
                    content = {"raw_text": content}
            
            processed_analysis = {
                "document_id": analysis.get("document_id"),
                "analysis_id": analysis.get("analysis_id"),
                "analyzed_at": analysis.get("analyzed_at"),
                "key_findings": self._extract_key_findings(content),
                "financial_data": self._extract_financial_data(content),
                "red_flags": self._extract_red_flags(content),
                "entities_mentioned": self._extract_entities(content),
                "dates_referenced": self._extract_dates(content),
                "confidence_indicators": self._extract_confidence(content),
                "raw_content": content  # Keep raw for detailed inspection if needed
            }
            processed.append(processed_analysis)
        
        return processed
    
    def _extract_key_findings(self, content: Dict[str, Any]) -> List[str]:
        """Extract key findings from analysis content."""
        findings = []
        
        # Look for various keys that might contain findings
        finding_keys = ["key_findings", "findings", "summary", "conclusions", "observations"]
        for key in finding_keys:
            if key in content:
                value = content[key]
                if isinstance(value, list):
                    findings.extend(value)
                elif isinstance(value, str):
                    findings.append(value)
                elif isinstance(value, dict):
                    findings.append(json.dumps(value))
        
        # Also extract from nested structures
        if "analysis" in content and isinstance(content["analysis"], dict):
            nested_findings = self._extract_key_findings(content["analysis"])
            findings.extend(nested_findings)
        
        return findings
    
    def _extract_financial_data(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract financial figures and amounts from analysis."""
        financial_data = {
            "amounts": [],
            "accounts": [],
            "transactions": [],
            "balances": {}
        }
        
        # Look for financial keys
        financial_keys = ["amounts", "financial_figures", "transactions", "balances", 
                         "income", "expenses", "assets", "liabilities", "accounts"]
        
        for key in financial_keys:
            if key in content:
                value = content[key]
                if isinstance(value, (int, float)):
                    financial_data["amounts"].append({key: value})
                elif isinstance(value, dict):
                    financial_data["balances"].update({key: value})
                elif isinstance(value, list):
                    if key == "accounts":
                        financial_data["accounts"].extend(value)
                    elif key == "transactions":
                        financial_data["transactions"].extend(value)
                    else:
                        financial_data["amounts"].extend(value)
        
        return financial_data
    
    def _extract_red_flags(self, content: Dict[str, Any]) -> List[str]:
        """Extract red flags and concerns from analysis."""
        red_flags = []
        
        # Look for red flag indicators
        flag_keys = ["red_flags", "concerns", "warnings", "issues", "discrepancies", 
                    "anomalies", "suspicious", "irregular"]
        
        for key in flag_keys:
            if key in content:
                value = content[key]
                if isinstance(value, list):
                    red_flags.extend(value)
                elif isinstance(value, str):
                    red_flags.append(value)
                elif isinstance(value, dict):
                    red_flags.append(f"{key}: {json.dumps(value)}")
        
        return red_flags
    
    def _extract_entities(self, content: Dict[str, Any]) -> List[str]:
        """Extract mentioned entities (people, companies, institutions)."""
        entities = []
        
        entity_keys = ["entities", "names", "parties", "institutions", "companies", "banks"]
        for key in entity_keys:
            if key in content and isinstance(content[key], list):
                entities.extend(content[key])
        
        return list(set(entities))  # Remove duplicates
    
    def _extract_dates(self, content: Dict[str, Any]) -> List[str]:
        """Extract important dates from analysis."""
        dates = []
        
        date_keys = ["dates", "date_range", "period", "transaction_dates", "statement_date"]
        for key in date_keys:
            if key in content:
                value = content[key]
                if isinstance(value, list):
                    dates.extend(value)
                elif isinstance(value, str):
                    dates.append(value)
        
        return dates
    
    def _extract_confidence(self, content: Dict[str, Any]) -> Dict[str, Any]:
        """Extract confidence indicators about document authenticity and completeness."""
        confidence = {
            "authenticity": None,
            "completeness": None,
            "reliability": None
        }
        
        confidence_keys = ["confidence", "authenticity", "completeness", "reliability", "quality"]
        for key in confidence_keys:
            if key in content:
                if key in confidence:
                    confidence[key] = content[key]
                else:
                    confidence[key] = content[key]
        
        return confidence
    
    def _extract_cross_document_patterns(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Identify patterns and relationships across all documents.
        This is where the senior partner insight comes in.
        """
        patterns = {
            "recurring_entities": self._find_recurring_entities(analyses),
            "financial_timeline": self._build_financial_timeline(analyses),
            "red_flag_patterns": self._identify_red_flag_patterns(analyses),
            "document_gaps": self._identify_document_gaps(analyses),
            "inconsistencies": self._detect_inconsistencies(analyses),
            "trend_analysis": self._analyze_trends(analyses)
        }
        
        return patterns
    
    def _find_recurring_entities(self, analyses: List[Dict[str, Any]]) -> Dict[str, int]:
        """Find entities that appear across multiple documents."""
        entity_counts = {}
        for analysis in analyses:
            for entity in analysis.get("entities_mentioned", []):
                entity_counts[entity] = entity_counts.get(entity, 0) + 1
        
        # Return only entities that appear in multiple documents
        return {k: v for k, v in entity_counts.items() if v > 1}
    
    def _build_financial_timeline(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Build a timeline of financial events across all documents."""
        timeline = []
        
        for analysis in analyses:
            dates = analysis.get("dates_referenced", [])
            financial_data = analysis.get("financial_data", {})
            
            for date in dates:
                timeline.append({
                    "date": date,
                    "document_id": analysis.get("document_id"),
                    "financial_data": financial_data
                })
        
        # Sort by date if possible
        return sorted(timeline, key=lambda x: x.get("date", ""), reverse=True)
    
    def _identify_red_flag_patterns(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Identify patterns in red flags across documents."""
        all_red_flags = []
        documents_with_red_flags = 0
        
        for analysis in analyses:
            flags = analysis.get("red_flags", [])
            if flags:
                documents_with_red_flags += 1
                all_red_flags.extend(flags)
        
        return {
            "total_red_flags": len(all_red_flags),
            "documents_with_red_flags": documents_with_red_flags,
            "percentage_flagged": (documents_with_red_flags / len(analyses) * 100) if analyses else 0,
            "common_issues": list(set(all_red_flags))  # Unique issues
        }
    
    def _identify_document_gaps(self, analyses: List[Dict[str, Any]]) -> List[str]:
        """Identify potential gaps in documentation."""
        gaps = []
        
        # Check for date gaps
        all_dates = []
        for analysis in analyses:
            all_dates.extend(analysis.get("dates_referenced", []))
        
        # This is simplified - in production, would do more sophisticated gap analysis
        if not all_dates:
            gaps.append("No dated documents found")
        
        return gaps
    
    def _detect_inconsistencies(self, analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect inconsistencies across documents."""
        inconsistencies = []
        
        # Compare financial data across documents for inconsistencies
        # This is a simplified version - real implementation would be more sophisticated
        financial_summaries = {}
        for analysis in analyses:
            doc_id = analysis.get("document_id")
            financial_data = analysis.get("financial_data", {})
            if financial_data.get("balances"):
                financial_summaries[doc_id] = financial_data["balances"]
        
        # Would implement actual inconsistency detection logic here
        
        return inconsistencies
    
    def _analyze_trends(self, analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze trends across documents."""
        return {
            "document_count_trend": len(analyses),
            "analysis_period": {
                "start": self._get_oldest_analysis_date({"analyses": analyses}),
                "end": self._get_newest_analysis_date({"analyses": analyses})
            }
        }
    
    def _get_newest_analysis_date(self, data: Dict[str, Any]) -> Optional[str]:
        """Get the date of the newest analysis."""
        analyses = data.get("analyses", [])
        if analyses:
            return analyses[0].get("analyzed_at")  # Already sorted by analyzed_at DESC
        return None
    
    def _get_oldest_analysis_date(self, data: Dict[str, Any]) -> Optional[str]:
        """Get the date of the oldest analysis."""
        analyses = data.get("analyses", [])
        if analyses:
            return analyses[-1].get("analyzed_at")
        return None
    
    def _get_unique_models(self, data: Dict[str, Any]) -> List[str]:
        """Get list of unique models used in analyses."""
        models = set()
        for analysis in data.get("analyses", []):
            model = analysis.get("model_used")
            if model:
                models.add(model)
        return list(models)