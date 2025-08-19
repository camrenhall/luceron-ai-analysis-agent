"""
Tool for evaluating document satisfaction and managing document completion status.
"""

import json
import logging
import os
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from langchain.tools import BaseTool

from services import backend_api_service

logger = logging.getLogger(__name__)


class DocumentSatisfactionTool(BaseTool):
    """Tool to evaluate if documents satisfy requirements and mark them as completed"""
    name: str = "evaluate_document_satisfaction"
    description: str = """Evaluate if submitted documents satisfy requested document requirements and mark them as completed. 
    Input: JSON with case_id, document_analysis_results (list of analyzed documents), and optional specific_requirements.
    This tool applies document satisfaction criteria and can mark multiple documents as completed in batch."""
    
    def __init__(self):
        super().__init__()
        self.criteria_file_path = "/Users/camrenhall/Documents/blueprint-venture-capital-llc/principal-development-files/luceron-ai-analysis-agent/document_satisfaction_criteria.md"
        self._load_satisfaction_criteria()
    
    def _load_satisfaction_criteria(self) -> str:
        """Load the document satisfaction criteria from markdown file"""
        try:
            if os.path.exists(self.criteria_file_path):
                with open(self.criteria_file_path, 'r') as f:
                    return f.read()
            else:
                logger.warning(f"Satisfaction criteria file not found at {self.criteria_file_path}")
                return ""
        except Exception as e:
            logger.error(f"Failed to load satisfaction criteria: {e}")
            return ""
    
    def _run(self, input_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, input_data: str) -> str:
        try:
            # Parse input data
            try:
                data = json.loads(input_data)
            except json.JSONDecodeError:
                return json.dumps({"error": "Invalid JSON input"})
            
            case_id = data.get("case_id")
            document_analysis_results = data.get("document_analysis_results", [])
            specific_requirements = data.get("specific_requirements", [])
            
            if not case_id:
                return json.dumps({"error": "case_id is required"})
            
            logger.info(f"🔍 Evaluating document satisfaction for case {case_id}")
            
            # Get requested documents for the case
            case_data = await backend_api_service.get_requested_documents(case_id)
            if not case_data:
                return json.dumps({"error": f"Could not retrieve case data for {case_id}"})
            
            requested_documents = case_data.get("requested_documents", [])
            
            # Evaluate satisfaction for each requested document
            satisfaction_results = []
            documents_to_complete = []
            
            for requested_doc in requested_documents:
                if requested_doc.get("is_completed", False):
                    continue  # Skip already completed documents
                
                satisfaction_result = await self._evaluate_single_document(
                    requested_doc, document_analysis_results, specific_requirements
                )
                satisfaction_results.append(satisfaction_result)
                
                # If satisfied, mark for completion
                if satisfaction_result["is_satisfied"]:
                    documents_to_complete.append({
                        "requested_doc_id": requested_doc["requested_doc_id"],
                        "satisfaction_reason": satisfaction_result["reason"],
                        "matched_documents": satisfaction_result["matched_documents"]
                    })
            
            # Update document statuses in backend
            completion_results = []
            for doc_to_complete in documents_to_complete:
                result = await backend_api_service.update_document_status(
                    requested_doc_id=doc_to_complete["requested_doc_id"],
                    is_completed=True,
                    notes=f"Auto-completed: {doc_to_complete['satisfaction_reason']}"
                )
                completion_results.append({
                    "requested_doc_id": doc_to_complete["requested_doc_id"],
                    "updated": result is not None,
                    "reason": doc_to_complete["satisfaction_reason"]
                })
            
            # Prepare final response
            response = {
                "case_id": case_id,
                "evaluation_timestamp": datetime.now().isoformat(),
                "total_requested_documents": len(requested_documents),
                "documents_evaluated": len(satisfaction_results),
                "documents_satisfied": len(documents_to_complete),
                "satisfaction_results": satisfaction_results,
                "completion_results": completion_results,
                "summary": {
                    "newly_completed": len([r for r in completion_results if r["updated"]]),
                    "failed_to_update": len([r for r in completion_results if not r["updated"]]),
                    "still_pending": len([r for r in satisfaction_results if not r["is_satisfied"]])
                }
            }
            
            logger.info(f"✅ Completed satisfaction evaluation for case {case_id}. {len(documents_to_complete)} documents satisfied.")
            return json.dumps(response, indent=2)
            
        except Exception as e:
            error_msg = f"Document satisfaction evaluation failed: {str(e)}"
            logger.error(f"❌ {error_msg}")
            return json.dumps({"error": error_msg})
    
    async def _evaluate_single_document(
        self, 
        requested_doc: Dict, 
        analysis_results: List[Dict], 
        specific_requirements: List[str]
    ) -> Dict:
        """Evaluate if a single requested document is satisfied by analysis results"""
        
        requested_name = requested_doc.get("document_name", "").lower()
        requested_description = requested_doc.get("description", "").lower()
        
        satisfaction_result = {
            "requested_doc_id": requested_doc["requested_doc_id"],
            "requested_document_name": requested_doc.get("document_name"),
            "requested_description": requested_doc.get("description"),
            "is_satisfied": False,
            "reason": "",
            "matched_documents": [],
            "evaluation_details": []
        }
        
        # Check each analyzed document against this requirement
        for analysis in analysis_results:
            evaluation = self._check_document_match(
                requested_name, requested_description, analysis
            )
            satisfaction_result["evaluation_details"].append(evaluation)
            
            if evaluation["matches"]:
                satisfaction_result["is_satisfied"] = True
                satisfaction_result["reason"] = evaluation["match_reason"]
                satisfaction_result["matched_documents"].append({
                    "document_id": analysis.get("document_id"),
                    "analysis_id": analysis.get("analysis_id"),
                    "match_confidence": evaluation["confidence"]
                })
        
        # If not satisfied, provide reason
        if not satisfaction_result["is_satisfied"]:
            satisfaction_result["reason"] = self._generate_non_satisfaction_reason(
                requested_name, requested_description, analysis_results
            )
        
        return satisfaction_result
    
    def _check_document_match(self, requested_name: str, requested_description: str, analysis: Dict) -> Dict:
        """Check if an analyzed document matches a requested document"""
        
        evaluation = {
            "document_id": analysis.get("document_id"),
            "matches": False,
            "match_reason": "",
            "confidence": 0.0,
            "criteria_applied": []
        }
        
        # Extract document type and metadata from analysis
        analysis_content = analysis.get("analysis_content", {})
        if isinstance(analysis_content, str):
            try:
                analysis_content = json.loads(analysis_content)
            except:
                analysis_content = {}
        
        document_type = self._extract_document_type(analysis_content)
        document_year = self._extract_document_year(analysis_content)
        document_metadata = self._extract_document_metadata(analysis_content)
        
        # Apply satisfaction criteria
        
        # 1. Document Type Matching
        if self._check_document_type_match(requested_name, document_type):
            evaluation["criteria_applied"].append("document_type_match")
            evaluation["confidence"] += 0.4
            
            # 2. Year/Date Specificity
            if self._check_year_match(requested_name, requested_description, document_year):
                evaluation["criteria_applied"].append("year_match")
                evaluation["confidence"] += 0.3
                
                # 3. Completeness check
                if self._check_completeness(requested_name, requested_description, analysis_content):
                    evaluation["criteria_applied"].append("completeness_check")
                    evaluation["confidence"] += 0.3
                    
                    # Document satisfies requirements
                    if evaluation["confidence"] >= 0.8:
                        evaluation["matches"] = True
                        evaluation["match_reason"] = f"Document type '{document_type}' matches requested '{requested_name}'"
                        if document_year:
                            evaluation["match_reason"] += f" for year {document_year}"
        
        return evaluation
    
    def _extract_document_type(self, analysis_content: Dict) -> str:
        """Extract document type from analysis content"""
        
        # Look for document type in various fields
        type_fields = ["document_type", "type", "category", "document_category"]
        for field in type_fields:
            if field in analysis_content:
                return str(analysis_content[field]).lower()
        
        # Extract from key findings or summary
        key_findings = analysis_content.get("key_findings", [])
        if isinstance(key_findings, list):
            for finding in key_findings:
                if isinstance(finding, str):
                    # Look for common document type patterns
                    if "w-2" in finding.lower() or "w2" in finding.lower():
                        return "w-2"
                    elif "1099" in finding.lower():
                        return "1099"
                    elif "tax return" in finding.lower():
                        return "tax return"
                    elif "bank statement" in finding.lower():
                        return "bank statement"
                    elif "paystub" in finding.lower():
                        return "paystub"
        
        return ""
    
    def _extract_document_year(self, analysis_content: Dict) -> Optional[str]:
        """Extract year from document analysis"""
        
        # Look for year in specific fields
        year_fields = ["year", "tax_year", "statement_year", "period"]
        for field in year_fields:
            if field in analysis_content:
                year_value = analysis_content[field]
                if isinstance(year_value, (int, str)):
                    year_str = str(year_value)
                    if re.match(r'^\d{4}$', year_str):
                        return year_str
        
        # Extract year from dates
        dates = analysis_content.get("dates", [])
        if isinstance(dates, list):
            for date in dates:
                year_match = re.search(r'\b(20\d{2})\b', str(date))
                if year_match:
                    return year_match.group(1)
        
        return None
    
    def _extract_document_metadata(self, analysis_content: Dict) -> Dict:
        """Extract additional metadata from document analysis"""
        return {
            "entities": analysis_content.get("entities", []),
            "dates": analysis_content.get("dates", []),
            "amounts": analysis_content.get("amounts", []),
            "confidence": analysis_content.get("confidence", {})
        }
    
    def _check_document_type_match(self, requested_name: str, document_type: str) -> bool:
        """Check if document types match according to satisfaction criteria"""
        
        if not document_type:
            return False
        
        # Normalize names for comparison
        requested_normalized = requested_name.lower().strip()
        document_normalized = document_type.lower().strip()
        
        # Direct match
        if requested_normalized == document_normalized:
            return True
        
        # Common variations and synonyms
        type_mappings = {
            "w2": ["w-2", "w 2", "form w-2"],
            "w-2": ["w2", "w 2", "form w-2"],
            "1099": ["1099-misc", "1099-int", "1099-div", "form 1099"],
            "tax return": ["federal tax return", "form 1040", "1040"],
            "federal tax return": ["tax return", "form 1040", "1040"],
            "bank statement": ["statement", "checking statement", "savings statement"],
            "paystub": ["pay stub", "paycheck stub", "earnings statement"],
            "employment verification": ["letter of employment", "employment letter"]
        }
        
        # Check if requested type maps to document type
        for key, variations in type_mappings.items():
            if key in requested_normalized:
                if any(var in document_normalized for var in variations):
                    return True
                if document_normalized in variations:
                    return True
        
        # Partial match for complex names
        if len(requested_normalized) > 5 and requested_normalized in document_normalized:
            return True
        if len(document_normalized) > 5 and document_normalized in requested_normalized:
            return True
        
        return False
    
    def _check_year_match(self, requested_name: str, requested_description: str, document_year: Optional[str]) -> bool:
        """Check if years match when specified"""
        
        # Extract year from requested document name/description
        requested_text = f"{requested_name} {requested_description}".lower()
        year_match = re.search(r'\b(20\d{2})\b', requested_text)
        
        if year_match:
            requested_year = year_match.group(1)
            # If year is specified in request, it must match
            return document_year == requested_year
        
        # If no year specified in request, any year is acceptable
        return True
    
    def _check_completeness(self, requested_name: str, requested_description: str, analysis_content: Dict) -> bool:
        """Check if document appears complete based on analysis"""
        
        # Look for completeness indicators in analysis
        completeness = analysis_content.get("completeness", {})
        if isinstance(completeness, dict):
            complete_value = completeness.get("complete", completeness.get("is_complete"))
            if complete_value is not None:
                return bool(complete_value)
        
        # Check for red flags indicating incompleteness
        red_flags = analysis_content.get("red_flags", [])
        if isinstance(red_flags, list):
            incomplete_indicators = ["partial", "incomplete", "missing pages", "truncated"]
            for flag in red_flags:
                if any(indicator in str(flag).lower() for indicator in incomplete_indicators):
                    return False
        
        # Default to complete if no indicators found
        return True
    
    def _generate_non_satisfaction_reason(self, requested_name: str, requested_description: str, analysis_results: List[Dict]) -> str:
        """Generate reason why document requirement was not satisfied"""
        
        if not analysis_results:
            return "No analyzed documents available to satisfy this requirement"
        
        # Analyze why documents didn't match
        reasons = []
        document_types_found = []
        
        for analysis in analysis_results:
            analysis_content = analysis.get("analysis_content", {})
            if isinstance(analysis_content, str):
                try:
                    analysis_content = json.loads(analysis_content)
                except:
                    analysis_content = {}
            
            doc_type = self._extract_document_type(analysis_content)
            if doc_type:
                document_types_found.append(doc_type)
        
        if document_types_found:
            unique_types = list(set(document_types_found))
            reasons.append(f"Document types found: {', '.join(unique_types)}")
            reasons.append(f"Required document type: {requested_name}")
        else:
            reasons.append("Could not determine document types from analysis")
        
        return "; ".join(reasons)