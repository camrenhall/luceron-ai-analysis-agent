# Analysis Findings Templates

This document contains structured analysis findings that the Analysis Agent sends to the Communications Agent. The Communications Agent handles all client messaging based on these analytical findings.

## Document Analysis Finding Types

### Document Type Mismatch
```
ANALYSIS FINDING: Document Type Mismatch
Case ID: {case_id}
Expected Document: {document_type_required}
Received Document: {document_type_received} 
Document ID: {document_id}
Confidence: {confidence_level}
Analysis Details: The uploaded document appears to be a {document_type_received}, but the case requires a {document_type_required}. Document analysis shows {analysis_details}.
Recommendation: Request correct document type from client.
```

### Year/Date Mismatch
```
ANALYSIS FINDING: Year/Date Mismatch
Case ID: {case_id}
Expected Year: {year_required}
Received Year: {year_received}
Document Type: {document_type}
Document ID: {document_id}
Analysis Details: Document analysis indicates this is a {document_type} from {year_received}, but case requires {year_required}. Year verification based on {verification_method}.
Recommendation: Request document for correct year from client.
```

### Duplicate Document Detection
```
ANALYSIS FINDING: Duplicate Document Detected
Case ID: {case_id}
Document Type: {document_type}
Original Document ID: {original_document_id}
Duplicate Document ID: {duplicate_document_id}
Analysis Details: Content analysis indicates this document is substantially similar to previously uploaded document. Similarity score: {similarity_score}.
Still Needed: {remaining_document_types}
Recommendation: Acknowledge receipt and request remaining documents.
```

### Missing Document Requirements
```
ANALYSIS FINDING: Missing Required Documents
Case ID: {case_id}
Completion Status: {completed_count}/{total_required} documents satisfied
Missing Documents: {missing_document_list}
Analysis Details: Document satisfaction analysis shows {completed_count} requirements satisfied. Outstanding requirements: {detailed_missing_analysis}.
Priority: {priority_level}
Recommendation: Request missing documents from client.
```

### Document Quality Issues
```
ANALYSIS FINDING: Document Quality Issues
Case ID: {case_id}
Document Type: {document_type}
Document ID: {document_id}
Quality Issues: {quality_issues_list}
Analysis Details: Document analysis detected quality concerns: {detailed_quality_analysis}. Impact on case: {impact_assessment}.
Recommendation: Request higher quality version of document.
```

### Document Incompleteness
```
ANALYSIS FINDING: Incomplete Document
Case ID: {case_id}
Document Type: {document_type}
Document ID: {document_id}
Expected Content: {expected_content_description}
Found Content: {actual_content_description}
Analysis Details: Document appears to be missing {missing_content}. Completeness assessment: {completeness_percentage}%.
Recommendation: Request complete version of document.
```

### Document Acceptance
```
ANALYSIS FINDING: Document Requirements Satisfied
Case ID: {case_id}
Document Type: {document_type}
Document ID: {document_id}
Satisfaction Details: Document successfully satisfies requirement for {requirement_description}. Analysis confidence: {confidence_level}.
Remaining Requirements: {remaining_count} documents still needed: {remaining_list}
Recommendation: Acknowledge receipt and continue with remaining requirements if any.
```

### Case Completion Status
```
ANALYSIS FINDING: Case Document Status Update
Case ID: {case_id}
Total Requirements: {total_required}
Satisfied: {satisfied_count}
Pending: {pending_count}
Completion Percentage: {completion_percentage}%
Analysis Details: {detailed_status_analysis}
Next Steps: {recommended_next_steps}
```

## Template Variables Reference

### Document Information
- `{case_id}` - Case identifier
- `{document_id}` - Specific document identifier
- `{document_type}` - Type of document (W-2, Tax Return, etc.)
- `{document_type_required}` - Expected document type
- `{document_type_received}` - Actual document type received

### Temporal Information
- `{year_required}` - Required year for document
- `{year_received}` - Year of received document
- `{period_required}` - Required time period
- `{period_received}` - Time period of received document

### Analysis Results
- `{confidence_level}` - Analysis confidence (High/Medium/Low)
- `{analysis_details}` - Detailed analysis findings
- `{verification_method}` - How year/type was determined
- `{similarity_score}` - Duplicate detection score
- `{quality_issues_list}` - List of quality problems
- `{impact_assessment}` - How issues affect the case

### Status Information
- `{completed_count}` - Number of requirements satisfied
- `{total_required}` - Total requirements for case
- `{missing_document_list}` - List of missing documents
- `{remaining_document_types}` - Types of documents still needed
- `{completion_percentage}` - Percentage of requirements satisfied

### Content Assessment
- `{expected_content_description}` - What should be in the document
- `{actual_content_description}` - What was actually found
- `{missing_content}` - What appears to be missing
- `{completeness_percentage}` - How complete the document appears

## Usage Guidelines

1. **Factual Analysis**: Focus on analytical findings, not client communication
2. **Structured Format**: Use consistent template structure for reliable parsing
3. **Specific Details**: Include specific document IDs, confidence levels, and analysis details
4. **Clear Recommendations**: Provide clear next step recommendations for the Communications Agent
5. **Separation of Concerns**: Analysis Agent analyzes, Communications Agent communicates