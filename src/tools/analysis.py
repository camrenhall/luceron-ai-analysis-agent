"""
Document analysis tools using OpenAI.
"""

import json
import logging
from langchain.tools import BaseTool

from services import backend_api_service, s3_service, openai_service

logger = logging.getLogger(__name__)


class OpenAIDocumentAnalysisTool(BaseTool):
    """Tool for analyzing documents using OpenAI o3"""
    name: str = "analyze_documents_openai"
    description: str = "Download and analyze documents using OpenAI o3. Input: JSON with document_ids, analysis_type, case_context, workflow_id"
    
    def _run(self, analysis_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, analysis_data: str) -> str:
        try:
            data = json.loads(analysis_data)
            document_ids = data.get("document_ids", [])
            analysis_type = data.get("analysis_type", "comprehensive")
            case_context = data.get("case_context", "")
            workflow_id = data.get("workflow_id")
            
            logger.info(f"⚡ Starting analysis of {len(document_ids)} documents")
            
            results = []
            for document_id in document_ids:
                try:
                    # Get document metadata from backend
                    doc_metadata = await backend_api_service.get_document_metadata(document_id)
                    
                    # Download document from S3
                    s3_key = doc_metadata.get("s3_key")
                    if not s3_key:
                        raise ValueError("No S3 key found in document metadata")
                    
                    image_data = await s3_service.download_document(s3_key)
                    
                    # Analyze with OpenAI o3
                    analysis_result = await openai_service.analyze_document(
                        document_id, image_data, doc_metadata, analysis_type, case_context, workflow_id
                    )
                    
                    results.append(analysis_result)
                    
                except Exception as e:
                    logger.error(f"Failed to analyze document {document_id}: {e}")
                    results.append({
                        "document_id": document_id,
                        "error": str(e),
                        "status": "failed"
                    })
            
            logger.info(f"⚡ Analysis completed for {len(results)} documents")
            
            return json.dumps({
                "status": "analysis_complete",
                "results": results,
                "document_count": len(results)
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Document analysis failed: {str(e)}"
            logger.error(error_msg)
            return json.dumps({"error": error_msg})