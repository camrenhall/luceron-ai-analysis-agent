"""
Background task execution for document analysis workflows.
"""

import logging
from typing import List

from ..models import DocumentAnalysisStatus
from ..services import backend_api_service
from ..agents import DocumentAnalysisCallbackHandler, create_document_analysis_agent
from ..utils import load_prompt_template

logger = logging.getLogger(__name__)


async def execute_analysis_workflow(
    workflow_id: str, 
    case_id: str, 
    document_ids: List[str], 
    case_context: str
):
    """Execute document analysis workflow in background"""
    try:
        await backend_api_service.update_workflow_status(
            workflow_id, 
            DocumentAnalysisStatus.PENDING_PLANNING
        )
        
        callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
        agent = create_document_analysis_agent(workflow_id)
        
        # Load workflow prompt template and format with variables
        prompt_template = load_prompt_template('workflow_execution_prompt.md')
        prompt = prompt_template.format(
            workflow_id=workflow_id,
            case_id=case_id,
            document_ids=document_ids,
            case_context=case_context
        )
        
        result = await agent.ainvoke(
            {"input": prompt},
            config={"callbacks": [callback_handler]}
        )
        
        await backend_api_service.update_workflow_status(
            workflow_id, 
            DocumentAnalysisStatus.COMPLETED
        )
        logger.info(f"✅ Analysis workflow {workflow_id} completed successfully")
        
    except Exception as e:
        await backend_api_service.update_workflow_status(
            workflow_id, 
            DocumentAnalysisStatus.FAILED
        )
        logger.error(f"❌ Analysis workflow {workflow_id} failed: {e}")