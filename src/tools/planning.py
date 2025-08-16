"""
Planning tools for document analysis workflow.
"""

import json
import logging
from langchain.tools import BaseTool

from models import AnalysisTask, TaskGraph

logger = logging.getLogger(__name__)


class PlanAnalysisTasksTool(BaseTool):
    """Tool to create analysis task dependency graph"""
    name: str = "plan_analysis_tasks"
    description: str = "Create execution plan and task dependency graph for document analysis. Input: JSON with documents, case_context"
    
    def _run(self, plan_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, plan_data: str) -> str:
        try:
            data = json.loads(plan_data)
            documents = data.get("documents", [])
            case_context = data.get("case_context", "")
            
            logger.info(f"📋 Planning analysis for {len(documents)} documents")
            
            # Create task graph based on document types and dependencies
            tasks = []
            task_id = 1
            
            # Group similar documents for parallel processing
            bank_statements = [doc for doc in documents if "bank" in doc.lower()]
            tax_documents = [doc for doc in documents if "tax" in doc.lower() or "w2" in doc.lower()]
            other_documents = [doc for doc in documents if doc not in bank_statements + tax_documents]
            
            # Create parallel tasks for bank statements
            if bank_statements:
                tasks.append(AnalysisTask(
                    task_id=task_id,
                    name="Analyze Bank Statements",
                    document_ids=bank_statements,
                    analysis_type="financial_extraction",
                    depends_on=[]
                ))
                bank_task_id = task_id
                task_id += 1
            
            # Create parallel tasks for tax documents
            if tax_documents:
                tasks.append(AnalysisTask(
                    task_id=task_id,
                    name="Analyze Tax Documents",
                    document_ids=tax_documents,
                    analysis_type="tax_extraction",
                    depends_on=[]
                ))
                tax_task_id = task_id
                task_id += 1
            
            # Process other documents
            if other_documents:
                tasks.append(AnalysisTask(
                    task_id=task_id,
                    name="Analyze Supporting Documents",
                    document_ids=other_documents,
                    analysis_type="general_extraction",
                    depends_on=[]
                ))
                task_id += 1
            
            # Create synthesis task that depends on primary analyses
            if len(tasks) > 1:
                depends_on = [t.task_id for t in tasks]
                tasks.append(AnalysisTask(
                    task_id=task_id,
                    name="Cross-Reference Financial Data",
                    document_ids=[],
                    analysis_type="cross_validation",
                    depends_on=depends_on
                ))
            
            # Create execution plan description
            execution_plan = f"Created {len(tasks)} analysis tasks. "
            if bank_statements and tax_documents:
                execution_plan += "Will analyze bank statements and tax documents in parallel, then cross-validate results."
            elif bank_statements:
                execution_plan += "Will analyze bank statements for financial extraction."
            else:
                execution_plan += "Will analyze provided documents sequentially."
            
            task_graph = TaskGraph(tasks=tasks, execution_plan=execution_plan)
            
            return json.dumps({
                "status": "plan_created",
                "task_graph": task_graph.model_dump(),
                "total_tasks": len(tasks)
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            logger.error(f"📋 Planning ERROR: {error_msg}")
            return json.dumps({"error": error_msg})