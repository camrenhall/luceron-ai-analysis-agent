"""
Document Analysis Agent
AI-powered document intelligence orchestrator for Family Law financial discovery
"""

import os
import logging
import json
import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager
from enum import Enum

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import httpx
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.agents import AgentAction, AgentFinish

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")  # For batch analysis
BACKEND_URL = os.getenv("BACKEND_URL")
PORT = int(os.getenv("PORT", 8080))

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
if not BACKEND_URL:
    raise ValueError("BACKEND_URL environment variable is required")

# Global HTTP client
http_client = None

# Enhanced Workflow Models
class DocumentAnalysisStatus(str, Enum):
    PENDING_PLANNING = "PENDING_PLANNING"
    AWAITING_BATCH_COMPLETION = "AWAITING_BATCH_COMPLETION"
    SYNTHESIZING_RESULTS = "SYNTHESIZING_RESULTS"
    NEEDS_HUMAN_REVIEW = "NEEDS_HUMAN_REVIEW"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"

class AnalysisTask(BaseModel):
    task_id: int
    name: str
    document_ids: List[str]
    analysis_type: str
    status: str = "PENDING"  # PENDING, SUBMITTED, COMPLETED, FAILED
    depends_on: List[int] = []
    batch_job_id: Optional[str] = None
    results: Optional[Dict] = None
    confidence_score: Optional[int] = None

class TaskGraph(BaseModel):
    tasks: List[AnalysisTask]
    execution_plan: str

# Request Models
class TriggerDocumentAnalysisRequest(BaseModel):
    case_id: str
    document_ids: List[str]
    analysis_priority: str = "batch"  # "batch" or "immediate"
    case_context: Optional[str] = None

class DocumentAnalysisResponse(BaseModel):
    workflow_id: str
    status: DocumentAnalysisStatus
    message: str

class ChatRequest(BaseModel):
    message: str
    case_id: Optional[str] = None
    document_ids: Optional[List[str]] = None

# HTTP client management
async def init_http_client():
    """Initialize HTTP client"""
    global http_client
    http_client = httpx.AsyncClient(timeout=60.0)
    logger.info("HTTP client initialized")

async def close_http_client():
    """Close HTTP client"""
    global http_client
    if http_client:
        await http_client.aclose()
        logger.info("HTTP client closed")

# Backend API helpers
async def load_workflow_state(workflow_id: str) -> Optional[Dict]:
    """Load workflow state from backend"""
    try:
        response = await http_client.get(f"{BACKEND_URL}/api/workflows/{workflow_id}")
        if response.status_code == 404:
            return None
        response.raise_for_status()
        return response.json()
    except Exception as e:
        logger.error(f"Failed to load workflow state: {e}")
        return None

async def update_workflow_status(workflow_id: str, status: DocumentAnalysisStatus) -> None:
    """Update workflow status via backend"""
    try:
        response = await http_client.put(
            f"{BACKEND_URL}/api/workflows/{workflow_id}/status",
            json={"status": status.value}
        )
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to update workflow status: {e}")
        raise

async def add_reasoning_step(workflow_id: str, thought: str, action: str = None, 
                           action_input: Dict = None, action_output: str = None) -> None:
    """Add reasoning step to workflow"""
    try:
        step = {
            "timestamp": datetime.now().isoformat(),
            "thought": thought,
            "action": action,
            "action_input": action_input,
            "action_output": action_output
        }
        response = await http_client.post(
            f"{BACKEND_URL}/api/workflows/{workflow_id}/reasoning",
            json=step
        )
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to add reasoning step: {e}")

# Document Analysis Tools
class PlanAnalysisTasksTool(BaseTool):
    """Tool to create analysis task dependency graph"""
    name: str = "plan_analysis_tasks"
    description: str = "Create execution plan and task dependency graph for document analysis. Input: JSON with documents, case_context, priority"
    
    def _run(self, plan_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, plan_data: str) -> str:
        try:
            data = json.loads(plan_data)
            documents = data.get("documents", [])
            case_context = data.get("case_context", "")
            priority = data.get("priority", "batch")
            
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
                "task_graph": task_graph.dict(),
                "total_tasks": len(tasks),
                "priority": priority
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            logger.error(f"📋 Planning ERROR: {error_msg}")
            return json.dumps({"error": error_msg})

class SubmitBatchAnalysisTool(BaseTool):
    """Tool to submit documents to batch analysis API"""
    name: str = "submit_batch_analysis"
    description: str = "Submit documents to batch analysis API. Input: JSON with task_id, document_ids, analysis_type, case_context"
    
    def _run(self, batch_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, batch_data: str) -> str:
        try:
            data = json.loads(batch_data)
            task_id = data.get("task_id")
            document_ids = data.get("document_ids", [])
            analysis_type = data.get("analysis_type", "general")
            case_context = data.get("case_context", "")
            
            logger.info(f"🔄 Submitting batch analysis for task {task_id}")
            
            # Generate batch job ID (simulate API call)
            batch_job_id = f"batch_{uuid.uuid4().hex[:12]}"
            
            # Simulate batch API submission
            batch_payload = {
                "documents": document_ids,
                "analysis_instructions": f"Analyze documents for {analysis_type}. Context: {case_context}",
                "output_format": "structured_json",
                "include_confidence_scores": True
            }
            
            logger.info(f"🔄 Batch job {batch_job_id} submitted for {len(document_ids)} documents")
            logger.info(f"🔄 Analysis type: {analysis_type}")
            
            # In production, this would call OpenAI Batch API
            # response = await openai_client.post("/v1/batches", json=batch_payload)
            
            return json.dumps({
                "status": "batch_submitted",
                "batch_job_id": batch_job_id,
                "task_id": task_id,
                "documents_count": len(document_ids),
                "estimated_completion": "24 hours"
            })
            
        except Exception as e:
            error_msg = f"Batch submission failed: {str(e)}"
            logger.error(f"🔄 Batch ERROR: {error_msg}")
            return json.dumps({"error": error_msg})

class AnalyzeDocumentTool(BaseTool):
    """Tool for immediate document analysis"""
    name: str = "analyze_document_immediate"
    description: str = "Perform immediate document analysis for urgent cases. Input: JSON with document_id, analysis_type, case_context"
    
    def _run(self, analysis_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, analysis_data: str) -> str:
        try:
            data = json.loads(analysis_data)
            document_id = data.get("document_id")
            analysis_type = data.get("analysis_type", "general")
            case_context = data.get("case_context", "")
            
            logger.info(f"⚡ Immediate analysis of document {document_id}")
            
            # Simulate immediate analysis (in production, call OpenAI real-time API)
            analysis_result = {
                "document_id": document_id,
                "analysis_type": analysis_type,
                "extracted_data": {
                    "financial_summary": "Sample extracted financial data",
                    "key_figures": {"total_income": 75000, "total_expenses": 45000}
                },
                "confidence_score": 85,
                "confidence_justification": "Document quality is good. All key fields extracted successfully.",
                "flags": [],
                "processing_time_ms": 1500
            }
            
            logger.info(f"⚡ Immediate analysis completed with {analysis_result['confidence_score']}% confidence")
            
            return json.dumps({
                "status": "analysis_complete",
                "results": analysis_result
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Immediate analysis failed: {str(e)}"
            logger.error(f"⚡ Analysis ERROR: {error_msg}")
            return json.dumps({"error": error_msg})

class SynthesizeResultsTool(BaseTool):
    """Tool for validation and cross-document comparison"""
    name: str = "synthesize_results"
    description: str = "Validate analysis results or compare multiple documents. Input: JSON with synthesis_type, instructions, data_sources"
    
    def _run(self, synthesis_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, synthesis_data: str) -> str:
        try:
            data = json.loads(synthesis_data)
            synthesis_type = data.get("synthesis_type")  # "validation" or "comparison"
            instructions = data.get("instructions")
            data_sources = data.get("data_sources", [])
            
            logger.info(f"🔍 Synthesizing results: {synthesis_type}")
            
            # Simulate synthesis logic
            if synthesis_type == "validation":
                synthesis_result = {
                    "validation_type": "mathematical_consistency",
                    "status": "passed",
                    "findings": [
                        "All numerical columns sum correctly",
                        "No OCR errors detected in key fields"
                    ],
                    "confidence_score": 92,
                    "recommendations": ["Data appears accurate and ready for legal review"]
                }
            else:  # comparison
                synthesis_result = {
                    "comparison_type": "cross_document_validation",
                    "status": "discrepancy_found",
                    "findings": [
                        "Bank statement total: $75,000",
                        "Tax return reported income: $72,000",
                        "Discrepancy: $3,000 (4%)"
                    ],
                    "confidence_score": 88,
                    "recommendations": ["Minor discrepancy within acceptable range", "Flag for paralegal review"]
                }
            
            logger.info(f"🔍 Synthesis completed: {synthesis_result['status']}")
            
            return json.dumps({
                "status": "synthesis_complete",
                "results": synthesis_result
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Synthesis failed: {str(e)}"
            logger.error(f"🔍 Synthesis ERROR: {error_msg}")
            return json.dumps({"error": error_msg})

class GetCaseContextTool(BaseTool):
    """Tool to retrieve case context for informed analysis"""
    name: str = "get_case_context"
    description: str = "Retrieve case details and context. Input: case_id"
    
    def _run(self, case_id: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, case_id: str) -> str:
        try:
            logger.info(f"📁 Retrieving context for case {case_id}")
            
            # In production, call backend API for case details
            case_context = {
                "case_id": case_id,
                "case_type": "divorce_financial_discovery",
                "priority": "high_asset",
                "focus_areas": ["hidden_income", "asset_valuation", "expense_verification"],
                "known_concerns": ["Client suspects undisclosed business income"],
                "document_requirements": ["Bank statements", "Tax returns", "Business records"],
                "thresholds": {"discrepancy_alert": 50000, "review_required": 100000}
            }
            
            logger.info(f"📁 Retrieved context for {case_context['case_type']} case")
            
            return json.dumps(case_context, indent=2)
            
        except Exception as e:
            error_msg = f"Context retrieval failed: {str(e)}"
            logger.error(f"📁 Context ERROR: {error_msg}")
            return json.dumps({"error": error_msg})

# Workflow execution with state persistence
class DocumentAnalysisCallbackHandler(BaseCallbackHandler):
    """Callback handler for document analysis workflow state persistence"""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
    
    async def on_llm_start(self, serialized, prompts, **kwargs):
        await add_reasoning_step(self.workflow_id, "Agent analyzing document analysis strategy...")
    
    async def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get('name', 'Unknown tool')
        await add_reasoning_step(
            self.workflow_id,
            f"Executing {tool_name}",
            action=tool_name,
            action_input={"input": str(input_str)[:300]}
        )
    
    async def on_tool_end(self, output, **kwargs):
        await add_reasoning_step(
            self.workflow_id,
            "Tool execution completed",
            action_output=str(output)[:500]
        )
    
    async def on_agent_action(self, action: AgentAction, **kwargs):
        await add_reasoning_step(
            self.workflow_id,
            f"Agent decided: {action.tool}. Reasoning: {action.log[:300]}..."
        )

def create_document_analysis_agent(workflow_id: str) -> AgentExecutor:
    """Create document analysis agent with tools"""
    
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.1
    )
    
    tools = [
        PlanAnalysisTasksTool(),
        SubmitBatchAnalysisTool(),
        AnalyzeDocumentTool(),
        SynthesizeResultsTool(),
        GetCaseContextTool()
    ]
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Document Analysis Agent for a Family Law firm specializing in financial discovery.

CORE RESPONSIBILITY:
Orchestrate intelligent analysis of legal documents with focus on accuracy, auditability, and cost optimization.

WORKFLOW PHASES:
1. PLANNING: Always start by creating a task dependency graph using plan_analysis_tasks
2. EXECUTION: Submit tasks to batch processing or immediate analysis based on urgency
3. SYNTHESIS: Validate results and cross-reference documents for discrepancies
4. DECISION: Determine if results meet confidence thresholds or need human review

ANALYSIS PRINCIPLES:
- Batch processing for cost efficiency (50% savings) when time permits
- Immediate analysis for urgent legal documents (restraining orders, court deadlines)
- Always include confidence scores and justifications in analysis
- Cross-validate mathematical relationships between documents
- Flag discrepancies over case-specific thresholds for human review

QUALITY CONTROL:
- Validate OCR accuracy by checking mathematical consistency
- Compare related documents (bank statements vs tax returns)
- Identify potential hidden assets or income discrepancies
- Provide clear confidence scores and recommendations

LEGAL CONTEXT:
- Maintain complete audit trail of all analysis decisions
- Prioritize accuracy over speed for financial discovery
- Understand document relationships in divorce/family law context
- Apply case-specific thresholds and business rules

Execute systematically with focus on accuracy, cost efficiency, and legal compliance."""),
        MessagesPlaceholder("chat_history", optional=True),
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad")
    ])
    
    agent = create_tool_calling_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=20,
        return_intermediate_steps=True
    )

# FastAPI application
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_http_client()
    yield
    await close_http_client()

app = FastAPI(
    title="Document Analysis Agent",
    description="AI-powered document intelligence for Family Law financial discovery",
    version="1.0.0",
    lifespan=lifespan
)

@app.get("/")
async def health_check():
    """Service health check"""
    try:
        response = await http_client.get(f"{BACKEND_URL}/")
        response.raise_for_status()
        
        return {
            "status": "operational",
            "timestamp": datetime.now().isoformat(),
            "backend": "connected"
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Backend unavailable: {str(e)}")

@app.post("/workflows/trigger-analysis", response_model=DocumentAnalysisResponse)
async def trigger_document_analysis(request: TriggerDocumentAnalysisRequest, background_tasks: BackgroundTasks):
    """Trigger document analysis workflow"""
    
    workflow_id = f"wf_analysis_{uuid.uuid4().hex[:12]}"
    
    # Create workflow state in backend
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": DocumentAnalysisStatus.PENDING_PLANNING.value,
        "initial_prompt": f"Analyze documents for case {request.case_id}: {request.document_ids}. Priority: {request.analysis_priority}",
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    try:
        response = await http_client.post(f"{BACKEND_URL}/api/workflows", json=workflow_data)
        response.raise_for_status()
        
        # Trigger background processing
        background_tasks.add_task(
            execute_analysis_workflow, 
            workflow_id, 
            request.document_ids, 
            request.case_context or "",
            request.analysis_priority
        )
        
        return DocumentAnalysisResponse(
            workflow_id=workflow_id,
            status=DocumentAnalysisStatus.PENDING_PLANNING,
            message=f"Document analysis workflow {workflow_id} triggered"
        )
        
    except Exception as e:
        logger.error(f"Failed to trigger analysis workflow: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/workflows/{workflow_id}/status")
async def get_analysis_status(workflow_id: str):
    """Get document analysis workflow status"""
    state = await load_workflow_state(workflow_id)
    if not state:
        raise HTTPException(status_code=404, detail="Workflow not found")
    
    return state

@app.post("/webhooks/batch-complete")
async def batch_analysis_complete(batch_data: dict, background_tasks: BackgroundTasks):
    """Webhook for batch analysis completion"""
    
    batch_job_id = batch_data.get("batch_job_id")
    results = batch_data.get("results", {})
    
    logger.info(f"📥 Batch analysis complete: {batch_job_id}")
    
    # Find workflow by batch job ID (would need index in production)
    # For now, trigger synthesis phase for all awaiting workflows
    background_tasks.add_task(resume_synthesis_workflows, batch_job_id, results)
    
    return {"status": "webhook_processed", "batch_job_id": batch_job_id}

@app.post("/chat")
async def chat_with_analysis_agent(request: ChatRequest):
    """Interactive chat interface for document analysis"""
    
    workflow_id = f"wf_chat_{uuid.uuid4().hex[:8]}"
    
    # Create workflow for interactive session
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": DocumentAnalysisStatus.PENDING_PLANNING.value,
        "initial_prompt": request.message,
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat()
    }
    
    try:
        response = await http_client.post(f"{BACKEND_URL}/api/workflows", json=workflow_data)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create workflow: {e}")
    
    async def generate_stream():
        try:
            yield f"data: {json.dumps({'type': 'workflow_started', 'workflow_id': workflow_id})}\n\n"
            
            # Execute agent
            callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
            agent = create_document_analysis_agent(workflow_id)
            
            result = await agent.ainvoke(
                {"input": request.message},
                config={"callbacks": [callback_handler]}
            )
            
            # Update final status
            await update_workflow_status(workflow_id, DocumentAnalysisStatus.COMPLETED)
            
            # Stream final result
            output = result.get("output", "Analysis completed")
            yield f"data: {json.dumps({'type': 'final_response', 'response': output})}\n\n"
            yield f"data: {json.dumps({'type': 'workflow_complete', 'workflow_id': workflow_id})}\n\n"
            
        except Exception as e:
            await update_workflow_status(workflow_id, DocumentAnalysisStatus.FAILED)
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# Background task functions
async def execute_analysis_workflow(workflow_id: str, document_ids: List[str], case_context: str, priority: str):
    """Execute document analysis workflow in background"""
    try:
        await update_workflow_status(workflow_id, DocumentAnalysisStatus.PENDING_PLANNING)
        
        callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
        agent = create_document_analysis_agent(workflow_id)
        
        prompt = f"""Execute document analysis workflow:
        
Documents: {document_ids}
Case Context: {case_context}
Priority: {priority}

Start by creating an analysis plan, then execute based on priority level."""
        
        result = await agent.ainvoke(
            {"input": prompt},
            config={"callbacks": [callback_handler]}
        )
        
        await update_workflow_status(workflow_id, DocumentAnalysisStatus.COMPLETED)
        logger.info(f"✅ Analysis workflow {workflow_id} completed successfully")
        
    except Exception as e:
        await update_workflow_status(workflow_id, DocumentAnalysisStatus.FAILED)
        logger.error(f"❌ Analysis workflow {workflow_id} failed: {e}")

async def resume_synthesis_workflows(batch_job_id: str, results: Dict):
    """Resume workflows waiting for batch completion"""
    try:
        # In production, query workflows by batch_job_id
        logger.info(f"🔄 Resuming synthesis workflows for batch {batch_job_id}")
        
        # For now, simulate workflow resumption
        # This would involve loading workflows, updating task status, and continuing execution
        
    except Exception as e:
        logger.error(f"Failed to resume synthesis workflows: {e}")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Document Analysis Agent on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")