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
import boto3
import base64
from botocore.exceptions import NoCredentialsError, ClientError
from io import BytesIO

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
from openai import AsyncOpenAI

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Environment configuration
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL")
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY") 
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET_NAME = os.getenv("S3_BUCKET_NAME")
PORT = int(os.getenv("PORT", 8080))

if not ANTHROPIC_API_KEY:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is required")
if not BACKEND_URL:
    raise ValueError("BACKEND_URL environment variable is required")
if not AWS_ACCESS_KEY_ID or not AWS_SECRET_ACCESS_KEY:
    raise ValueError("AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY environment variables are required")
if not S3_BUCKET_NAME:
    raise ValueError("S3_BUCKET_NAME environment variable is required")

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

# System prompt loading
def load_system_prompt(filename: str) -> str:
    """Load system prompt from markdown file or return default"""
    try:
        with open(filename, 'r') as f:
            return f.read()
    except FileNotFoundError:
        logger.warning(f"System prompt file {filename} not found, using default")
        if 'document_analysis_system_prompt' in filename:
            return """You are an expert legal document analysis AI specializing in family law financial discovery.

Your role is to:
1. Extract financial data with precision and accuracy
2. Identify potential discrepancies or inconsistencies
3. Provide confidence scores for all extracted information
4. Flag any suspicious or unusual financial patterns
5. Cross-reference information across multiple documents

Always return structured JSON responses with:
- extracted_data: Key financial figures, dates, amounts
- confidence_score: Overall confidence (1-100)
- red_flags: Any concerning patterns or inconsistencies
- recommendations: Next steps or areas requiring attention

Be thorough, accurate, and maintain strict confidentiality."""
        else:
            return """You are a Document Analysis Agent for family law financial discovery cases.

You have access to these tools:
- plan_analysis_tasks: Create execution plans for document analysis
- analyze_documents_openai: Analyze documents using OpenAI (batch or immediate)
- check_batch_status: Check status of batch analysis jobs
- synthesize_results: Validate and cross-reference analysis results
- get_case_context: Retrieve case details and context

Your workflow:
1. Get case context to understand requirements
2. Plan analysis tasks based on document types and dependencies
3. Execute analysis (batch for cost efficiency, immediate for urgency)
4. Synthesize results and validate findings
5. Determine if human review is needed

Always maintain detailed reasoning for legal audit trails."""

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
        # Map DocumentAnalysisStatus to backend WorkflowStatus
        backend_status_map = {
            DocumentAnalysisStatus.PENDING_PLANNING: "PENDING",
            DocumentAnalysisStatus.AWAITING_BATCH_COMPLETION: "PROCESSING", 
            DocumentAnalysisStatus.SYNTHESIZING_RESULTS: "PROCESSING",
            DocumentAnalysisStatus.NEEDS_HUMAN_REVIEW: "PROCESSING",
            DocumentAnalysisStatus.COMPLETED: "COMPLETED",
            DocumentAnalysisStatus.FAILED: "FAILED"
        }
        
        backend_status = backend_status_map.get(status, "PROCESSING")
        
        response = await http_client.put(
            f"{BACKEND_URL}/api/workflows/{workflow_id}/status",
            json={"status": backend_status}
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
            f"{BACKEND_URL}/api/workflows/{workflow_id}/reasoning-step",
            json=step
        )
        response.raise_for_status()
    except Exception as e:
        logger.error(f"Failed to add reasoning step: {e}")
        
class DocumentAnalysisToolFactory:
    """Factory to create tools with injected dependencies"""
    
    def __init__(self):
        self.openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
            region_name=AWS_REGION
        )
        self.system_prompt = load_system_prompt('document_analysis_system_prompt.md')
        logger.info("Tool factory initialized with clients")
    
    def create_analysis_tool(self):
        """Create OpenAI analysis tool with injected dependencies"""
        
        class OpenAIDocumentAnalysisTool(BaseTool):
            name: str = "analyze_documents_openai"
            description: str = "Download and analyze documents using OpenAI o3. Input: JSON with document_ids, analysis_type, mode ('immediate'), case_context"
            
            def _run(self, analysis_data: str) -> str:
                raise NotImplementedError("Use async version")
            
            async def _arun(self, analysis_data: str) -> str:
                try:
                    data = json.loads(analysis_data)
                    document_ids = data.get("document_ids", [])
                    analysis_type = data.get("analysis_type", "comprehensive")
                    mode = data.get("mode", "immediate")
                    case_context = data.get("case_context", "")
                    
                    if mode != "immediate":
                        return json.dumps({"error": "Only immediate mode supported for MVP"})
                    
                    logger.info(f"⚡ Starting immediate analysis of {len(document_ids)} documents")
                    
                    results = []
                    for document_id in document_ids:
                        try:
                            # Get document metadata from backend
                            doc_metadata = await self._get_document_metadata(document_id)
                            
                            # Download document from S3
                            image_data = await self._download_document_from_s3(doc_metadata)
                            
                            # Analyze with o3
                            analysis_result = await self._analyze_with_o3(
                                document_id, image_data, doc_metadata, analysis_type, case_context
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
                        "mode": "immediate",
                        "results": results,
                        "document_count": len(results)
                    }, indent=2)
                    
                except Exception as e:
                    error_msg = f"Document analysis failed: {str(e)}"
                    logger.error(error_msg)
                    return json.dumps({"error": error_msg})
            
            async def _get_document_metadata(self, document_id: str) -> dict:
                """Get document metadata from backend"""
                response = await http_client.get(f"{BACKEND_URL}/api/documents/{document_id}")
                if response.status_code == 404:
                    raise ValueError(f"Document {document_id} not found")
                response.raise_for_status()
                return response.json()
            
            async def _download_document_from_s3(self, doc_metadata: dict) -> bytes:
                """Download document from S3 using factory's s3_client"""
                s3_key = doc_metadata.get("s3_key")
                if not s3_key:
                    raise ValueError("No S3 key found in document metadata")
                
                logger.info(f"📥 Downloading document from S3: {s3_key}")
                
                try:
                    # Access the factory's s3_client through closure
                    response = factory.s3_client.get_object(Bucket=S3_BUCKET_NAME, Key=s3_key)
                    image_data = response['Body'].read()
                    
                    logger.info(f"📥 Downloaded {len(image_data)} bytes from S3")
                    return image_data
                    
                except NoCredentialsError:
                    raise ValueError("AWS credentials not configured")
                except ClientError as e:
                    raise ValueError(f"S3 download failed: {e}")
            
            async def _analyze_with_o3(self, document_id: str, image_data: bytes, doc_metadata: dict, 
                                      analysis_type: str, case_context: str) -> dict:
                """Analyze document with OpenAI o3 using factory's client"""
                
                # Encode image as base64
                image_base64 = base64.b64encode(image_data).decode('utf-8')
                
                # Create analysis prompt
                analysis_prompt = f"""Analyze this financial document image for family law discovery purposes.

Document Details:
- Document ID: {document_id}
- Filename: {doc_metadata.get('filename', 'Unknown')}
- Document Type: {doc_metadata.get('document_type', 'other')}
- File Size: {doc_metadata.get('file_size', 0)} bytes

Case Context: {case_context}
Analysis Type: {analysis_type}

Please extract and analyze:
1. All financial figures, amounts, dates, and account information
2. Any patterns, trends, or anomalies in the financial data
3. Potential red flags or concerning information
4. Overall confidence in the document authenticity and completeness

Return a structured JSON response with:
{{
    "extracted_data": {{
        "amounts": [],
        "dates": [],
        "accounts": [],
        "key_figures": {{}}
    }},
    "confidence_score": 85,
    "red_flags": ["any concerning patterns"],
    "recommendations": "text recommendations",
    "summary": "brief summary of findings"
}}"""

                try:
                    logger.info(f"🧠 Sending document to o3 for analysis: {document_id}")
                    
                    # Use factory's openai_client through closure
                    response = await factory.openai_client.chat.completions.create(
                        model="o3",
                        messages=[
                            {"role": "system", "content": factory.system_prompt},
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": analysis_prompt},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{image_base64}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_completion_tokens=8000
                    )
                    
                    analysis_content = response.choices[0].message.content
                    usage = response.usage
                    
                    logger.info(f"🧠 o3 analysis completed for {document_id}")
                    
                    # Parse structured response if possible
                    extracted_data = None
                    confidence_score = None
                    red_flags = None
                    recommendations = None
                    
                    try:
                        # Try to extract JSON from response
                        import re
                        json_match = re.search(r'\{.*\}', analysis_content, re.DOTALL)
                        if json_match:
                            structured_data = json.loads(json_match.group())
                            extracted_data = structured_data.get("extracted_data")
                            confidence_score = structured_data.get("confidence_score")
                            red_flags = structured_data.get("red_flags")
                            recommendations = structured_data.get("recommendations")
                    except:
                        logger.warning("Could not parse structured JSON from o3 response")
                    
                    return {
                        "document_id": document_id,
                        "analysis_content": analysis_content,
                        "extracted_data": extracted_data,
                        "confidence_score": confidence_score,
                        "red_flags": red_flags,
                        "recommendations": recommendations,
                        "model_used": "o3",
                        "tokens_used": usage.total_tokens if usage else None,
                        "status": "completed"
                    }
                    
                except Exception as e:
                    logger.error(f"o3 analysis failed for {document_id}: {e}")
                    raise ValueError(f"o3 API analysis failed: {e}")
        
        # Create closure to capture factory reference
        factory = self
        return OpenAIDocumentAnalysisTool()

# Create global factory instance
tool_factory = DocumentAnalysisToolFactory()

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
                "task_graph": task_graph.dict(),
                "total_tasks": len(tasks),
                "priority": priority
            }, indent=2)
            
        except Exception as e:
            error_msg = f"Planning failed: {str(e)}"
            logger.error(f"📋 Planning ERROR: {error_msg}")
            return json.dumps({"error": error_msg})
        
class StoreAnalysisResultsTool(BaseTool):
    """Tool to store analysis results back to backend database"""
    name: str = "store_analysis_results"
    description: str = "Store document analysis results in database. Input: JSON with document_id, case_id, analysis_content, extracted_data, confidence_score, red_flags, recommendations"
    
    def _run(self, result_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, result_data: str) -> str:
        try:
            data = json.loads(result_data)
            document_id = data.get("document_id")
            
            logger.info(f"💾 Storing analysis results for document {document_id}")
            
            # Call backend API to store results
            response = await http_client.post(
                f"{BACKEND_URL}/api/documents/{document_id}/analysis",
                json=data
            )
            response.raise_for_status()
            
            result = response.json()
            analysis_id = result.get("analysis_id")
            
            logger.info(f"💾 Analysis results stored successfully: {analysis_id}")
            
            return json.dumps({
                "status": "stored",
                "analysis_id": analysis_id,
                "document_id": document_id
            })
            
        except Exception as e:
            error_msg = f"Failed to store analysis results: {str(e)}"
            logger.error(f"💾 Storage ERROR: {error_msg}")
            return json.dumps({"error": error_msg})

class GetCaseContextTool(BaseTool):
    """Tool to retrieve case context for informed analysis"""
    name: str = "get_case_context"
    description: str = "Retrieve case details and context. Input: case_id"
    
    def _run(self, case_id: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, case_id: str) -> str:
        try:
            logger.info(f"🔍 Retrieving context for case {case_id}")
            
            # Call backend API for case details
            response = await http_client.get(f"{BACKEND_URL}/api/cases/{case_id}")
            
            if response.status_code == 404:
                # Return basic context if case not found in detail
                case_context = {
                    "case_id": case_id,
                    "case_type": "family_law_financial_discovery",
                    "priority": "standard",
                    "focus_areas": ["asset_identification", "income_verification", "expense_analysis"],
                    "document_requirements": ["Financial statements", "Tax returns", "Bank records"]
                }
            else:
                response.raise_for_status()
                case_context = response.json()
            
            logger.info(f"🔍 Retrieved context for case {case_id}")
            
            return json.dumps(case_context, indent=2)
            
        except Exception as e:
            error_msg = f"Context retrieval failed: {str(e)}"
            logger.error(f"🔍 Context ERROR: {error_msg}")
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
    """Create document analysis agent using tool factory"""
    
    llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=ANTHROPIC_API_KEY,
        temperature=0.1
    )
    
    tools = [
        PlanAnalysisTasksTool(),
        tool_factory.create_analysis_tool(),  # Use factory method
        StoreAnalysisResultsTool(),
        GetCaseContextTool()
    ]
    
    system_prompt = load_system_prompt('document_analysis_agent_system_prompt.md')
    
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=system_prompt),
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
    
    # Create workflow state in backend - FIX: Use backend's WorkflowStatus enum values
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": "PENDING",  # Use backend's WorkflowStatus enum value instead of DocumentAnalysisStatus
        "initial_prompt": f"Analyze documents for case {request.case_id}: {request.document_ids}. Priority: {request.analysis_priority}"
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
    
    logger.info(f"🔥 Batch analysis complete: {batch_job_id}")
    
    # Find workflow by batch job ID and resume synthesis
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
        "document_ids": request.document_ids or [],
        "priority": "interactive"
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
        logger.info(f"📄 Resuming synthesis workflows for batch {batch_job_id}")
        
        # Find workflows containing this batch job ID
        # Since the backend endpoint doesn't exist, we'll search through workflow states
        response = await http_client.get(f"{BACKEND_URL}/api/workflows/pending")
        if response.status_code == 200:
            data = response.json()
            workflow_ids = data.get("workflow_ids", [])
            
            for workflow_id in workflow_ids:
                try:
                    # Get workflow state to check if it contains our batch job
                    workflow_state = await load_workflow_state(workflow_id)
                    if workflow_state and batch_job_id in workflow_state.get("batch_job_ids", []):
                        # Resume this workflow with synthesis phase
                        callback_handler = DocumentAnalysisCallbackHandler(workflow_id)
                        agent = create_document_analysis_agent(workflow_id)
                        
                        synthesis_prompt = f"""Batch analysis complete for job {batch_job_id}. Results: {json.dumps(results, indent=2)}
                        
Proceed with synthesis phase:
1. Validate the analysis results
2. Check for cross-document consistency
3. Generate final recommendations
4. Determine if human review is needed"""
                        
                        await update_workflow_status(workflow_id, DocumentAnalysisStatus.SYNTHESIZING_RESULTS)
                        
                        result = await agent.ainvoke(
                            {"input": synthesis_prompt},
                            config={"callbacks": [callback_handler]}
                        )
                        
                        await update_workflow_status(workflow_id, DocumentAnalysisStatus.COMPLETED)
                        logger.info(f"✅ Synthesis workflow {workflow_id} completed successfully")
                        
                except Exception as e:
                    await update_workflow_status(workflow_id, DocumentAnalysisStatus.FAILED)
                    logger.error(f"❌ Synthesis workflow {workflow_id} failed: {e}")
        
    except Exception as e:
        logger.error(f"Failed to resume synthesis workflows: {e}")

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting Document Analysis Agent on port {PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")