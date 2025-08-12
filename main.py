"""
Production Document Analysis Agent
Full production implementation with OpenAI Responses API, authentication, monitoring
"""

import os
import logging
import json
import asyncio
import uuid
import structlog
from datetime import datetime
from typing import Dict, List, Optional, Any
from contextlib import asynccontextmanager

import httpx
import jwt
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import SystemMessage
from langchain_core.callbacks import BaseCallbackHandler

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Environment configuration with production defaults
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
BACKEND_URL = os.getenv("BACKEND_URL")
JWT_SECRET = os.getenv("JWT_SECRET", "your-secret-key-change-in-production")
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
PORT = int(os.getenv("PORT", 8080))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Validate required environment variables
required_vars = {
    "ANTHROPIC_API_KEY": ANTHROPIC_API_KEY,
    "OPENAI_API_KEY": OPENAI_API_KEY,
    "BACKEND_URL": BACKEND_URL
}

for var_name, var_value in required_vars.items():
    if not var_value:
        raise ValueError(f"{var_name} environment variable is required")

# Prometheus metrics
REQUEST_COUNT = Counter('http_requests_total', 'Total HTTP requests', ['method', 'endpoint', 'status'])
REQUEST_LATENCY = Histogram('http_request_duration_seconds', 'HTTP request latency')
WORKFLOW_COUNT = Counter('workflows_total', 'Total workflows', ['agent_type', 'status'])
BATCH_JOB_COUNT = Counter('batch_jobs_total', 'Total batch jobs', ['status'])

# Global HTTP clients
http_client = None
openai_client = None

# Security
security = HTTPBearer()

# Enhanced Models
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
    status: str = "PENDING"
    depends_on: List[int] = []
    batch_job_id: Optional[str] = None
    results: Optional[Dict] = None
    confidence_score: Optional[int] = None
    error_message: Optional[str] = None
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)

class TaskGraph(BaseModel):
    tasks: List[AnalysisTask]
    execution_plan: str
    total_estimated_cost: Optional[float] = None
    estimated_completion_time: Optional[str] = None

class DocumentAnalysisRequest(BaseModel):
    case_id: str
    document_ids: List[str]
    analysis_priority: str = Field(default="batch", regex="^(batch|immediate)$")
    case_context: Optional[str] = None
    custom_instructions: Optional[str] = None

class DocumentAnalysisResponse(BaseModel):
    workflow_id: str
    status: DocumentAnalysisStatus
    message: str
    estimated_completion: Optional[str] = None
    cost_estimate: Optional[float] = None

# Authentication utilities
def verify_jwt_token(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict:
    """Verify JWT token and return user data"""
    try:
        token = credentials.credentials
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token has expired")
    except jwt.JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def create_access_token(data: Dict) -> str:
    """Create JWT access token for testing"""
    return jwt.encode(data, JWT_SECRET, algorithm=JWT_ALGORITHM)

# HTTP client management
async def init_http_clients():
    """Initialize HTTP clients with production configuration"""
    global http_client, openai_client
    
    # Backend client
    http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(30.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        retries=3
    )
    
    # OpenAI client  
    openai_client = httpx.AsyncClient(
        base_url="https://api.openai.com/v1",
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json"
        },
        timeout=httpx.Timeout(60.0, connect=10.0),
        limits=httpx.Limits(max_keepalive_connections=10, max_connections=50)
    )
    
    logger.info("HTTP clients initialized", 
                backend_url=BACKEND_URL,
                openai_configured=bool(OPENAI_API_KEY))

async def close_http_clients():
    """Close HTTP clients"""
    global http_client, openai_client
    if http_client:
        await http_client.aclose()
    if openai_client:
        await openai_client.aclose()
    logger.info("HTTP clients closed")

# Production-ready Document Analysis Tools
class PlanAnalysisTasksTool(BaseTool):
    """Production tool to create analysis task dependency graph"""
    name: str = "plan_analysis_tasks"
    description: str = "Create execution plan and task dependency graph for document analysis"
    
    def _run(self, plan_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, plan_data: str) -> str:
        try:
            data = json.loads(plan_data)
            documents = data.get("documents", [])
            case_context = data.get("case_context", "")
            priority = data.get("priority", "batch")
            custom_instructions = data.get("custom_instructions", "")
            
            logger.info("Planning document analysis", 
                       document_count=len(documents),
                       priority=priority,
                       case_context=bool(case_context))
            
            # Advanced document categorization
            document_categories = {
                "financial": [doc for doc in documents if any(term in doc.lower() 
                             for term in ["bank", "statement", "financial", "tax", "w2", "1099"])],
                "legal": [doc for doc in documents if any(term in doc.lower() 
                         for term in ["contract", "agreement", "court", "legal", "filing"])],
                "property": [doc for doc in documents if any(term in doc.lower() 
                            for term in ["deed", "title", "property", "appraisal", "mortgage"])],
                "other": []
            }
            
            # Categorize remaining documents
            categorized = set()
            for category_docs in document_categories.values():
                categorized.update(category_docs)
            document_categories["other"] = [doc for doc in documents if doc not in categorized]
            
            # Create sophisticated task graph
            tasks = []
            task_id = 1
            
            # Create parallel analysis tasks for each category
            analysis_tasks = []
            for category, docs in document_categories.items():
                if docs:
                    task = AnalysisTask(
                        task_id=task_id,
                        name=f"Analyze {category.title()} Documents",
                        document_ids=docs,
                        analysis_type=f"{category}_analysis",
                        depends_on=[]
                    )
                    tasks.append(task)
                    analysis_tasks.append(task_id)
                    task_id += 1
            
            # Create synthesis task if multiple categories exist
            if len(analysis_tasks) > 1:
                synthesis_task = AnalysisTask(
                    task_id=task_id,
                    name="Cross-Document Validation & Synthesis",
                    document_ids=[],
                    analysis_type="cross_validation",
                    depends_on=analysis_tasks
                )
                tasks.append(synthesis_task)
                task_id += 1
            
            # Calculate cost estimates (production pricing)
            total_cost = len(documents) * 0.015  # Estimated $0.015 per document for batch processing
            if priority == "immediate":
                total_cost *= 2.5  # Immediate processing premium
            
            # Create execution plan
            execution_plan = f"""
            Document Analysis Plan:
            - Total documents: {len(documents)}
            - Processing mode: {priority}
            - Estimated cost: ${total_cost:.2f}
            - Categories: {', '.join([f"{k}: {len(v)}" for k, v in document_categories.items() if v])}
            """
            
            if priority == "batch":
                execution_plan += "\n- Estimated completion: 24 hours"
            else:
                execution_plan += "\n- Estimated completion: 15-30 minutes"
            
            task_graph = TaskGraph(
                tasks=tasks,
                execution_plan=execution_plan.strip(),
                total_estimated_cost=total_cost,
                estimated_completion_time="24h" if priority == "batch" else "30m"
            )
            
            logger.info("Analysis plan created",
                       total_tasks=len(tasks),
                       estimated_cost=total_cost,
                       priority=priority)
            
            return json.dumps({
                "status": "plan_created",
                "task_graph": task_graph.dict(),
                "total_tasks": len(tasks),
                "estimated_cost": total_cost,
                "priority": priority
            }, indent=2)
            
        except Exception as e:
            logger.error("Planning failed", error=str(e), exc_info=True)
            return json.dumps({"error": f"Planning failed: {str(e)}"})

class SubmitBatchAnalysisTool(BaseTool):
    """Production tool for OpenAI Responses API batch submission"""
    name: str = "submit_batch_analysis"
    description: str = "Submit documents to OpenAI Responses API for batch analysis"
    
    def _run(self, batch_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, batch_data: str) -> str:
        try:
            data = json.loads(batch_data)
            task_id = data.get("task_id")
            document_ids = data.get("document_ids", [])
            analysis_type = data.get("analysis_type", "general")
            case_context = data.get("case_context", "")
            custom_instructions = data.get("custom_instructions", "")
            
            logger.info("Submitting batch analysis",
                       task_id=task_id,
                       document_count=len(document_ids),
                       analysis_type=analysis_type)
            
            # Create analysis requests for OpenAI Responses API
            requests = []
            for doc_id in document_ids:
                analysis_prompt = self._create_analysis_prompt(analysis_type, case_context, custom_instructions)
                
                request = {
                    "custom_id": f"task_{task_id}_doc_{doc_id}",
                    "method": "POST",
                    "url": "/v1/responses",
                    "body": {
                        "model": "gpt-4o",
                        "messages": [
                            {
                                "role": "system",
                                "content": analysis_prompt
                            },
                            {
                                "role": "user",
                                "content": f"Analyze document: {doc_id}"
                            }
                        ],
                        "response_format": {
                            "type": "json_schema",
                            "json_schema": {
                                "name": "document_analysis",
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "document_id": {"type": "string"},
                                        "analysis_type": {"type": "string"},
                                        "extracted_data": {"type": "object"},
                                        "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
                                        "confidence_justification": {"type": "string"},
                                        "flags": {"type": "array", "items": {"type": "string"}},
                                        "key_findings": {"type": "array", "items": {"type": "string"}},
                                        "discrepancies": {"type": "array", "items": {"type": "string"}}
                                    },
                                    "required": ["document_id", "analysis_type", "confidence_score", "confidence_justification"]
                                }
                            }
                        }
                    }
                }
                requests.append(request)
            
            # Submit to OpenAI Batch API
            try:
                payload = {
                    "endpoint": "/v1/responses",
                    "completion_window": "24h",
                    "requests": requests,
                }
                
                response = await openai_client.post("/batches", json=payload)
                response.raise_for_status()
                
                batch_response = response.json()
                batch_job_id = batch_response.get("id")
                
                BATCH_JOB_COUNT.labels(status="submitted").inc()
                
                logger.info("Batch analysis submitted successfully",
                           batch_job_id=batch_job_id,
                           request_count=len(requests),
                           task_id=task_id)
                
                return json.dumps({
                    "status": "batch_submitted",
                    "batch_job_id": batch_job_id,
                    "task_id": task_id,
                    "documents_count": len(document_ids),
                    "estimated_completion": "24 hours",
                    "requests_submitted": len(requests)
                })
                
            except httpx.HTTPError as e:
                BATCH_JOB_COUNT.labels(status="failed").inc()
                logger.error("OpenAI API request failed", 
                           error=str(e), 
                           status_code=getattr(e.response, 'status_code', None),
                           task_id=task_id)
                return json.dumps({"error": f"OpenAI API error: {str(e)}"})
                
        except Exception as e:
            logger.error("Batch submission failed", error=str(e), exc_info=True)
            return json.dumps({"error": f"Batch submission failed: {str(e)}"})
    
    def _create_analysis_prompt(self, analysis_type: str, case_context: str, custom_instructions: str) -> str:
        """Create specialized analysis prompt based on document type"""
        
        base_prompt = """You are an expert document analyst specializing in legal and financial document analysis for Family Law cases. 

Analyze the provided document with extreme attention to detail and accuracy."""
        
        type_prompts = {
            "financial_analysis": """
Focus on:
- All monetary amounts, income sources, expenses
- Mathematical relationships between figures
- Account numbers, transaction patterns
- Discrepancies or unusual transactions
- Asset valuations and holdings
""",
            "legal_analysis": """
Focus on:
- Key legal terms, obligations, and rights
- Important dates and deadlines
- Parties involved and their roles
- Enforcement mechanisms
- Potential legal implications
""",
            "property_analysis": """
Focus on:
- Property descriptions and valuations
- Ownership details and transfer history
- Liens, encumbrances, or claims
- Market values and assessments
- Geographic and legal descriptions
""",
            "cross_validation": """
Focus on:
- Consistency across multiple documents
- Mathematical verification of totals
- Timeline consistency
- Discrepancies in reported values
- Missing information or gaps
"""
        }
        
        prompt = base_prompt + type_prompts.get(analysis_type, "")
        
        if case_context:
            prompt += f"\n\nCase Context: {case_context}"
        
        if custom_instructions:
            prompt += f"\n\nSpecial Instructions: {custom_instructions}"
        
        prompt += """

CRITICAL REQUIREMENTS:
1. Provide a confidence score (0-100) with detailed justification
2. Flag any OCR errors, unclear text, or quality issues
3. Identify mathematical inconsistencies or suspicious patterns
4. Extract all key data points in structured format
5. Note any information requiring human review
6. Maintain strict confidentiality and professional standards

Return analysis in the specified JSON format with complete accuracy."""
        
        return prompt

class AnalyzeDocumentTool(BaseTool):
    """Production tool for immediate document analysis"""
    name: str = "analyze_document_immediate"
    description: str = "Perform immediate document analysis for urgent cases using OpenAI"
    
    def _run(self, analysis_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, analysis_data: str) -> str:
        try:
            data = json.loads(analysis_data)
            document_id = data.get("document_id")
            analysis_type = data.get("analysis_type", "general")
            case_context = data.get("case_context", "")
            
            logger.info("Starting immediate document analysis",
                       document_id=document_id,
                       analysis_type=analysis_type)
            
            # Create analysis prompt
            submit_tool = SubmitBatchAnalysisTool()
            analysis_prompt = submit_tool._create_analysis_prompt(analysis_type, case_context, "")
            
            # Immediate OpenAI API call
            try:
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": analysis_prompt},
                        {"role": "user", "content": f"Analyze document: {document_id}"}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "document_analysis",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "document_id": {"type": "string"},
                                    "analysis_type": {"type": "string"},
                                    "extracted_data": {"type": "object"},
                                    "confidence_score": {"type": "integer", "minimum": 0, "maximum": 100},
                                    "confidence_justification": {"type": "string"},
                                    "flags": {"type": "array", "items": {"type": "string"}},
                                    "key_findings": {"type": "array", "items": {"type": "string"}}
                                },
                                "required": ["document_id", "analysis_type", "confidence_score", "confidence_justification"]
                            }
                        }
                    },
                    "temperature": 0.1
                }
                
                response = await openai_client.post("/chat/completions", json=payload)
                response.raise_for_status()
                
                result = response.json()
                analysis_content = json.loads(result["choices"][0]["message"]["content"])
                
                logger.info("Immediate analysis completed",
                           document_id=document_id,
                           confidence=analysis_content.get("confidence_score"),
                           flags_count=len(analysis_content.get("flags", [])))
                
                return json.dumps({
                    "status": "analysis_complete",
                    "results": analysis_content,
                    "processing_time_ms": 1500  # Estimated
                }, indent=2)
                
            except httpx.HTTPError as e:
                logger.error("OpenAI immediate analysis failed",
                           error=str(e),
                           document_id=document_id)
                return json.dumps({"error": f"OpenAI API error: {str(e)}"})
                
        except Exception as e:
            logger.error("Immediate analysis failed", error=str(e), exc_info=True)
            return json.dumps({"error": f"Immediate analysis failed: {str(e)}"})

class SynthesizeResultsTool(BaseTool):
    """Production tool for validation and cross-document comparison"""
    name: str = "synthesize_results"
    description: str = "Validate analysis results or compare multiple documents with AI assistance"
    
    def _run(self, synthesis_data: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, synthesis_data: str) -> str:
        try:
            data = json.loads(synthesis_data)
            synthesis_type = data.get("synthesis_type")
            instructions = data.get("instructions")
            data_sources = data.get("data_sources", [])
            
            logger.info("Starting result synthesis",
                       synthesis_type=synthesis_type,
                       sources_count=len(data_sources))
            
            # Create synthesis prompt
            synthesis_prompt = f"""You are an expert analyst performing {synthesis_type} analysis.

Instructions: {instructions}

Data Sources: {json.dumps(data_sources, indent=2)}

Perform the requested analysis and provide:
1. Detailed findings
2. Confidence assessment
3. Recommendations for next steps
4. Any issues requiring human review

Return structured analysis with clear conclusions."""
            
            try:
                payload = {
                    "model": "gpt-4o",
                    "messages": [
                        {"role": "system", "content": synthesis_prompt}
                    ],
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": "synthesis_analysis",
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "synthesis_type": {"type": "string"},
                                    "status": {"type": "string"},
                                    "findings": {"type": "array", "items": {"type": "string"}},
                                    "confidence_score": {"type": "integer"},
                                    "recommendations": {"type": "array", "items": {"type": "string"}},
                                    "discrepancies": {"type": "array", "items": {"type": "string"}},
                                    "requires_human_review": {"type": "boolean"}
                                },
                                "required": ["synthesis_type", "status", "confidence_score"]
                            }
                        }
                    },
                    "temperature": 0.1
                }
                
                response = await openai_client.post("/chat/completions", json=payload)
                response.raise_for_status()
                
                result = response.json()
                synthesis_result = json.loads(result["choices"][0]["message"]["content"])
                
                logger.info("Synthesis completed",
                           synthesis_type=synthesis_type,
                           status=synthesis_result.get("status"),
                           confidence=synthesis_result.get("confidence_score"))
                
                return json.dumps({
                    "status": "synthesis_complete",
                    "results": synthesis_result
                }, indent=2)
                
            except httpx.HTTPError as e:
                logger.error("OpenAI synthesis failed", error=str(e))
                return json.dumps({"error": f"OpenAI API error: {str(e)}"})
                
        except Exception as e:
            logger.error("Synthesis failed", error=str(e), exc_info=True)
            return json.dumps({"error": f"Synthesis failed: {str(e)}"})

class GetCaseContextTool(BaseTool):
    """Production tool to retrieve case context from backend"""
    name: str = "get_case_context"
    description: str = "Retrieve comprehensive case details and context from backend system"
    
    def _run(self, case_id: str) -> str:
        raise NotImplementedError("Use async version")
    
    async def _arun(self, case_id: str) -> str:
        try:
            logger.info("Retrieving case context", case_id=case_id)
            
            # Real backend API call
            response = await http_client.get(f"{BACKEND_URL}/api/cases/{case_id}")
            
            if response.status_code == 404:
                return json.dumps({"error": "Case not found"})
            
            response.raise_for_status()
            case_data = response.json()
            
            logger.info("Case context retrieved",
                       case_id=case_id,
                       case_type=case_data.get("case_type"))
            
            return json.dumps(case_data, indent=2)
            
        except httpx.HTTPError as e:
            logger.error("Failed to retrieve case context",
                        case_id=case_id,
                        error=str(e))
            return json.dumps({"error": f"Backend API error: {str(e)}"})
        except Exception as e:
            logger.error("Case context retrieval failed",
                        case_id=case_id,
                        error=str(e))
            return json.dumps({"error": f"Context retrieval failed: {str(e)}"})

# Production workflow execution with monitoring
class ProductionCallbackHandler(BaseCallbackHandler):
    """Production callback handler with comprehensive monitoring"""
    
    def __init__(self, workflow_id: str):
        self.workflow_id = workflow_id
        self.start_time = datetime.now()
    
    async def on_llm_start(self, serialized, prompts, **kwargs):
        await self._add_reasoning_step("Agent analyzing document analysis strategy...")
    
    async def on_tool_start(self, serialized, input_str, **kwargs):
        tool_name = serialized.get('name', 'Unknown tool')
        await self._add_reasoning_step(
            f"Executing {tool_name}",
            action=tool_name,
            action_input={"input": str(input_str)[:500]}
        )
    
    async def on_tool_end(self, output, **kwargs):
        await self._add_reasoning_step(
            "Tool execution completed",
            action_output=str(output)[:1000]
        )
    
    async def on_agent_action(self, action, **kwargs):
        await self._add_reasoning_step(
            f"Agent decided: {action.tool}. Reasoning: {action.log[:500]}..."
        )
    
    async def _add_reasoning_step(self, thought: str, action: str = None, 
                                action_input: Dict = None, action_output: str = None):
        try:
            step = {
                "timestamp": datetime.now().isoformat(),
                "thought": thought,
                "action": action,
                "action_input": action_input,
                "action_output": action_output
            }
            
            response = await http_client.post(
                f"{BACKEND_URL}/api/workflows/{self.workflow_id}/reasoning",
                json=step
            )
            response.raise_for_status()
            
        except Exception as e:
            logger.error("Failed to log reasoning step",
                        workflow_id=self.workflow_id,
                        error=str(e))

def create_production_agent(workflow_id: str) -> AgentExecutor:
    """Create production document analysis agent"""
    
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
        SystemMessage(content="""You are a Production Document Analysis Agent for a Family Law firm specializing in financial discovery.

PRODUCTION RESPONSIBILITIES:
- Orchestrate intelligent document analysis with cost optimization and accuracy focus
- Maintain complete audit trails for legal compliance
- Apply sophisticated quality control and validation processes
- Make informed decisions about batch vs immediate processing
- Ensure all analysis meets professional legal standards

WORKFLOW EXECUTION:
1. PLANNING: Create comprehensive task dependency graphs based on document types and relationships
2. EXECUTION: Submit to appropriate processing pipeline (batch for cost efficiency, immediate for urgency)
3. MONITORING: Track analysis progress and quality metrics
4. SYNTHESIS: Perform cross-document validation and discrepancy detection
5. QUALITY CONTROL: Apply confidence thresholds and human review triggers

COST OPTIMIZATION:
- Default to batch processing for 50% cost savings when time permits
- Use immediate processing only for urgent legal deadlines
- Consider document complexity in processing decisions
- Optimize batch job grouping for efficiency

QUALITY STANDARDS:
- Require confidence scores ≥85% for automated acceptance
- Flag mathematical inconsistencies or OCR errors
- Cross-validate related documents for discrepancies
- Apply case-specific thresholds for human review triggers
- Maintain detailed reasoning chains for legal audit requirements

LEGAL COMPLIANCE:
- Preserve complete workflow audit trails
- Apply appropriate confidentiality measures
- Flag potential privileged or sensitive content
- Ensure professional standards in all analysis outputs

Execute with precision, efficiency, and unwavering attention to legal and professional standards."""),
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
        max_iterations=25,
        return_intermediate_steps=True
    )

# FastAPI application with production configuration
@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_http_clients()
    logger.info("Document Analysis Agent started", port=PORT)
    yield
    await close_http_clients()
    logger.info("Document Analysis Agent stopped")

app = FastAPI(
    title="Production Document Analysis Agent",
    description="AI-powered document intelligence for Family Law financial discovery",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware for metrics and logging
@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    start_time = datetime.now()
    
    response = await call_next(request)
    
    # Record metrics
    duration = (datetime.now() - start_time).total_seconds()
    REQUEST_LATENCY.observe(duration)
    REQUEST_COUNT.labels(
        method=request.method,
        endpoint=request.url.path,
        status=response.status_code
    ).inc()
    
    # Log request
    logger.info("Request processed",
                method=request.method,
                path=request.url.path,
                status_code=response.status_code,
                duration_ms=duration * 1000)
    
    return response

# Production endpoints
@app.get("/")
async def health_check():
    """Comprehensive health check"""
    try:
        # Test backend connectivity
        backend_response = await http_client.get(f"{BACKEND_URL}/", timeout=5.0)
        backend_status = "healthy" if backend_response.status_code == 200 else "unhealthy"
    except Exception:
        backend_status = "unreachable"
    
    # Test OpenAI connectivity  
    try:
        openai_response = await openai_client.get("/models", timeout=5.0)
        openai_status = "healthy" if openai_response.status_code == 200 else "unhealthy"
    except Exception:
        openai_status = "unreachable"
    
    overall_status = "healthy" if all(s == "healthy" for s in [backend_status, openai_status]) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0",
        "services": {
            "backend": backend_status,
            "openai": openai_status
        },
        "environment": {
            "anthropic_configured": bool(ANTHROPIC_API_KEY),
            "openai_configured": bool(OPENAI_API_KEY),
            "backend_url": BACKEND_URL
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint"""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/auth/token")
async def create_token(user_data: dict):
    """Create JWT token for testing (remove in production)"""
    if os.getenv("ENVIRONMENT") == "production":
        raise HTTPException(status_code=404, detail="Not found")
    
    token = create_access_token(user_data)
    return {"access_token": token, "token_type": "bearer"}

@app.post("/workflows/trigger-analysis", response_model=DocumentAnalysisResponse)
async def trigger_document_analysis(
    request: DocumentAnalysisRequest, 
    background_tasks: BackgroundTasks,
    user: Dict = Depends(verify_jwt_token)
):
    """Trigger production document analysis workflow"""
    
    workflow_id = f"wf_analysis_{uuid.uuid4().hex[:12]}"
    
    # Log user action
    logger.info("Document analysis triggered",
                workflow_id=workflow_id,
                case_id=request.case_id,
                document_count=len(request.document_ids),
                priority=request.analysis_priority,
                user_id=user.get("user_id"))
    
    # Create workflow in backend
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_type": "DocumentAnalysisAgent",
        "case_id": request.case_id,
        "status": "PENDING_PLANNING",
        "initial_prompt": f"Analyze documents for case {request.case_id}: {request.document_ids}. Priority: {request.analysis_priority}",
        "document_ids": request.document_ids,
        "case_context": request.case_context,
        "priority": request.analysis_priority
    }
    
    try:
        response = await http_client.post(f"{BACKEND_URL}/api/workflows", json=workflow_data)
        response.raise_for_status()
        
        WORKFLOW_COUNT.labels(agent_type="DocumentAnalysisAgent", status="triggered").inc()
        
        # Execute in background
        background_tasks.add_task(
            execute_production_workflow,
            workflow_id,
            request.document_ids,
            request.case_context or "",
            request.analysis_priority,
            request.custom_instructions or ""
        )
        
        # Calculate estimates
        cost_estimate = len(request.document_ids) * 0.015
        if request.analysis_priority == "immediate":
            cost_estimate *= 2.5
            completion_time = "15-30 minutes"
        else:
            completion_time = "24 hours"
        
        return DocumentAnalysisResponse(
            workflow_id=workflow_id,
            status=DocumentAnalysisStatus.PENDING_PLANNING,
            message=f"Document analysis workflow {workflow_id} initiated",
            estimated_completion=completion_time,
            cost_estimate=cost_estimate
        )
        
    except Exception as e:
        logger.error("Failed to trigger analysis workflow",
                    workflow_id=workflow_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail=f"Workflow creation failed: {str(e)}")

@app.get("/workflows/{workflow_id}")
async def get_workflow_status(workflow_id: str, user: Dict = Depends(verify_jwt_token)):
    """Get production workflow status with detailed information"""
    try:
        response = await http_client.get(f"{BACKEND_URL}/api/workflows/{workflow_id}")
        
        if response.status_code == 404:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        response.raise_for_status()
        workflow_data = response.json()
        
        # Add computed fields
        workflow_data["user_access"] = user.get("user_id")
        
        return workflow_data
        
    except httpx.HTTPError as e:
        logger.error("Failed to retrieve workflow status",
                    workflow_id=workflow_id,
                    error=str(e))
        raise HTTPException(status_code=500, detail="Failed to retrieve workflow status")

@app.post("/webhooks/batch-complete")
async def production_batch_complete(batch_data: dict):
    """Production webhook for OpenAI batch completion"""
    batch_job_id = batch_data.get("id") or batch_data.get("batch_job_id")
    status = batch_data.get("status", "completed")
    
    logger.info("Batch analysis webhook received",
                batch_job_id=batch_job_id,
                status=status)
    
    if status == "completed":
        BATCH_JOB_COUNT.labels(status="completed").inc()
        
        # Process batch results
        try:
            # Get batch results from OpenAI
            response = await openai_client.get(f"/batches/{batch_job_id}")
            response.raise_for_status()
            
            batch_info = response.json()
            output_file_id = batch_info.get("output_file_id")
            
            if output_file_id:
                # Download results
                file_response = await openai_client.get(f"/files/{output_file_id}/content")
                file_response.raise_for_status()
                
                # Process results and update workflows
                results = {"batch_results": file_response.text, "status": "completed"}
                
                # Forward to backend webhook
                webhook_response = await http_client.post(
                    f"{BACKEND_URL}/api/webhooks/batch-complete",
                    json={"batch_job_id": batch_job_id, "results": results}
                )
                webhook_response.raise_for_status()
                
                logger.info("Batch results processed successfully",
                           batch_job_id=batch_job_id,
                           output_file_id=output_file_id)
            
        except Exception as e:
            logger.error("Failed to process batch results",
                        batch_job_id=batch_job_id,
                        error=str(e))
            BATCH_JOB_COUNT.labels(status="failed").inc()
    
    return {"status": "webhook_processed", "batch_job_id": batch_job_id}

@app.post("/chat")
async def production_chat(
    request: dict, 
    user: Dict = Depends(verify_jwt_token)
):
    """Production chat interface with authentication"""
    
    workflow_id = f"wf_chat_{uuid.uuid4().hex[:8]}"
    message = request.get("message", "")
    case_id = request.get("case_id")
    
    logger.info("Interactive chat session started",
                workflow_id=workflow_id,
                user_id=user.get("user_id"),
                case_id=case_id)
    
    # Create workflow
    workflow_data = {
        "workflow_id": workflow_id,
        "agent_type": "DocumentAnalysisAgent",
        "case_id": case_id,
        "status": "PENDING_PLANNING",
        "initial_prompt": message,
        "priority": "interactive"
    }
    
    try:
        response = await http_client.post(f"{BACKEND_URL}/api/workflows", json=workflow_data)
        response.raise_for_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create chat workflow: {e}")
    
    async def generate_production_stream():
        try:
            yield f"data: {json.dumps({'type': 'workflow_started', 'workflow_id': workflow_id})}\n\n"
            
            # Execute agent with production monitoring
            callback_handler = ProductionCallbackHandler(workflow_id)
            agent = create_production_agent(workflow_id)
            
            result = await agent.ainvoke(
                {"input": message},
                config={"callbacks": [callback_handler]}
            )
            
            # Update workflow status
            await http_client.put(
                f"{BACKEND_URL}/api/workflows/{workflow_id}/status",
                json={"status": "COMPLETED"}
            )
            
            WORKFLOW_COUNT.labels(agent_type="DocumentAnalysisAgent", status="completed").inc()
            
            # Stream final result
            output = result.get("output", "Analysis completed")
            yield f"data: {json.dumps({'type': 'final_response', 'response': output})}\n\n"
            yield f"data: {json.dumps({'type': 'workflow_complete', 'workflow_id': workflow_id})}\n\n"
            
        except Exception as e:
            logger.error("Chat workflow failed",
                        workflow_id=workflow_id,
                        error=str(e))
            
            await http_client.put(
                f"{BACKEND_URL}/api/workflows/{workflow_id}/status",
                json={"status": "FAILED"}
            )
            
            WORKFLOW_COUNT.labels(agent_type="DocumentAnalysisAgent", status="failed").inc()
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
    
    return StreamingResponse(
        generate_production_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive"
        }
    )

# Production background tasks
async def execute_production_workflow(
    workflow_id: str, 
    document_ids: List[str], 
    case_context: str, 
    priority: str,
    custom_instructions: str
):
    """Execute document analysis workflow in production environment"""
    try:
        logger.info("Starting production workflow execution",
                   workflow_id=workflow_id,
                   document_count=len(document_ids),
                   priority=priority)
        
        # Update status
        await http_client.put(
            f"{BACKEND_URL}/api/workflows/{workflow_id}/status",
            json={"status": "PENDING_PLANNING"}
        )
        
        # Execute agent
        callback_handler = ProductionCallbackHandler(workflow_id)
        agent = create_production_agent(workflow_id)
        
        prompt = f"""Execute production document analysis workflow:
        
Documents: {document_ids}
Case Context: {case_context}
Priority: {priority}
Custom Instructions: {custom_instructions}

Create comprehensive analysis plan and execute with full quality controls and cost optimization."""
        
        result = await agent.ainvoke(
            {"input": prompt},
            config={"callbacks": [callback_handler]}
        )
        
        # Update final status
        await http_client.put(
            f"{BACKEND_URL}/api/workflows/{workflow_id}/status",
            json={"status": "COMPLETED"}
        )
        
        WORKFLOW_COUNT.labels(agent_type="DocumentAnalysisAgent", status="completed").inc()
        
        logger.info("Production workflow completed successfully",
                   workflow_id=workflow_id)
        
    except Exception as e:
        logger.error("Production workflow failed",
                    workflow_id=workflow_id,
                    error=str(e))
        
        await http_client.put(
            f"{BACKEND_URL}/api/workflows/{workflow_id}/status",
            json={"status": "FAILED"}
        )
        
        WORKFLOW_COUNT.labels(agent_type="DocumentAnalysisAgent", status="failed").inc()

if __name__ == "__main__":
    import uvicorn
    
    # Production-ready server configuration
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=PORT,
        log_level=LOG_LEVEL.lower(),
        access_log=True,
        loop="asyncio",
        http="httptools"
    )