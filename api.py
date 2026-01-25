"""
FastAPI endpoints for the Data Engineer Agent system.

Run with: uvicorn api:app --reload
Docs at: http://localhost:8000/docs
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid

# Import existing workflow and agents
from langgraph_workflow import execute_workflow
from agents.connector import connector_agent
from agents.data_transformer import smart_transformer_agent
from agents.bigquery_assistant import bigquery_agent

# ============================================
# Pydantic Models
# ============================================

class WorkflowRequest(BaseModel):
    """Request model for workflow execution"""
    request: str = Field(..., description="Natural language request for the data engineering task")
    
    class Config:
        json_schema_extra = {
            "example": {
                "request": "Download sales.csv from bucket my-bucket in project my-project, add a date column, then upload to processed-bucket"
            }
        }

class AgentRequest(BaseModel):
    """Request model for individual agent calls"""
    message: str = Field(..., description="Task description for the agent")
    
    class Config:
        json_schema_extra = {
            "example": {
                "message": "Download file.csv from bucket data_storage in project my-gcp-project"
            }
        }

class TaskInfo(BaseModel):
    """Individual task information"""
    agent: str
    task: str

class WorkflowResponse(BaseModel):
    """Response model for workflow execution"""
    status: str = Field(..., description="Status of the workflow: 'completed' or 'failed'")
    request: str = Field(..., description="Original user request")
    tasks_completed: List[Dict[str, Any]] = Field(default=[], description="List of completed tasks")
    final_agent: str = Field(..., description="Last agent that was called")
    error: Optional[str] = Field(None, description="Error details if workflow failed")
    execution_id: str = Field(..., description="Unique execution identifier")
    timestamp: str = Field(..., description="Execution timestamp")

class AgentResponse(BaseModel):
    """Response model for individual agent calls"""
    status: str
    agent: str
    message: str
    response: str
    timestamp: str

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str
    timestamp: str

# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="Data Engineer Agent API",
    description="""
    API for the multi-agent Data Engineering system.
    
    This system orchestrates multiple AI agents to perform data engineering tasks:
    - **Connector Agent**: GCS operations (upload, download files)
    - **Transformer Agent**: Pandas-based data transformations
    - **BigQuery Agent**: BigQuery operations (datasets, tables, queries)
    
    The workflow automatically routes tasks to the appropriate agent and handles errors
    through a conflict resolver.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================
# Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """
    Health check endpoint to verify the API is running.
    """
    return HealthResponse(
        status="healthy",
        service="Data Engineer Agent API",
        version="1.0.0",
        timestamp=datetime.now().isoformat()
    )


@app.post("/workflow", response_model=WorkflowResponse, tags=["Workflow"])
async def run_workflow(request: WorkflowRequest):
    """
    Execute the full multi-agent workflow.
    
    This endpoint takes a natural language request and orchestrates the appropriate
    agents to complete the task. The workflow will:
    
    1. Parse the request and break it into atomic tasks
    2. Route each task to the appropriate agent
    3. Handle any errors through the conflict resolver
    4. Return the final state with all completed tasks
    
    **Example requests:**
    - "Download sales.csv from bucket my-bucket in project my-project"
    - "Merge file1.csv and file2.csv on column id, save to merged.csv"
    - "Create BigQuery dataset analytics in project my-project"
    """
    execution_id = str(uuid.uuid4())[:8]
    
    try:
        # Execute the workflow
        result = execute_workflow(request.request)
        
        # Determine status based on final state
        status = "completed" if result.get('next_agent') == 'END' else "incomplete"
        
        # Filter out empty task entries
        tasks = [t for t in result.get('tasks_done', []) if t and any(t.values())]
        
        return WorkflowResponse(
            status=status,
            request=request.request,
            tasks_completed=tasks,
            final_agent=result.get('next_agent', 'unknown'),
            error=result.get('error_details') if result.get('error_details') else None,
            execution_id=execution_id,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": str(e),
                "execution_id": execution_id,
                "message": "Workflow execution failed"
            }
        )


@app.post("/agents/connector", response_model=AgentResponse, tags=["Agents"])
async def call_connector(request: AgentRequest):
    """
    Call the Connector Agent directly for GCS operations.
    
    The Connector Agent can:
    - Download files from GCS to local storage
    - Upload files from local storage to GCS
    - Create new GCS buckets
    
    **Example messages:**
    - "Download file.csv from bucket data_storage in project my-gcp-project"
    - "Upload ./data/output.csv to bucket results in project my-gcp-project as output.csv"
    """
    try:
        result = connector_agent.invoke({
            "messages": [{"role": "user", "content": request.message}]
        })
        
        response_text = result['messages'][-1].content
        status = "error" if "ERROR" in response_text.upper() else "success"
        
        return AgentResponse(
            status=status,
            agent="connector_agent",
            message=request.message,
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/transformer", response_model=AgentResponse, tags=["Agents"])
async def call_transformer(request: AgentRequest):
    """
    Call the Smart Transformer Agent directly for data transformations.
    
    The Transformer Agent can:
    - Preview data files (CSV, Excel, JSON, Parquet)
    - Generate and execute pandas transformations
    - Save transformed data to various formats
    
    **Example messages:**
    - "Preview ./data/sales.csv"
    - "Read ./data/input.csv, filter rows where amount > 100, save to ./data/filtered.csv"
    - "Merge ./data/file1.csv and ./data/file2.csv on column id"
    """
    try:
        result = smart_transformer_agent.invoke({
            "messages": [{"role": "user", "content": request.message}]
        })
        
        response_text = result['messages'][-1].content
        status = "error" if "ERROR" in response_text.upper() else "success"
        
        return AgentResponse(
            status=status,
            agent="smart_transformer_agent",
            message=request.message,
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/agents/bigquery", response_model=AgentResponse, tags=["Agents"])
async def call_bigquery(request: AgentRequest):
    """
    Call the BigQuery Agent directly for BigQuery operations.
    
    The BigQuery Agent can:
    - Create datasets and tables
    - Execute SQL queries
    - Load data from GCS into BigQuery
    - Insert rows into tables
    
    **Example messages:**
    - "Create dataset sales_data in project my-gcp-project"
    - "Query SELECT * FROM sales_data.transactions LIMIT 10 in project my-gcp-project"
    - "Create table users in dataset analytics with schema [{'name': 'id', 'type': 'STRING'}]"
    """
    try:
        result = bigquery_agent.invoke({
            "messages": [{"role": "user", "content": request.message}]
        })
        
        response_text = result['messages'][-1].content
        status = "error" if "ERROR" in response_text.upper() else "success"
        
        return AgentResponse(
            status=status,
            agent="bigquery_agent",
            message=request.message,
            response=response_text,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================
# Run with: uvicorn api:app --reload
# ============================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
