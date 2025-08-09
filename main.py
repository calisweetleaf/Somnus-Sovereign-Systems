import asyncio
import logging
import signal
import sys
import argparse
import yaml
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Core imports
from core.session_manager import SessionManager
from core.model_loader import ModelLoader
from core.security_layer import SecurityEnforcer

# New advanced feature imports
from core.memory_manager import MemoryManager, MemoryConfiguration, MemoryType, MemoryImportance
from core.memory_integration import EnhancedSessionManager, create_memory_enhanced_session_manager
from core.file_upload_system import FileUploadManager, FileType, ProcessingStatus
from core.web_search_research import WebSearchEngine, DeepResearchEngine, ResearchQuery, create_research_system
from core.artifact_system import ArtifactManager, ArtifactType, SecurityLevel

# Schema imports
from schemas.session import (
    SessionCreationRequest, SessionResponse, SessionListResponse,
    SessionMetadata, SessionID
)
from schemas.models import ModelLoadRequest, ModelLoadResponse
from schemas.memory import (
    MemoryCreationRequest, MemorySearchRequest, MemorySearchResponse,
    MemoryExportRequest
)
from schemas.research import ResearchRequest, ResearchResponse
from schemas.artifacts import ArtifactCreationRequest, ArtifactResponse


# Global system components
enhanced_session_manager: Optional[EnhancedSessionManager] = None
memory_manager: Optional[MemoryManager] = None
model_loader: Optional[ModelLoader] = None
security_enforcer: Optional[SecurityEnforcer] = None
file_upload_manager: Optional[FileUploadManager] = None
web_search_engine: Optional[WebSearchEngine] = None
research_engine: Optional[DeepResearchEngine] = None
artifact_manager: Optional[ArtifactManager] = None
system_config: Dict[str, Any] = {}


class ChatMessage(BaseModel):
    """Enhanced chat message with advanced options"""
    message: str
    user_id: str
    session_id: Optional[str] = None
    enable_web_search: bool = False
    enable_research: bool = False
    research_depth: int = 2
    store_memory: bool = True
    create_artifacts: bool = True
    custom_instructions: Optional[str] = None


class ChatResponse(BaseModel):
    """Enhanced chat response with comprehensive metadata"""
    message: str
    session_id: str
    memory_stored: bool = False
    memory_ids: List[str] = []
    artifacts_created: List[Dict[str, Any]] = []
    research_conducted: bool = False
    research_sources: int = 0
    web_searches_performed: int = 0
    processing_time_ms: float = 0
    tokens_used: Dict[str, int] = {}
    confidence_score: float = 1.0


def load_configuration(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Enhanced configuration loading with new feature configs"""
    config_paths = [
        config_path,
        "config/production.yaml",
        "config/base.yaml",
        "config.yaml",
        "/etc/morpheus-chat/config.yaml"
    ]
    
    config = {}
    
    for path in config_paths:
        if path and Path(path).exists():
            try:
                with open(path, 'r') as f:
                    loaded_config = yaml.safe_load(f)
                    if loaded_config:
                        config = {**config, **loaded_config}
                        logging.info(f"Loaded configuration from {path}")
                        break
            except Exception as e:
                logging.error(f"Failed to load config from {path}: {e}")
    
    if not config:
        # Enhanced fallback configuration
        config = {
            "system": {
                "name": "morpheus-chat",
                "version": "1.0.0",
                "environment": "development"
            },
            "session_management": {
                "max_sessions": 10,
                "session_timeout": 3600,
                "resources": {
                    "cpu_limit": "2.0",
                    "memory_limit": "4G",
                    "disk_limit": "2G"
                }
            },
            "memory_system": {
                "storage": {
                    "vector_db_path": "data/memory/vectors",
                    "metadata_db_path": "data/memory/metadata.db"
                },
                "embeddings": {
                    "model": "sentence-transformers/all-MiniLM-L6-v2"
                }
            },
            "web_search": {
                "enabled": True,
                "provider": "duckduckgo"
            },
            "document_processing": {
                "max_file_size_mb": 100,
                "supported_formats": [".txt", ".pdf", ".docx", ".md", ".py", ".js"]
            },
            "artifacts": {
                "enabled": True,
                "storage": "data/artifacts"
            }
        }
        logging.warning("Using default configuration")
    
    return config


def setup_logging(config: Dict[str, Any]):
    """Enhanced logging configuration"""
    log_config = config.get("logging", {})
    
    logging.basicConfig(
        level=getattr(logging, log_config.get("level", "INFO")),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ]
    )
    
    # File handler
    if log_config.get("file"):
        file_handler = logging.FileHandler(log_config["file"])
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logging.getLogger().addHandler(file_handler)
    
    # Set third-party library log levels
    third_party_loggers = [
        "transformers", "torch", "docker", "chromadb", "sentence_transformers",
        "httpx", "urllib3", "PIL", "matplotlib"
    ]
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
    
    logging.info("Enhanced logging system initialized")


async def initialize_system(config: Dict[str, Any]) -> bool:
    """Enhanced system initialization with all new components"""
    global (enhanced_session_manager, memory_manager, model_loader, security_enforcer,
            file_upload_manager, web_search_engine, research_engine, artifact_manager)
    
    try:
        logging.info("Initializing Morpheus Chat enhanced system...")
        
        # Initialize Security Enforcer
        security_config = config.get("security", {})
        security_enforcer = SecurityEnforcer(security_config)
        logging.info("✓ Security enforcer initialized")
        
        # Initialize Model Loader
        model_config = {
            "model_registry": config.get("model_registry", {}),
            "inference": config.get("inference", {}),
            "model_management": config.get("model_management", {})
        }
        model_loader = ModelLoader(model_config)
        logging.info("✓ Model loader initialized")
        
        # Initialize Memory Manager
        memory_config = MemoryConfiguration(**config.get("memory_system", {}))
        memory_manager = MemoryManager(memory_config)
        await memory_manager.initialize()
        logging.info("✓ Memory manager initialized")
        
        # Initialize Base Session Manager
        session_config = {
            "session_management": config.get("session_management", {}),
            "container_runtime": config.get("container_runtime", {}),
            "security": security_config
        }
        base_session_manager = SessionManager(session_config)
        await base_session_manager.start()
        
        # Create Enhanced Session Manager with Memory
        enhanced_session_manager = EnhancedSessionManager(base_session_manager, memory_manager)
        logging.info("✓ Enhanced session manager initialized")
        
        # Initialize File Upload Manager
        upload_config = config.get("document_processing", {})
        file_upload_manager = FileUploadManager(
            upload_dir="data/uploads",
            gguf_model_path=upload_config.get("gguf_embedding_model"),
            max_file_size_mb=upload_config.get("max_file_size_mb", 100)
        )
        logging.info("✓ File upload manager initialized")
        
        # Initialize Web Search and Research
        web_search_engine, research_engine = await create_research_system()
        logging.info("✓ Web search and research engines initialized")
        
        # Initialize Artifact Manager
        artifact_config = config.get("artifacts", {})
        artifact_manager = ArtifactManager(
            storage_dir=artifact_config.get("storage", "data/artifacts")
        )
        logging.info("✓ Artifact manager initialized")
        
        # Verify system health
        health_status = await perform_health_check()
        if not health_status.get("healthy", False):
            logging.error("System health check failed")
            return False
        
        logging.info("🚀 Morpheus Chat enhanced system initialized successfully")
        return True
        
    except Exception as e:
        logging.error(f"System initialization failed: {e}")
        return False


async def shutdown_system():
    """Enhanced graceful shutdown"""
    global (enhanced_session_manager, memory_manager, web_search_engine, 
            research_engine, artifact_manager)
    
    logging.info("Shutting down Morpheus Chat enhanced system...")
    
    # Shutdown in reverse order
    shutdown_tasks = []
    
    if enhanced_session_manager:
        shutdown_tasks.append(enhanced_session_manager.session_manager.shutdown())
    
    if memory_manager:
        shutdown_tasks.append(memory_manager.shutdown())
    
    if web_search_engine:
        shutdown_tasks.append(web_search_engine.__aexit__(None, None, None))
    
    # Execute shutdowns concurrently
    await asyncio.gather(*shutdown_tasks, return_exceptions=True)
    
    logging.info("🛑 Enhanced system shutdown complete")


async def perform_health_check() -> Dict[str, Any]:
    """Enhanced health check for all components"""
    health_status = {
        "healthy": True,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "components": {},
        "issues": []
    }
    
    # Check Memory Manager
    if memory_manager:
        try:
            metrics = memory_manager.get_metrics()
            health_status["components"]["memory_manager"] = {
                "status": "healthy",
                "metrics": metrics
            }
        except Exception as e:
            health_status["healthy"] = False
            health_status["issues"].append(f"Memory manager unhealthy: {e}")
            health_status["components"]["memory_manager"] = {"status": "unhealthy"}
    
    # Check Enhanced Session Manager
    if enhanced_session_manager:
        try:
            session_metrics = enhanced_session_manager.session_manager.get_metrics()
            health_status["components"]["session_manager"] = {
                "status": "healthy",
                "metrics": session_metrics
            }
        except Exception as e:
            health_status["healthy"] = False
            health_status["issues"].append(f"Session manager unhealthy: {e}")
    
    # Check File Upload Manager
    if file_upload_manager:
        health_status["components"]["file_upload"] = {
            "status": "healthy",
            "upload_dir_exists": Path(file_upload_manager.upload_dir).exists()
        }
    
    # Check Artifact Manager
    if artifact_manager:
        health_status["components"]["artifact_manager"] = {
            "status": "healthy",
            "storage_dir_exists": Path(artifact_manager.storage_dir).exists()
        }
    
    # Check Research Components
    if web_search_engine and research_engine:
        health_status["components"]["research_system"] = {
            "status": "healthy",
            "web_search_available": True,
            "research_engine_available": True
        }
    
    return health_status


# FastAPI application setup
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Enhanced application lifespan management"""
    # Startup
    if not await initialize_system(system_config):
        logging.error("Failed to initialize enhanced system")
        sys.exit(1)
    
    yield
    
    # Shutdown
    await shutdown_system()


app = FastAPI(
    title="Morpheus Chat - Enhanced",
    description="Production-grade ChatGPT-equivalent system with memory, research, and artifacts",
    version="1.0.0",
    lifespan=lifespan
)

# Enhanced middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(GZipMiddleware, minimum_size=1000)

# Serve static files (UI)
app.mount("/static", StaticFiles(directory="static"), name="static")


# Enhanced API Routes

@app.get("/", response_class=HTMLResponse)
async def serve_ui():
    """Serve the web UI dashboard"""
    ui_path = Path("static/index.html")
    if ui_path.exists():
        with open(ui_path, 'r') as f:
            return f.read()
    else:
        return """
        <!DOCTYPE html>
        <html><head><title>Morpheus Chat</title></head>
        <body>
        <h1>Morpheus Chat - Enhanced AI Assistant</h1>
        <p>Web UI not found. Please ensure static files are properly deployed.</p>
        <p><a href="/docs">API Documentation</a></p>
        </body></html>
        """


@app.get("/health")
async def health_check():
    """Enhanced system health check"""
    return await perform_health_check()


@app.get("/status")
async def system_status():
    """Enhanced system status with all components"""
    try:
        import psutil
        
        status = {
            "status": "operational",
            "version": system_config.get("system", {}).get("version", "1.0.0"),
            "uptime_seconds": 0.0,
            "system_metrics": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            },
            "components": {}
        }
        
        # Session metrics
        if enhanced_session_manager:
            session_metrics = enhanced_session_manager.session_manager.get_metrics()
            status["components"]["sessions"] = {
                "active": session_metrics.get("active_sessions", 0),
                "total_created": session_metrics.get("sessions_created", 0)
            }
        
        # Memory metrics
        if memory_manager:
            memory_metrics = memory_manager.get_metrics()
            status["components"]["memory"] = memory_metrics
        
        # Model metrics
        if model_loader:
            model_info = model_loader.get_system_info()
            status["components"]["models"] = {
                "loaded_models": model_info.get("loaded_models", 0)
            }
        
        # File upload metrics
        if file_upload_manager:
            status["components"]["file_processing"] = {
                "upload_dir": str(file_upload_manager.upload_dir),
                "max_file_size_mb": file_upload_manager.max_file_size / (1024*1024)
            }
        
        # Artifact metrics
        if artifact_manager:
            status["components"]["artifacts"] = {
                "cache_size": len(artifact_manager.artifact_cache)
            }
        
        return status
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system status: {e}")


# Enhanced Chat API
@app.post("/api/chat", response_model=ChatResponse)
async def enhanced_chat(message_data: ChatMessage):
    """Enhanced chat endpoint with memory, research, and artifact capabilities"""
    start_time = asyncio.get_event_loop().time()
    
    try:
        user_id = message_data.user_id
        message = message_data.message
        session_id = message_data.session_id
        
        # Get or create session with memory context
        if not session_id:
            session_request = SessionCreationRequest(
                user_id=user_id,
                model_id="gpt-4Ø",  # Default model
                custom_instructions=message_data.custom_instructions
            )
            session_response, memory_context = await enhanced_session_manager.create_session_with_memory(session_request)
            session_id = str(session_response.session_id)
        else:
            memory_context = enhanced_session_manager.get_memory_context(session_id)
        
        if not memory_context:
            raise HTTPException(status_code=404, detail="Session not found")
        
        # Initialize response tracking
        response_data = {
            "message": "",
            "session_id": session_id,
            "memory_stored": False,
            "memory_ids": [],
            "artifacts_created": [],
            "research_conducted": False,
            "research_sources": 0,
            "web_searches_performed": 0,
            "processing_time_ms": 0,
            "tokens_used": {},
            "confidence_score": 1.0
        }
        
        # Enhanced context with relevant memories
        enhanced_memories = await memory_context.enhance_context_with_query(message)
        
        # Prepare enhanced prompt
        enhanced_prompt = message
        if enhanced_memories:
            memory_context_text = "\n".join([
                f"Relevant context: {mem['content'][:200]}..."
                for mem in enhanced_memories[:3]
            ])
            enhanced_prompt = f"Context from previous conversations:\n{memory_context_text}\n\nUser: {message}"
        
        # Web search if requested
        search_results = []
        if message_data.enable_web_search:
            try:
                search_results = await web_search_engine.search(message, max_results=5)
                response_data["web_searches_performed"] = 1
                
                if search_results:
                    search_context = "\n".join([
                        f"Search result: {result.title} - {result.snippet}"
                        for result in search_results[:3]
                    ])
                    enhanced_prompt += f"\n\nWeb search results:\n{search_context}"
            except Exception as e:
                logging.error(f"Web search failed: {e}")
        
        # Deep research if requested
        research_report = None
        if message_data.enable_research:
            try:
                research_query = ResearchQuery(
                    original_query=message,
                    user_id=user_id,
                    session_id=session_id,
                    max_depth=message_data.research_depth
                )
                research_report = await research_engine.conduct_research(research_query)
                response_data["research_conducted"] = True
                response_data["research_sources"] = research_report.total_sources
                
                # Add research findings to prompt
                research_context = f"""
Research findings on "{message}":
{research_report.executive_summary}

Key findings:
{chr(10).join(f"- {finding}" for finding in research_report.key_findings[:3])}
"""
                enhanced_prompt += f"\n\nResearch findings:\n{research_context}"
                
            except Exception as e:
                logging.error(f"Research failed: {e}")
        
        # Generate AI response (placeholder - integrate with your model)
        ai_response = await generate_ai_response(
            enhanced_prompt, 
            memory_context, 
            search_results, 
            research_report
        )
        
        response_data["message"] = ai_response["content"]
        response_data["tokens_used"] = ai_response.get("tokens", {})
        response_data["confidence_score"] = ai_response.get("confidence", 1.0)
        
        # Store conversation in memory
        if message_data.store_memory:
            try:
                memory_id = await enhanced_session_manager.process_message_with_memory(
                    session_id=session_id,
                    user_message=message,
                    assistant_response=ai_response["content"],
                    tools_used=[]
                )
                response_data["memory_stored"] = True
                response_data["memory_ids"] = [str(memory_id.get("memory_id", ""))]
            except Exception as e:
                logging.error(f"Memory storage failed: {e}")
        
        # Create artifacts if requested and content suggests it
        if message_data.create_artifacts and should_create_artifact(message, ai_response["content"]):
            try:
                artifacts = await create_artifacts_from_response(
                    ai_response["content"], 
                    user_id, 
                    session_id
                )
                response_data["artifacts_created"] = artifacts
            except Exception as e:
                logging.error(f"Artifact creation failed: {e}")
        
        # Calculate processing time
        response_data["processing_time_ms"] = (asyncio.get_event_loop().time() - start_time) * 1000
        
        return ChatResponse(**response_data)
        
    except Exception as e:
        logging.error(f"Enhanced chat failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")


async def generate_ai_response(
    prompt: str, 
    memory_context, 
    search_results: List = None,
    research_report = None
) -> Dict[str, Any]:
    """
    Generate AI response using loaded model or API.
    This is a placeholder - integrate with your actual model loading system.
    """
    # This would integrate with your model_loader and generate actual responses
    # For now, return a structured response
    
    response_content = f"I understand you're asking about: {prompt[:100]}..."
    
    # Add search results if available
    if search_results:
        response_content += f"\n\nBased on web search, I found {len(search_results)} relevant sources."
    
    # Add research findings if available
    if research_report:
        response_content += f"\n\nFrom my research of {research_report.total_sources} sources: {research_report.executive_summary[:200]}..."
    
    # Add memory context acknowledgment
    if memory_context and memory_context.context_memories:
        response_content += "\n\nI'm taking into account our previous conversations."
    
    return {
        "content": response_content,
        "tokens": {"prompt": len(prompt.split()), "completion": len(response_content.split())},
        "confidence": 0.85
    }


def should_create_artifact(user_message: str, ai_response: str) -> bool:
    """Determine if artifacts should be created based on content"""
    artifact_triggers = [
        "create", "generate", "build", "make", "write code", "draw", "chart",
        "visualize", "design", "implement", "develop", "script", "function"
    ]
    
    combined_text = (user_message + " " + ai_response).lower()
    return any(trigger in combined_text for trigger in artifact_triggers)


async def create_artifacts_from_response(
    response_content: str, 
    user_id: str, 
    session_id: str
) -> List[Dict[str, Any]]:
    """Extract and create artifacts from AI response"""
    artifacts = []
    
    # Simple code block detection (would be more sophisticated in production)
    import re
    code_blocks = re.findall(r'```(\w+)?\n(.*?)\n```', response_content, re.DOTALL)
    
    for i, (language, code) in enumerate(code_blocks):
        if not language:
            language = "text"
        
        # Map language to artifact type
        artifact_type_map = {
            "python": ArtifactType.PYTHON,
            "javascript": ArtifactType.JAVASCRIPT,
            "html": ArtifactType.HTML,
            "json": ArtifactType.JSON,
            "markdown": ArtifactType.MARKDOWN
        }
        
        artifact_type = artifact_type_map.get(language.lower(), ArtifactType.PYTHON)
        
        try:
            artifact_content = await artifact_manager.create_artifact(
                name=f"Generated {language.title()} Code {i+1}",
                content=code.strip(),
                artifact_type=artifact_type,
                user_id=user_id,
                session_id=session_id,
                description=f"Code generated from conversation"
            )
            
            artifacts.append({
                "artifact_id": str(artifact_content.metadata.artifact_id),
                "name": artifact_content.metadata.name,
                "type": artifact_content.metadata.artifact_type.value,
                "status": artifact_content.metadata.status.value
            })
            
        except Exception as e:
            logging.error(f"Failed to create artifact: {e}")
    
    return artifacts


# Memory API Endpoints
@app.post("/api/memory/search")
async def search_memories(request: MemorySearchRequest):
    """Search user memories with semantic similarity"""
    try:
        memories = await memory_manager.retrieve_memories(
            user_id=request.user_id,
            query=request.query,
            memory_types=request.memory_types,
            importance_threshold=request.importance_threshold,
            limit=request.limit
        )
        
        return MemorySearchResponse(
            memories=memories,
            total_found=len(memories),
            query=request.query
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory search failed: {str(e)}")


@app.get("/api/memory/stats")
async def get_memory_stats(user_id: str):
    """Get comprehensive memory statistics for user"""
    try:
        stats = await memory_manager.get_user_memory_stats(user_id)
        return stats
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory stats: {str(e)}")


@app.get("/api/memory/dashboard")
async def get_memory_dashboard(user_id: str):
    """Get memory dashboard for user"""
    try:
        dashboard = await enhanced_session_manager.get_user_memory_dashboard(user_id)
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get memory dashboard: {str(e)}")


@app.post("/api/memory/export")
async def export_user_memories(request: MemoryExportRequest):
    """Export all user memories for backup"""
    try:
        export_data = await memory_manager.export_user_memories(request.user_id)
        
        # Return as downloadable JSON
        import io
        export_json = json.dumps(export_data, indent=2, default=str)
        return StreamingResponse(
            io.StringIO(export_json),
            media_type="application/json",
            headers={"Content-Disposition": f"attachment; filename=memories_{request.user_id}.json"}
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory export failed: {str(e)}")


# File Upload API Endpoints
@app.post("/api/files/upload")
async def upload_files(
    files: List[UploadFile] = File(...),
    user_id: str = Form(...),
    session_id: str = Form(...)
):
    """Upload and process files with embedding generation"""
    try:
        upload_results = []
        
        for file in files:
            # Read file data
            file_data = await file.read()
            
            # Upload and process
            metadata = await file_upload_manager.upload_file(
                file_data=file_data,
                filename=file.filename,
                user_id=user_id,
                session_id=session_id
            )
            
            upload_results.append({
                "file_id": str(metadata.file_id),
                "filename": metadata.filename,
                "file_type": metadata.file_type.value,
                "status": metadata.processing_status.value,
                "size_mb": metadata.file_size_mb,
                "text_extracted": metadata.text_length > 0,
                "security_clean": metadata.is_safe
            })
        
        return {"uploaded_files": upload_results}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")


@app.get("/api/files/search")
async def search_files(
    query: str,
    user_id: str,
    file_types: Optional[str] = None,
    limit: int = 10
):
    """Search uploaded files by content"""
    try:
        # Parse file types
        types = []
        if file_types:
            type_map = {
                "text": FileType.TEXT,
                "document": FileType.DOCUMENT,
                "image": FileType.IMAGE,
                "code": FileType.CODE,
                "data": FileType.DATA
            }
            types = [type_map[t] for t in file_types.split(",") if t in type_map]
        
        results = await file_upload_manager.search_files(
            query=query,
            user_id=user_id,
            file_types=types,
            limit=limit
        )
        
        return {
            "files": [
                {
                    "file_id": str(f.file_id),
                    "filename": f.filename,
                    "file_type": f.file_type.value,
                    "text_preview": f.extracted_text[:200] if f.extracted_text else "",
                    "uploaded_at": f.uploaded_at.isoformat()
                }
                for f in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File search failed: {str(e)}")


# Research API Endpoints
@app.post("/api/research")
async def conduct_research(request: ResearchRequest):
    """Conduct deep research with multiple sources"""
    try:
        research_query = ResearchQuery(
            original_query=request.query,
            user_id=request.user_id,
            session_id=request.session_id,
            max_depth=request.max_depth,
            max_sources=request.max_sources,
            source_types=request.source_types
        )
        
        # Conduct research
        report = await research_engine.conduct_research(research_query)
        
        return ResearchResponse(
            report_id=str(report.report_id),
            executive_summary=report.executive_summary,
            key_findings=report.key_findings,
            total_sources=report.total_sources,
            credible_sources=report.credible_sources,
            confidence_level=report.confidence_level,
            research_duration=report.research_duration,
            citations=report.citations
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")


# Artifact API Endpoints
@app.post("/api/artifacts", response_model=ArtifactResponse)
async def create_artifact(request: ArtifactCreationRequest):
    """Create new artifact with security validation"""
    try:
        artifact_content = await artifact_manager.create_artifact(
            name=request.name,
            content=request.content,
            artifact_type=request.artifact_type,
            user_id=request.user_id,
            session_id=request.session_id,
            description=request.description,
            tags=request.tags
        )
        
        return ArtifactResponse(
            artifact_id=str(artifact_content.metadata.artifact_id),
            name=artifact_content.metadata.name,
            artifact_type=artifact_content.metadata.artifact_type.value,
            status=artifact_content.metadata.status.value,
            security_level=artifact_content.metadata.security_level.value,
            created_at=artifact_content.metadata.created_at.isoformat(),
            preview_available=artifact_content.metadata.preview_available
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifact creation failed: {str(e)}")


@app.get("/api/artifacts")
async def list_artifacts(user_id: str, limit: int = 50, offset: int = 0):
    """List user artifacts"""
    try:
        artifacts = await artifact_manager.list_user_artifacts(user_id, limit, offset)
        
        return {
            "artifacts": [
                {
                    "artifact_id": str(a.artifact_id),
                    "name": a.name,
                    "artifact_type": a.artifact_type.value,
                    "status": a.status.value,
                    "created_at": a.created_at.isoformat(),
                    "preview_available": a.preview_available
                }
                for a in artifacts
            ]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list artifacts: {str(e)}")


@app.get("/api/artifacts/{artifact_id}/render")
async def render_artifact(artifact_id: str):
    """Render artifact to HTML"""
    try:
        rendered_html = await artifact_manager.render_artifact(artifact_id)
        return HTMLResponse(content=rendered_html)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Artifact rendering failed: {str(e)}")


# Enhanced Session API (inherits from original)
@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreationRequest):
    """Create session with enhanced memory context"""
    try:
        session_response, memory_context = await enhanced_session_manager.create_session_with_memory(request)
        return session_response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")


@app.get("/api/sessions")
async def list_sessions(user_id: Optional[str] = None):
    """List sessions with memory context"""
    try:
        sessions = await enhanced_session_manager.list_sessions(user_id)
        resource_util = enhanced_session_manager.session_manager.resource_manager.get_resource_utilization()
        
        return {
            "sessions": [
                {
                    "session_id": str(s.session_id),
                    "user_id": s.user_id,
                    "created_at": s.created_at.isoformat(),
                    "state": s.state.value,
                    "message_count": s.message_count,
                    "memory_enhanced": True
                }
                for s in sessions
            ],
            "total_count": len(sessions),
            "resource_usage": resource_util
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logging.info(f"Received signal {signum}, initiating graceful shutdown...")
    sys.exit(0)


def main():
    """Enhanced main entry point"""
    global system_config
    
    parser = argparse.ArgumentParser(description="Morpheus Chat - Enhanced AI Assistant")
    parser.add_argument("--config", "-c", help="Configuration file path")
    parser.add_argument("--debug", "-d", action="store_true", help="Enable debug mode")
    parser.add_argument("--port", "-p", type=int, default=8000, help="Server port")
    parser.add_argument("--host", default="0.0.0.0", help="Server host")
    
    args = parser.parse_args()
    
    # Load configuration
    system_config = load_configuration(args.config)
    
    # Setup logging
    if args.debug:
        system_config.setdefault("logging", {})["level"] = "DEBUG"
    
    setup_logging(system_config)
    
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Log startup banner
    logging.info("=" * 80)
    logging.info("🧠 MORPHEUS CHAT - Enhanced AI Assistant System")
    logging.info("=" * 80)
    logging.info(f"📍 Version: {system_config.get('system', {}).get('version', '1.0.0')}")
    logging.info(f"🌍 Environment: {system_config.get('system', {}).get('environment', 'development')}")
    logging.info(f"🚀 Server: {args.host}:{args.port}")
    logging.info(f"🔧 Debug mode: {args.debug}")
    logging.info("✨ Features: Memory, Research, Artifacts, File Processing")
    logging.info("=" * 80)
    
    # Run the enhanced server
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="debug" if args.debug else "info",
        access_log=True
    )


if __name__ == "__main__":
    main()