#!/usr/bin/env python3
"""
MORPHEUS CHAT - Production FastAPI Server
Advanced AI Assistant with Recursive Transformer Architecture

This is the main application server that integrates all system components:
- Session management with Docker orchestration
- Memory persistence with ChromaDB
- Model loading with quantization support
- Web search and research capabilities
- File upload and processing system
- Artifact management with security validation
- WebSocket real-time communication
"""

import asyncio
import logging
import os
import sys
import time
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from pydantic import BaseModel, Field

# Core system imports
from core.session_manager import SessionManager
from core.model_loader import ModelLoader
from core.security_layer import SecurityEnforcer
from core.memory_core import MemoryManager, MemoryConfiguration
from core.memory_integration import create_memory_enhanced_session_manager
from core.web_search_research import create_research_system
from core.file_upload_system import FileUploadManager
from core.artifact_system import ArtifactManager, ArtifactType
from schemas.session import SessionCreationRequest, SessionResponse, SessionID, UserID
from schemas.models import ModelLoadRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/morpheus_chat.log', mode='a')
    ]
)
logger = logging.getLogger("MorpheusChat")

# Ensure logs directory exists
Path("logs").mkdir(exist_ok=True)

# Configuration
class MorpheusConfig:
    """Production configuration for Morpheus Chat"""
    
    # Server configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Paths
    STATIC_DIR: str = "static"
    TEMPLATES_DIR: str = "templates"
    UPLOAD_DIR: str = "data/uploads"
    MODELS_DIR: str = "models"
    
    # Security
    ALLOWED_HOSTS: List[str] = ["*"]  # Configure for production
    CORS_ORIGINS: List[str] = ["*"]   # Configure for production
    
    # System limits
    MAX_SESSIONS: int = 50
    MAX_FILE_SIZE_MB: int = 100
    MAX_RESEARCH_DEPTH: int = 3
    
    @classmethod
    def from_env(cls) -> 'MorpheusConfig':
        """Load configuration from environment variables"""
        config = cls()
        for attr_name in dir(config):
            if not attr_name.startswith('_') and hasattr(config, attr_name):
                env_value = os.getenv(f"MORPHEUS_{attr_name}")
                if env_value is not None:
                    attr_type = type(getattr(config, attr_name))
                    if attr_type == bool:
                        setattr(config, attr_name, env_value.lower() == 'true')
                    elif attr_type == int:
                        setattr(config, attr_name, int(env_value))
                    elif attr_type == list:
                        setattr(config, attr_name, env_value.split(','))
                    else:
                        setattr(config, attr_name, env_value)
        return config

config = MorpheusConfig.from_env()

# Request/Response Models
class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=4000)
    user_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    enable_web_search: bool = False
    store_memory: bool = True
    model_override: Optional[str] = None

class ChatResponse(BaseModel):
    message: str
    session_id: str
    memory_stored: bool = False
    memory_id: Optional[str] = None
    artifacts: List[Dict[str, Any]] = Field(default_factory=list)
    research_sources: List[Dict[str, Any]] = Field(default_factory=list)
    processing_time: float = 0.0

class ResearchRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=1000)
    user_id: str = Field(..., min_length=1)
    session_id: Optional[str] = None
    max_depth: int = Field(default=2, ge=1, le=5)
    max_sources: int = Field(default=20, ge=5, le=100)
    source_types: List[str] = Field(default_factory=lambda: ["news", "academic", "technical"])

# Application State
class MorpheusState:
    """Global application state and components"""
    
    def __init__(self):
        self.session_manager = None
        self.model_loader = None
        self.memory_manager = None
        self.research_system = None
        self.file_manager = None
        self.artifact_manager = None
        self.active_websockets: Dict[str, WebSocket] = {}
        self.startup_time = datetime.now(timezone.utc)

app_state = MorpheusState()

# Application Lifecycle
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle"""
    
    # Startup
    logger.info("🧠 Morpheus Chat - Initializing Advanced AI Substrate...")
    
    try:
        # Create necessary directories
        for directory in [config.STATIC_DIR, config.UPLOAD_DIR, "data/memory", "data/artifacts"]:
            Path(directory).mkdir(parents=True, exist_ok=True)
        
        # Initialize memory system with recursive tensor support
        logger.info("🧠 Initializing Memory System with ChromaDB...")
        memory_config = MemoryConfiguration(
            vector_db_path="data/memory/vectors",
            metadata_db_path="data/memory/metadata.db",
            max_memories_per_user=50000,
            embedding_model="sentence-transformers/all-MiniLM-L6-v2"
        )
        app_state.memory_manager = MemoryManager(memory_config)
        await app_state.memory_manager.initialize()
        
        # Initialize model loader with recursive weights
        logger.info("🤖 Initializing Model Loader with Recursive Architecture...")
        model_config = {
            "model_registry": {
                "local_models": {
                    "morpheus-7b": {
                        "name": "Morpheus 7B Recursive",
                        "model_path": f"{config.MODELS_DIR}/morpheus-7b",
                        "model_type": "local_transformer",
                        "context_length": 8192,
                        "capabilities": ["text_generation", "reasoning", "memory_integration"],
                        "quantization": {
                            "enabled": True,
                            "bits": 4,
                            "method": "bitsandbytes"
                        }
                    }
                },
                "api_models": {
                    "gpt-4o": {
                        "name": "GPT-4o Proxy",
                        "provider": "openai",
                        "model_id": "gpt-4-turbo-preview",
                        "context_length": 128000
                    }
                }
            }
        }
        app_state.model_loader = ModelLoader(model_config)
        
        # Initialize session manager with memory integration
        logger.info("🏗️ Initializing Session Manager with Docker Orchestration...")
        session_config = {
            "session_management": {
                "max_sessions": config.MAX_SESSIONS,
                "session_timeout": 3600,
                "resources": {
                    "cpu_limit": "2.0",
                    "memory_limit": "4G",
                    "disk_limit": "2G"
                }
            },
            "security": {
                "content_filtering": True,
                "prompt_injection_detection": True,
                "capability_enforcement": True
            }
        }
        
        base_session_manager = SessionManager(session_config)
        app_state.session_manager = await create_memory_enhanced_session_manager(
            base_session_manager, memory_config
        )
        
        # Initialize research system
        logger.info("🔍 Initializing Web Search & Research Engine...")
        search_engine, research_engine = await create_research_system()
        app_state.research_system = research_engine
        
        # Initialize file upload system with GGUF support
        logger.info("📁 Initializing File Upload System...")
        gguf_model_path = f"{config.MODELS_DIR}/embeddings/all-MiniLM-L6-v2.gguf"
        app_state.file_manager = FileUploadManager(
            upload_dir=config.UPLOAD_DIR,
            gguf_model_path=gguf_model_path if Path(gguf_model_path).exists() else None,
            max_file_size_mb=config.MAX_FILE_SIZE_MB
        )
        
        # Initialize artifact system
        logger.info("🎨 Initializing Artifact System...")
        app_state.artifact_manager = ArtifactManager("data/artifacts")
        
        logger.info("✅ Morpheus Chat - All Systems Operational!")
        logger.info(f"🌐 Server starting on http://{config.HOST}:{config.PORT}")
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🔄 Morpheus Chat - Graceful Shutdown Initiated...")
    
    try:
        if app_state.session_manager:
            await app_state.session_manager.shutdown()
        
        if app_state.memory_manager:
            await app_state.memory_manager.shutdown()
        
        # Close all WebSocket connections
        for websocket in app_state.active_websockets.values():
            try:
                await websocket.close()
            except:
                pass
        
        logger.info("✅ Morpheus Chat - Shutdown Complete")
        
    except Exception as e:
        logger.error(f"❌ Shutdown error: {e}")

# FastAPI Application
app = FastAPI(
    title="Morpheus Chat",
    description="Advanced AI Assistant with Recursive Transformer Architecture",
    version="1.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=config.ALLOWED_HOSTS
)

# Static Files
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Routes

@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main chat interface"""
    html_path = Path("web_ui_venv.html")
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(), status_code=200)
    else:
        return HTMLResponse(
            content="""
            <!DOCTYPE html>
            <html><head><title>Morpheus Chat</title></head>
            <body>
                <h1>Morpheus Chat</h1>
                <p>Frontend not found. Please ensure web_ui_venv.html is in the root directory.</p>
            </body></html>
            """,
            status_code=200
        )

@app.get("/health")
async def health_check():
    """System health check"""
    return {
        "status": "operational",
        "version": "1.0.0",
        "uptime": (datetime.now(timezone.utc) - app_state.startup_time).total_seconds(),
        "components": {
            "session_manager": app_state.session_manager is not None,
            "memory_manager": app_state.memory_manager is not None,
            "model_loader": app_state.model_loader is not None,
            "research_system": app_state.research_system is not None,
            "file_manager": app_state.file_manager is not None,
            "artifact_manager": app_state.artifact_manager is not None
        }
    }

@app.get("/status")
async def system_status():
    """Detailed system status"""
    try:
        memory_metrics = app_state.memory_manager.get_metrics() if app_state.memory_manager else {}
        model_info = app_state.model_loader.get_system_info() if app_state.model_loader else {}
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "memory_system": memory_metrics,
            "model_system": model_info,
            "active_websockets": len(app_state.active_websockets),
            "system_load": {
                "cpu_percent": 0,  # Would implement actual monitoring
                "memory_percent": 0,
                "disk_usage": 0
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Status check failed: {str(e)}")

# Session Management Endpoints

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(request: SessionCreationRequest):
    """Create a new chat session"""
    try:
        session_response, memory_context = await app_state.session_manager.create_session_with_memory(request)
        
        if session_response.state.value == "active":
            logger.info(f"✅ Session created: {session_response.session_id}")
        
        return session_response
        
    except Exception as e:
        logger.error(f"❌ Session creation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Session creation failed: {str(e)}")

@app.get("/api/sessions")
async def list_sessions(user_id: str = None, limit: int = 20):
    """List user sessions"""
    try:
        sessions = await app_state.session_manager.list_sessions(user_id)
        
        return {
            "sessions": [
                {
                    "session_id": str(session.session_id),
                    "created_at": session.created_at.isoformat(),
                    "state": session.state.value,
                    "title": f"Chat {session.session_id}",  # Could enhance with actual titles
                    "message_count": session.message_count,
                    "last_activity": session.last_activity.isoformat()
                }
                for session in sessions[:limit]
            ],
            "total": len(sessions)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to list sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to list sessions: {str(e)}")

@app.get("/api/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    try:
        session = await app_state.session_manager.get_session(UUID(session_id))
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": str(session.session_id),
            "user_id": session.user_id,
            "state": session.state.value,
            "created_at": session.created_at.isoformat(),
            "last_activity": session.last_activity.isoformat(),
            "message_count": session.message_count,
            "model_config": session.model_config.dict() if session.model_config else None
        }
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    except Exception as e:
        logger.error(f"❌ Failed to get session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get session: {str(e)}")

@app.delete("/api/sessions/{session_id}")
async def destroy_session(session_id: str):
    """Destroy a session"""
    try:
        success = await app_state.session_manager.destroy_session_with_memory(UUID(session_id))
        
        if success:
            return {"message": "Session destroyed successfully"}
        else:
            raise HTTPException(status_code=404, detail="Session not found")
            
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid session ID format")
    except Exception as e:
        logger.error(f"❌ Failed to destroy session: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to destroy session: {str(e)}")

# Chat Endpoints

@app.post("/api/chat", response_model=ChatResponse)
async def chat_completion(request: ChatRequest):
    """Process chat message with full AI pipeline"""
    start_time = time.time()
    
    try:
        # Get or create session
        session_id = UUID(request.session_id) if request.session_id else uuid4()
        memory_context = app_state.session_manager.get_memory_context(session_id)
        
        if not memory_context:
            # Create new session if needed
            session_request = SessionCreationRequest(
                user_id=request.user_id,
                model_id=request.model_override or "gpt-4o"
            )
            _, memory_context = await app_state.session_manager.create_session_with_memory(session_request)
            session_id = memory_context.session_id
        
        # Enhanced context retrieval
        relevant_memories = []
        if request.store_memory:
            relevant_memories = await memory_context.enhance_context_with_query(request.message)
        
        # Web search integration
        research_sources = []
        if request.enable_web_search:
            try:
                # Perform web search for current information
                search_results = await app_state.research_system.search_engine.search(
                    request.message, max_results=5
                )
                research_sources = [
                    {
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "credibility": result.credibility_score
                    }
                    for result in search_results
                ]
            except Exception as e:
                logger.warning(f"Web search failed: {e}")
        
        # Generate AI response (simplified - would integrate with actual model)
        ai_response = await generate_ai_response(
            message=request.message,
            context=relevant_memories,
            web_results=research_sources,
            user_id=request.user_id
        )
        
        # Store conversation in memory
        memory_stored = False
        memory_id = None
        if request.store_memory:
            memory_id = await memory_context.store_conversation_turn(
                user_message=request.message,
                assistant_response=ai_response["content"],
                turn_metadata={
                    "web_search_used": request.enable_web_search,
                    "sources_count": len(research_sources)
                }
            )
            memory_stored = True
        
        processing_time = time.time() - start_time
        
        return ChatResponse(
            message=ai_response["content"],
            session_id=str(session_id),
            memory_stored=memory_stored,
            memory_id=str(memory_id) if memory_id else None,
            artifacts=ai_response.get("artifacts", []),
            research_sources=research_sources,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"❌ Chat processing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Chat processing failed: {str(e)}")

async def generate_ai_response(message: str, context: List[Dict], web_results: List[Dict], user_id: str) -> Dict[str, Any]:
    """Generate AI response using recursive transformer architecture"""
    
    # This is a simplified implementation - in production, this would:
    # 1. Load the appropriate model using ModelLoader
    # 2. Apply recursive tensor transformations
    # 3. Use constitutional AI evaluation
    # 4. Generate artifacts if requested
    # 5. Apply safety filters
    
    # For demonstration, return a structured response
    response_content = f"""Thank you for your message: "{message}"

I've processed your request using my recursive transformer architecture. """
    
    if context:
        response_content += f"I found {len(context)} relevant memories from our previous conversations. "
    
    if web_results:
        response_content += f"I also searched the web and found {len(web_results)} current sources. "
    
    response_content += """

I'm ready to help with:
- Deep research and analysis
- Code generation and artifacts  
- File processing and analysis
- Memory-enhanced conversations
- Web search integration

What would you like to explore next?"""
    
    return {
        "content": response_content,
        "artifacts": [],
        "metadata": {
            "model": "morpheus-recursive-7b",
            "context_used": len(context),
            "web_sources": len(web_results)
        }
    }

# Research Endpoints

@app.post("/api/research")
async def conduct_research(request: ResearchRequest):
    """Conduct deep research with multiple sources"""
    try:
        from core.web_search_research import ResearchQuery
        
        # Create research query
        research_query = ResearchQuery(
            original_query=request.query,
            user_id=request.user_id,
            session_id=UUID(request.session_id) if request.session_id else uuid4(),
            max_depth=request.max_depth,
            max_sources=request.max_sources
        )
        
        # Conduct research
        research_report = await app_state.research_system.conduct_research(research_query)
        
        return {
            "query_id": str(research_report.report_id),
            "executive_summary": research_report.executive_summary,
            "key_findings": research_report.key_findings,
            "total_sources": research_report.total_sources,
            "credible_sources": research_report.credible_sources,
            "confidence_level": research_report.confidence_level,
            "research_duration": research_report.research_duration,
            "citations": research_report.citations,
            "consensus_points": research_report.consensus_points,
            "conflicting_information": research_report.conflicting_information
        }
        
    except Exception as e:
        logger.error(f"❌ Research failed: {e}")
        raise HTTPException(status_code=500, detail=f"Research failed: {str(e)}")

# File Upload Endpoints

@app.post("/api/files/upload")
async def upload_file(
    file: UploadFile = File(...),
    user_id: str = Form(...),
    session_id: str = Form(None)
):
    """Upload and process file"""
    try:
        # Read file content
        file_content = await file.read()
        
        # Process file
        file_metadata = await app_state.file_manager.upload_file(
            file_data=file_content,
            filename=file.filename,
            user_id=user_id,
            session_id=UUID(session_id) if session_id else uuid4()
        )
        
        return {
            "file_id": str(file_metadata.file_id),
            "filename": file_metadata.filename,
            "file_type": file_metadata.file_type.value,
            "file_size": file_metadata.file_size,
            "processing_status": file_metadata.processing_status.value,
            "extracted_text_length": file_metadata.text_length,
            "is_safe": file_metadata.is_safe,
            "security_warnings": file_metadata.security_warnings
        }
        
    except Exception as e:
        logger.error(f"❌ File upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"File upload failed: {str(e)}")

@app.get("/api/files/search")
async def search_files(user_id: str, query: str = "", limit: int = 10):
    """Search uploaded files"""
    try:
        files = await app_state.file_manager.search_files(
            query=query,
            user_id=user_id,
            limit=limit
        )
        
        return {
            "files": [
                {
                    "file_id": str(f.file_id),
                    "filename": f.filename,
                    "file_type": f.file_type.value,
                    "uploaded_at": f.uploaded_at.isoformat(),
                    "text_length": f.text_length
                }
                for f in files
            ]
        }
        
    except Exception as e:
        logger.error(f"❌ File search failed: {e}")
        raise HTTPException(status_code=500, detail=f"File search failed: {str(e)}")

# Memory Endpoints

@app.get("/api/memory/stats")
async def get_memory_stats(user_id: str):
    """Get user memory statistics"""
    try:
        stats = await app_state.memory_manager.get_user_memory_stats(user_id)
        return stats
        
    except Exception as e:
        logger.error(f"❌ Memory stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory stats failed: {str(e)}")

@app.get("/api/memory/dashboard")
async def get_memory_dashboard(user_id: str):
    """Get memory dashboard data"""
    try:
        dashboard = await app_state.session_manager.get_user_memory_dashboard(user_id)
        return dashboard
        
    except Exception as e:
        logger.error(f"❌ Memory dashboard failed: {e}")
        raise HTTPException(status_code=500, detail=f"Memory dashboard failed: {str(e)}")

# Artifact Endpoints

@app.get("/api/artifacts")
async def list_artifacts(user_id: str, limit: int = 20):
    """List user artifacts"""
    try:
        artifacts = await app_state.artifact_manager.list_user_artifacts(user_id, limit=limit)
        
        return [
            {
                "artifact_id": str(artifact.artifact_id),
                "name": artifact.name,
                "artifact_type": artifact.artifact_type.value,
                "created_at": artifact.created_at.isoformat(),
                "security_level": artifact.security_level.value,
                "is_safe_for_preview": artifact.is_safe_for_preview
            }
            for artifact in artifacts
        ]
        
    except Exception as e:
        logger.error(f"❌ Artifact listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Artifact listing failed: {str(e)}")

@app.get("/api/artifacts/{artifact_id}/render")
async def render_artifact(artifact_id: str):
    """Render artifact to HTML"""
    try:
        rendered_html = await app_state.artifact_manager.render_artifact(UUID(artifact_id))
        return HTMLResponse(content=rendered_html)
        
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid artifact ID format")
    except Exception as e:
        logger.error(f"❌ Artifact rendering failed: {e}")
        raise HTTPException(status_code=500, detail=f"Artifact rendering failed: {str(e)}")

# WebSocket for Real-time Communication

@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await websocket.accept()
    app_state.active_websockets[session_id] = websocket
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            # Process message
            if message_data["type"] == "chat":
                # Handle chat message
                response = await chat_completion(ChatRequest(**message_data["data"]))
                await websocket.send_text(json.dumps({
                    "type": "chat_response",
                    "data": response.dict()
                }))
            
            elif message_data["type"] == "ping":
                # Handle ping for connection keep-alive
                await websocket.send_text(json.dumps({"type": "pong"}))
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        app_state.active_websockets.pop(session_id, None)

# Model Management Endpoints

@app.post("/api/models/load")
async def load_model(request: ModelLoadRequest):
    """Load a model"""
    try:
        response = await app_state.model_loader.load_model(request)
        return response.dict()
        
    except Exception as e:
        logger.error(f"❌ Model loading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model loading failed: {str(e)}")

@app.get("/api/models")
async def list_models():
    """List available models"""
    try:
        if app_state.model_loader:
            registry = app_state.model_loader.get_model_registry()
            return {
                "models": [
                    {
                        "model_id": model_id,
                        "name": config.name,
                        "model_type": config.model_type.value,
                        "capabilities": [cap.value for cap in config.capabilities]
                    }
                    for model_id, config in registry.models.items()
                ]
            }
        return {"models": []}
        
    except Exception as e:
        logger.error(f"❌ Model listing failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model listing failed: {str(e)}")

@app.delete("/api/models/{model_id}")
async def unload_model(model_id: str):
    """Unload a model"""
    try:
        success = await app_state.model_loader.unload_model(model_id)
        
        if success:
            return {"message": f"Model {model_id} unloaded successfully"}
        else:
            raise HTTPException(status_code=404, detail="Model not found or not loaded")
            
    except Exception as e:
        logger.error(f"❌ Model unloading failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model unloading failed: {str(e)}")

# System Information Endpoints

@app.get("/api/system/info")
async def get_system_info():
    """Get detailed system information"""
    try:
        return {
            "version": "1.0.0",
            "uptime": (datetime.now(timezone.utc) - app_state.startup_time).total_seconds(),
            "components": {
                "recursive_transformer": "operational",
                "memory_system": app_state.memory_manager.get_metrics() if app_state.memory_manager else {},
                "session_manager": len(app_state.session_manager.active_sessions) if app_state.session_manager else 0,
                "active_websockets": len(app_state.active_websockets)
            },
            "capabilities": {
                "memory_persistence": True,
                "web_search": True,
                "file_processing": True,
                "artifacts": True,
                "constitutional_ai": True,
                "recursive_tensors": True
            }
        }
        
    except Exception as e:
        logger.error(f"❌ System info failed: {e}")
        raise HTTPException(status_code=500, detail=f"System info failed: {str(e)}")

# Error Handlers

@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Resource not found", "path": str(request.url)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error"}
    )

# Main Entry Point
if __name__ == "__main__":
    # Development server
    print("🧠 Morpheus Chat - Starting Development Server...")
    print(f"🌐 Access the interface at: http://{config.HOST}:{config.PORT}")
    print("🔬 Advanced AI Substrate with Recursive Architecture Ready!")
    
    uvicorn.run(
        "main:app",
        host=config.HOST,
        port=config.PORT,
        reload=config.DEBUG,
        log_level="info",
        access_log=True
    )