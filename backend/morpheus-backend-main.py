#!/usr/bin/env python3
"""
MORPHEUS CHAT - Production Backend Server
Sovereign AI Operating Environment with Plugin Architecture

This is the main application server that integrates all Morpheus components
into a unified, extensible platform with full local autonomy.
"""

import asyncio
import json
import logging
import os
import sys
import signal
import time
import importlib
import inspect
from pathlib import Path
from typing import Dict, List, Optional, Any, Type, Callable
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from uuid import UUID, uuid4

# FastAPI and WebSocket support
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Request, Response, UploadFile, File, Form, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import uvicorn

# Import Morpheus components
sys.path.append(str(Path(__file__).parent))

from core.session_manager import SessionManager
from memory_core import MemoryManager, MemoryConfiguration
from memory_integration import create_memory_enhanced_session_manager
from model_loader import ModelLoader
from container_runtime import ContainerRuntime
from prompt_manager import SystemPromptManager, PromptConfiguration
from file_upload_system import FileUploadManager
from artifact_system import ArtifactManager
from agent_collaboration_core import MultiAgentCollaborationHub
from web_search_research import create_research_system

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('morpheus_chat.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


# ============================================================================
# Plugin Architecture
# ============================================================================

class PluginMetadata(BaseModel):
    """Plugin metadata specification"""
    name: str
    version: str
    description: str
    author: str
    requires: List[str] = Field(default_factory=list)
    capabilities: List[str] = Field(default_factory=list)
    api_endpoints: List[str] = Field(default_factory=list)


class PluginInterface:
    """Base interface for all Morpheus plugins"""
    
    def __init__(self, app: FastAPI, morpheus: 'MorpheusCore'):
        self.app = app
        self.morpheus = morpheus
        self.logger = logging.getLogger(f"plugin.{self.__class__.__name__}")
    
    @property
    def metadata(self) -> PluginMetadata:
        """Plugin metadata - must be overridden"""
        raise NotImplementedError("Plugins must define metadata")
    
    async def initialize(self):
        """Initialize plugin - override for setup logic"""
        pass
    
    async def shutdown(self):
        """Cleanup plugin resources - override for cleanup logic"""
        pass
    
    def register_endpoints(self):
        """Register FastAPI endpoints - override to add routes"""
        pass
    
    def register_websocket_handlers(self) -> Dict[str, Callable]:
        """Register WebSocket message handlers - override to handle custom messages"""
        return {}


class PluginManager:
    """Dynamic plugin management system"""
    
    def __init__(self, app: FastAPI, morpheus: 'MorpheusCore'):
        self.app = app
        self.morpheus = morpheus
        self.plugins: Dict[str, PluginInterface] = {}
        self.plugin_dir = Path("plugins")
        self.plugin_dir.mkdir(exist_ok=True)
        
        # Add plugins directory to Python path
        sys.path.insert(0, str(self.plugin_dir))
        
        logger.info(f"Plugin manager initialized with directory: {self.plugin_dir}")
    
    async def load_plugins(self):
        """Dynamically load all plugins from the plugins directory"""
        for plugin_file in self.plugin_dir.glob("*.py"):
            if plugin_file.stem.startswith("_"):
                continue
            
            try:
                await self.load_plugin(plugin_file.stem)
            except Exception as e:
                logger.error(f"Failed to load plugin {plugin_file.name}: {e}")
    
    async def load_plugin(self, module_name: str) -> bool:
        """Load a single plugin by module name"""
        try:
            # Import the module
            module = importlib.import_module(module_name)
            
            # Find plugin class (must inherit from PluginInterface)
            plugin_class = None
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, PluginInterface) and obj != PluginInterface:
                    plugin_class = obj
                    break
            
            if not plugin_class:
                logger.warning(f"No plugin class found in {module_name}")
                return False
            
            # Instantiate plugin
            plugin = plugin_class(self.app, self.morpheus)
            
            # Validate metadata
            metadata = plugin.metadata
            
            # Check dependencies
            for req in metadata.requires:
                if req not in self.plugins:
                    logger.error(f"Plugin {metadata.name} requires {req} which is not loaded")
                    return False
            
            # Initialize plugin
            await plugin.initialize()
            
            # Register endpoints
            plugin.register_endpoints()
            
            # Store plugin
            self.plugins[metadata.name] = plugin
            
            logger.info(f"Loaded plugin: {metadata.name} v{metadata.version}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading plugin {module_name}: {e}")
            return False
    
    async def unload_plugin(self, plugin_name: str) -> bool:
        """Unload a plugin"""
        if plugin_name not in self.plugins:
            return False
        
        try:
            plugin = self.plugins[plugin_name]
            await plugin.shutdown()
            del self.plugins[plugin_name]
            
            logger.info(f"Unloaded plugin: {plugin_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error unloading plugin {plugin_name}: {e}")
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[PluginInterface]:
        """Get a loaded plugin by name"""
        return self.plugins.get(plugin_name)
    
    def list_plugins(self) -> List[PluginMetadata]:
        """List all loaded plugins"""
        return [plugin.metadata for plugin in self.plugins.values()]
    
    async def reload_plugin(self, plugin_name: str) -> bool:
        """Reload a plugin (unload and load again)"""
        if plugin_name in self.plugins:
            await self.unload_plugin(plugin_name)
        
        # Find module name
        for plugin_file in self.plugin_dir.glob("*.py"):
            module = importlib.import_module(plugin_file.stem)
            importlib.reload(module)  # Reload the module
            
            # Check if this module contains the plugin
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, PluginInterface) and obj != PluginInterface:
                    plugin = obj(self.app, self.morpheus)
                    if plugin.metadata.name == plugin_name:
                        return await self.load_plugin(plugin_file.stem)
        
        return False


# ============================================================================
# Core Morpheus System
# ============================================================================

class MorpheusCore:
    """Core Morpheus system integrating all components"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = self._load_config(config_path)
        self.components = {}
        self.active_connections: Dict[str, WebSocket] = {}
        self.user_sessions: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Initializing Morpheus Core...")
    
    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load system configuration"""
        # For now, return default config
        # In production, load from YAML/JSON file
        return {
            "system": {
                "name": "morpheus-chat",
                "version": "3.0.0",
                "environment": "production"
            },
            "session_management": {
                "max_sessions": 50,
                "session_timeout": 3600,
                "resources": {
                    "cpu_limit": "2.0",
                    "memory_limit": "4G",
                    "disk_limit": "2G"
                }
            },
            "memory": {
                "vector_db_path": "data/memory/vectors",
                "metadata_db_path": "data/memory/metadata.db",
                "max_memories_per_user": 50000
            },
            "models": {
                "model_registry": {
                    "local_models": {},
                    "api_models": {}
                }
            },
            "plugins": {
                "enabled": True,
                "auto_load": True
            }
        }
    
    async def initialize(self):
        """Initialize all Morpheus components"""
        try:
            # Initialize Docker client for container runtime
            import docker
            docker_client = docker.from_env()
            
            # Initialize core components
            self.components['container_runtime'] = ContainerRuntime(docker_client, self.config)
            self.components['session_manager'] = SessionManager(self.config)
            self.components['memory_manager'] = MemoryManager(MemoryConfiguration(**self.config['memory']))
            self.components['model_loader'] = ModelLoader(self.config)
            self.components['prompt_manager'] = SystemPromptManager(PromptConfiguration())
            self.components['file_manager'] = FileUploadManager("data/uploads")
            self.components['artifact_manager'] = ArtifactManager("data/artifacts")
            
            # Initialize memory manager
            await self.components['memory_manager'].initialize()
            
            # Create enhanced session manager with memory
            self.components['enhanced_session_manager'] = await create_memory_enhanced_session_manager(
                self.components['session_manager'],
                MemoryConfiguration(**self.config['memory'])
            )
            
            # Initialize multi-agent collaboration
            self.components['agent_hub'] = MultiAgentCollaborationHub(
                self.components['model_loader'],
                self.components['memory_manager']
            )
            await self.components['agent_hub'].initialize()
            
            # Initialize research system
            search_engine, research_engine = await create_research_system()
            self.components['search_engine'] = search_engine
            self.components['research_engine'] = research_engine
            
            # Start session manager
            await self.components['session_manager'].start()
            
            logger.info("Morpheus Core initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Morpheus Core: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown of all components"""
        logger.info("Shutting down Morpheus Core...")
        
        # Close all WebSocket connections
        for ws in list(self.active_connections.values()):
            await ws.close()
        
        # Shutdown components in reverse order
        if 'session_manager' in self.components:
            await self.components['session_manager'].shutdown()
        
        if 'memory_manager' in self.components:
            await self.components['memory_manager'].shutdown()
        
        if 'search_engine' in self.components:
            await self.components['search_engine'].__aexit__(None, None, None)
        
        logger.info("Morpheus Core shutdown complete")
    
    def get_component(self, component_name: str) -> Any:
        """Get a system component by name"""
        return self.components.get(component_name)


# ============================================================================
# API Models
# ============================================================================

class ChatMessage(BaseModel):
    """Chat message model"""
    role: str = Field(..., description="Message role: user, assistant, system")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    """Chat completion request"""
    message: str
    session_id: Optional[UUID] = None
    model_id: Optional[str] = None
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = 2048
    stream: Optional[bool] = True


class TaskDelegationRequest(BaseModel):
    """Multi-agent task delegation request"""
    task_description: str
    agents: List[Dict[str, str]]  # [{"agent_id": "alpha", "role": "researcher"}]
    priority: str = "medium"
    context: Optional[Dict[str, Any]] = None


class ResearchRequest(BaseModel):
    """Deep research request"""
    query: str
    depth: str = "standard"  # quick, standard, deep, expert
    sources: List[str] = Field(default_factory=lambda: ["academic", "news", "web"])
    max_sources: int = 20


class PluginAction(BaseModel):
    """Plugin action request"""
    plugin_name: str
    action: str
    parameters: Optional[Dict[str, Any]] = None


# ============================================================================
# FastAPI Application
# ============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting Morpheus Chat...")
    
    # Initialize core system
    morpheus = MorpheusCore()
    await morpheus.initialize()
    app.state.morpheus = morpheus
    
    # Initialize plugin manager
    plugin_manager = PluginManager(app, morpheus)
    app.state.plugin_manager = plugin_manager
    
    # Load plugins if enabled
    if morpheus.config.get('plugins', {}).get('auto_load', True):
        await plugin_manager.load_plugins()
    
    logger.info("Morpheus Chat started successfully")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Morpheus Chat...")
    
    # Unload all plugins
    for plugin_name in list(plugin_manager.plugins.keys()):
        await plugin_manager.unload_plugin(plugin_name)
    
    # Shutdown core
    await morpheus.shutdown()
    
    logger.info("Morpheus Chat shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Morpheus Chat",
    description="Sovereign AI Operating Environment",
    version="3.0.0",
    lifespan=lifespan
)

# Add CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================================
# Static File Serving
# ============================================================================

# Get the directory containing this script
BASE_DIR = Path(__file__).parent

# Mount static directories
app.mount("/js", StaticFiles(directory=BASE_DIR / "frontend" / "js"), name="js")
app.mount("/css", StaticFiles(directory=BASE_DIR / "frontend" / "css"), name="css")
app.mount("/assets", StaticFiles(directory=BASE_DIR / "frontend" / "assets"), name="assets")


@app.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the main Morpheus Chat interface"""
    html_path = BASE_DIR / "frontend" / "morpheus_chat.html"
    if not html_path.exists():
        # Fallback: look in the same directory as the script
        html_path = BASE_DIR / "morpheus_chat.html"
    
    if html_path.exists():
        with open(html_path, 'r', encoding='utf-8') as f:
            return HTMLResponse(content=f.read())
    else:
        return HTMLResponse(content="<h1>Morpheus Chat frontend not found. Please ensure morpheus_chat.html is in the frontend directory.</h1>", status_code=404)


# ============================================================================
# Core API Endpoints
# ============================================================================

@app.get("/api/health")
async def health_check():
    """System health check"""
    morpheus = app.state.morpheus
    
    return {
        "status": "healthy",
        "version": morpheus.config['system']['version'],
        "components": {
            "session_manager": morpheus.get_component('session_manager') is not None,
            "memory_manager": morpheus.get_component('memory_manager') is not None,
            "model_loader": morpheus.get_component('model_loader') is not None,
            "agent_hub": morpheus.get_component('agent_hub') is not None,
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }


@app.post("/api/sessions")
async def create_session(user_id: Optional[str] = None):
    """Create a new chat session"""
    morpheus = app.state.morpheus
    session_manager = morpheus.get_component('enhanced_session_manager')
    
    from schemas.session import SessionCreationRequest
    
    request = SessionCreationRequest(
        user_id=user_id or f"user_{uuid4()}",
        model_id="default",
        session_timeout_minutes=60
    )
    
    session_response, memory_context = await session_manager.create_session_with_memory(request)
    
    return {
        "session_id": str(session_response.session_id),
        "user_id": request.user_id,
        "status": session_response.state.value,
        "websocket_url": f"/ws/{session_response.session_id}"
    }


@app.get("/api/sessions/{session_id}")
async def get_session(session_id: UUID):
    """Get session information"""
    morpheus = app.state.morpheus
    session_manager = morpheus.get_component('session_manager')
    
    session = await session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {
        "session_id": str(session.session_id),
        "user_id": session.user_id,
        "state": session.state.value,
        "created_at": session.created_at.isoformat(),
        "last_activity": session.last_activity.isoformat(),
        "message_count": session.message_count
    }


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: UUID):
    """Delete a session"""
    morpheus = app.state.morpheus
    session_manager = morpheus.get_component('enhanced_session_manager')
    
    success = await session_manager.destroy_session_with_memory(session_id)
    
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return {"status": "deleted", "session_id": str(session_id)}


@app.post("/api/chat")
async def chat_completion(request: ChatRequest):
    """Non-streaming chat completion"""
    morpheus = app.state.morpheus
    
    # Get or create session
    if not request.session_id:
        session_response = await create_session()
        request.session_id = UUID(session_response["session_id"])
    
    # Process through agent hub
    agent_hub = morpheus.get_component('agent_hub')
    response = await agent_hub.process_user_input(
        request.message,
        {"session_id": str(request.session_id)}
    )
    
    return {
        "response": response,
        "session_id": str(request.session_id),
        "model_used": request.model_id or "default"
    }


@app.post("/api/delegate-task")
async def delegate_task(request: TaskDelegationRequest):
    """Delegate a task to multiple AI agents"""
    morpheus = app.state.morpheus
    agent_hub = morpheus.get_component('agent_hub')
    
    # Add agents if not already present
    for agent_config in request.agents:
        agent_id = uuid4()
        await agent_hub.add_agent({
            "name": agent_config.get("name", f"Agent-{agent_id}"),
            "model_id": agent_config.get("model_id", "default"),
            "capabilities": agent_config.get("capabilities", [])
        })
    
    # Process task through collaboration
    response = await agent_hub.process_user_input(
        request.task_description,
        request.context or {}
    )
    
    return {
        "status": "delegated",
        "task_id": str(uuid4()),
        "agents_assigned": len(request.agents),
        "initial_response": response
    }


@app.post("/api/research")
async def conduct_research(request: ResearchRequest):
    """Conduct deep research on a topic"""
    morpheus = app.state.morpheus
    research_engine = morpheus.get_component('research_engine')
    
    from web_search_research import ResearchQuery, SourceType
    
    # Map source preferences
    source_types = []
    if "academic" in request.sources:
        source_types.append(SourceType.ACADEMIC)
    if "news" in request.sources:
        source_types.append(SourceType.NEWS)
    if "web" in request.sources:
        source_types.extend([SourceType.TECHNICAL, SourceType.BLOG])
    
    # Create research query
    depth_map = {"quick": 1, "standard": 2, "deep": 3, "expert": 4}
    research_query = ResearchQuery(
        original_query=request.query,
        user_id="default_user",
        session_id=uuid4(),
        max_depth=depth_map.get(request.depth, 2),
        max_sources=request.max_sources,
        source_types=source_types
    )
    
    # Conduct research
    report = await research_engine.conduct_research(research_query)
    
    return {
        "query": request.query,
        "executive_summary": report.executive_summary,
        "key_findings": report.key_findings,
        "total_sources": report.total_sources,
        "credible_sources": report.credible_sources,
        "confidence_level": report.confidence_level,
        "citations": report.citations[:10]  # Limit citations in response
    }


@app.post("/api/upload")
async def upload_file(
    file: UploadFile = File(...),
    session_id: Optional[UUID] = Form(None),
    user_id: Optional[str] = Form(None)
):
    """Upload a file for processing"""
    morpheus = app.state.morpheus
    file_manager = morpheus.get_component('file_manager')
    
    # Read file content
    content = await file.read()
    
    # Process file
    metadata = await file_manager.upload_file(
        file_data=content,
        filename=file.filename,
        user_id=user_id or "default_user",
        session_id=session_id or uuid4()
    )
    
    return {
        "file_id": str(metadata.file_id),
        "filename": metadata.filename,
        "file_type": metadata.file_type.value,
        "file_size": metadata.file_size,
        "processing_status": metadata.processing_status.value,
        "extracted_text_length": metadata.text_length
    }


@app.post("/api/artifacts")
async def create_artifact(
    name: str = Form(...),
    content: str = Form(...),
    artifact_type: str = Form(...),
    user_id: Optional[str] = Form(None),
    session_id: Optional[UUID] = Form(None)
):
    """Create a new artifact"""
    morpheus = app.state.morpheus
    artifact_manager = morpheus.get_component('artifact_manager')
    
    from artifact_system import ArtifactType
    
    # Map artifact type
    type_map = {
        "markdown": ArtifactType.MARKDOWN,
        "html": ArtifactType.HTML,
        "javascript": ArtifactType.JAVASCRIPT,
        "python": ArtifactType.PYTHON,
        "json": ArtifactType.JSON,
        "svg": ArtifactType.SVG
    }
    
    artifact_content = await artifact_manager.create_artifact(
        name=name,
        content=content,
        artifact_type=type_map.get(artifact_type, ArtifactType.MARKDOWN),
        user_id=user_id or "default_user",
        session_id=session_id or uuid4()
    )
    
    return {
        "artifact_id": str(artifact_content.metadata.artifact_id),
        "name": artifact_content.metadata.name,
        "type": artifact_content.metadata.artifact_type.value,
        "status": artifact_content.metadata.status.value,
        "security_level": artifact_content.metadata.security_level.value
    }


@app.get("/api/artifacts/{artifact_id}/render")
async def render_artifact(artifact_id: UUID):
    """Render an artifact to HTML"""
    morpheus = app.state.morpheus
    artifact_manager = morpheus.get_component('artifact_manager')
    
    html_content = await artifact_manager.render_artifact(artifact_id)
    
    return HTMLResponse(content=html_content)


# ============================================================================
# Plugin Management Endpoints
# ============================================================================

@app.get("/api/plugins")
async def list_plugins():
    """List all loaded plugins"""
    plugin_manager = app.state.plugin_manager
    
    return {
        "plugins": [
            {
                "name": metadata.name,
                "version": metadata.version,
                "description": metadata.description,
                "author": metadata.author,
                "capabilities": metadata.capabilities
            }
            for metadata in plugin_manager.list_plugins()
        ]
    }


@app.post("/api/plugins/{plugin_name}/load")
async def load_plugin(plugin_name: str):
    """Load a plugin by name"""
    plugin_manager = app.state.plugin_manager
    
    success = await plugin_manager.load_plugin(plugin_name)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to load plugin {plugin_name}")
    
    return {"status": "loaded", "plugin": plugin_name}


@app.post("/api/plugins/{plugin_name}/unload")
async def unload_plugin(plugin_name: str):
    """Unload a plugin"""
    plugin_manager = app.state.plugin_manager
    
    success = await plugin_manager.unload_plugin(plugin_name)
    
    if not success:
        raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} not found")
    
    return {"status": "unloaded", "plugin": plugin_name}


@app.post("/api/plugins/{plugin_name}/reload")
async def reload_plugin(plugin_name: str):
    """Reload a plugin"""
    plugin_manager = app.state.plugin_manager
    
    success = await plugin_manager.reload_plugin(plugin_name)
    
    if not success:
        raise HTTPException(status_code=400, detail=f"Failed to reload plugin {plugin_name}")
    
    return {"status": "reloaded", "plugin": plugin_name}


@app.post("/api/plugins/{plugin_name}/action")
async def execute_plugin_action(plugin_name: str, action: PluginAction):
    """Execute a plugin action"""
    plugin_manager = app.state.plugin_manager
    
    plugin = plugin_manager.get_plugin(plugin_name)
    if not plugin:
        raise HTTPException(status_code=404, detail=f"Plugin {plugin_name} not found")
    
    # Execute action if plugin has the method
    if hasattr(plugin, action.action):
        method = getattr(plugin, action.action)
        if callable(method):
            if asyncio.iscoroutinefunction(method):
                result = await method(**(action.parameters or {}))
            else:
                result = method(**(action.parameters or {}))
            
            return {"status": "executed", "result": result}
    
    raise HTTPException(status_code=400, detail=f"Plugin {plugin_name} does not support action {action.action}")


# ============================================================================
# WebSocket Handler
# ============================================================================

class ConnectionManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket connected: {client_id}")
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket disconnected: {client_id}")
    
    async def send_message(self, client_id: str, message: dict):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)
    
    async def broadcast(self, message: dict, exclude: Optional[str] = None):
        for client_id, connection in self.active_connections.items():
            if client_id != exclude:
                await connection.send_json(message)


# Create connection manager
manager = ConnectionManager()


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time chat"""
    await manager.connect(websocket, session_id)
    morpheus = app.state.morpheus
    plugin_manager = app.state.plugin_manager
    
    # Get session
    session_manager = morpheus.get_component('enhanced_session_manager')
    session = await session_manager.session_manager.get_session(UUID(session_id))
    
    if not session:
        await websocket.close(code=1008, reason="Session not found")
        return
    
    # Collect WebSocket handlers from plugins
    ws_handlers = {}
    for plugin in plugin_manager.plugins.values():
        handlers = plugin.register_websocket_handlers()
        ws_handlers.update(handlers)
    
    try:
        while True:
            # Receive message
            data = await websocket.receive_json()
            message_type = data.get("type", "chat")
            
            # Check for plugin handlers first
            if message_type in ws_handlers:
                response = await ws_handlers[message_type](data, session)
                await manager.send_message(session_id, response)
                continue
            
            # Handle built-in message types
            if message_type == "chat":
                # Process chat message
                user_message = data.get("message", "")
                
                # Store in memory
                memory_context = session_manager.get_memory_context(UUID(session_id))
                if memory_context:
                    # Process through agent hub
                    agent_hub = morpheus.get_component('agent_hub')
                    assistant_response = await agent_hub.process_user_input(
                        user_message,
                        {"session_id": session_id}
                    )
                    
                    # Store conversation turn
                    await memory_context.store_conversation_turn(
                        user_message=user_message,
                        assistant_response=assistant_response
                    )
                    
                    # Send response
                    await manager.send_message(session_id, {
                        "type": "chat_response",
                        "message": assistant_response,
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
            
            elif message_type == "task_delegation":
                # Handle task delegation
                task_data = data.get("task", {})
                agent_hub = morpheus.get_component('agent_hub')
                
                # Process task
                response = await agent_hub.process_user_input(
                    task_data.get("description", ""),
                    task_data.get("context", {})
                )
                
                await manager.send_message(session_id, {
                    "type": "task_update",
                    "status": "delegated",
                    "response": response
                })
            
            elif message_type == "research":
                # Handle research request
                research_data = data.get("research", {})
                research_engine = morpheus.get_component('research_engine')
                
                # Send progress updates
                await manager.send_message(session_id, {
                    "type": "research_progress",
                    "status": "started",
                    "stage": "Initializing research..."
                })
                
                # Conduct research (simplified - in production, send progress updates)
                from web_search_research import ResearchQuery
                
                query = ResearchQuery(
                    original_query=research_data.get("query", ""),
                    user_id=session.user_id or "default",
                    session_id=UUID(session_id),
                    max_depth=2,
                    max_sources=10
                )
                
                report = await research_engine.conduct_research(query)
                
                await manager.send_message(session_id, {
                    "type": "research_complete",
                    "report": {
                        "summary": report.executive_summary,
                        "findings": report.key_findings,
                        "sources": report.total_sources
                    }
                })
            
            elif message_type == "ping":
                # Simple ping/pong for keepalive
                await manager.send_message(session_id, {"type": "pong"})
            
    except WebSocketDisconnect:
        manager.disconnect(session_id)
    except Exception as e:
        logger.error(f"WebSocket error for session {session_id}: {e}")
        manager.disconnect(session_id)


# ============================================================================
# LONPT Training Dashboard API (Mock Implementation)
# ============================================================================

@app.get("/api/lonpt/status")
async def lonpt_status():
    """Get LONPT system status"""
    # Mock implementation - replace with actual LONPT integration
    return {
        "jobs_by_state": {
            "training": 2,
            "completed": 15,
            "failed": 1
        },
        "scheduler_status": {
            "queued_jobs": 3
        },
        "hardware_profile": {
            "device_type": "GPU",
            "gpu_name": "NVIDIA RTX 3080",
            "total_memory_gb": 10.0
        },
        "hardware_utilization": {
            "cpu_percent": 45.5,
            "memory_percent": 60.2,
            "gpu_memory_percent": 75.8
        }
    }


@app.get("/api/lonpt/jobs")
async def lonpt_jobs():
    """List LONPT training jobs"""
    # Mock implementation
    return {
        "jobs": [
            {
                "job_id": "job_12345",
                "name": "Foundation Model v1",
                "state": "training",
                "progress_percentage": 65.3,
                "current_step": 65300,
                "total_steps": 100000
            }
        ]
    }


@app.post("/api/lonpt/jobs")
async def create_lonpt_job(job_config: Dict[str, Any]):
    """Create new LONPT training job"""
    # Mock implementation
    job_id = f"job_{int(time.time())}"
    
    return {
        "job_id": job_id,
        "status": "created",
        "message": f"Training job {job_config.get('model_name', 'unnamed')} created successfully"
    }


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    """Main entry point for Morpheus Chat"""
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Morpheus Chat - Sovereign AI Operating Environment")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    parser.add_argument("--config", default="config.yaml", help="Configuration file path")
    
    args = parser.parse_args()
    
    # Setup signal handlers for graceful shutdown
    def signal_handler(sig, frame):
        logger.info("Received interrupt signal, shutting down...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Print startup banner
    print("\n" + "="*60)
    print("🧠 MORPHEUS CHAT v3.0.0")
    print("Sovereign AI Operating Environment")
    print("="*60)
    print(f"\n📍 Starting server at http://{args.host}:{args.port}")
    print("📝 API documentation at http://{args.host}:{args.port}/docs")
    print("\n⚡ Press Ctrl+C to stop the server\n")
    
    # Run the application
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
        access_log=True
    )


if __name__ == "__main__":
    main()
