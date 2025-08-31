"""
MORPHEUS CHAT - Complete FastAPI Artifact System Integration
Production-ready API with real-time collaboration, VM integration, and terminal access
"""

import asyncio
import json
import logging
import time
import traceback
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

import psutil
from fastapi import (
    APIRouter, HTTPException, WebSocket, WebSocketDisconnect,
    Depends, BackgroundTasks, Request, Response, status, Query
)
from fastapi.middleware.base import BaseHTTPMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field, validator
from starlette.websockets import WebSocketState

from schemas.session import SessionID, UserID
from core.memory_core import MemoryManager
from core.security_layer import SecurityEnforcer
from .artifact_system import (
    ArtifactManager, ArtifactContent, ArtifactType, ArtifactStatus,
    SecurityLevel, ExecutionResult, ArtifactMetrics
)
from .morpheus_artifact_v2 import (
    EnhancedArtifactManager, CollaborationMode, CollaborationState,
    ModelComparison, TerminalSession
)

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ArtifactCreateRequest(BaseModel):
    """Request for creating artifacts"""
    name: str = Field(..., min_length=1, max_length=200)
    content: str = Field(..., min_length=1)
    artifact_type: ArtifactType
    user_id: UserID
    session_id: SessionID
    description: Optional[str] = None
    security_level: SecurityLevel = SecurityLevel.SANDBOXED
    collaboration_mode: CollaborationMode = CollaborationMode.SOLO
    working_directory: Optional[str] = None
    enable_vm: bool = False
    vm_config: Optional[Dict[str, Any]] = None


class ArtifactUpdateRequest(BaseModel):
    """Request for updating artifacts"""
    user_id: UserID
    content: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None


class ArtifactExecuteRequest(BaseModel):
    """Request for executing artifacts"""
    user_id: UserID
    use_vm: bool = False
    vm_config: Optional[Dict[str, Any]] = None
    timeout: Optional[int] = 300


class MultiModelRequest(BaseModel):
    """Request for multi-model comparison"""
    user_id: UserID
    models: List[str] = Field(..., min_items=2, max_items=5)
    original_prompt: str


class TerminalRequest(BaseModel):
    """Request for terminal session"""
    user_id: UserID
    session_id: SessionID
    working_directory: str = "/tmp"
    share_session: bool = False


class CollaborationJoinRequest(BaseModel):
    """Request to join collaboration"""
    user_id: UserID
    collaboration_mode: CollaborationMode = CollaborationMode.REAL_TIME


class ArtifactResponse(BaseModel):
    """Standard artifact response"""
    success: bool
    artifact: Optional[Dict[str, Any]] = None
    message: Optional[str] = None
    execution_result: Optional[Dict[str, Any]] = None
    collaboration_state: Optional[Dict[str, Any]] = None
    easter_eggs_unlocked: Optional[List[str]] = None
    progression_update: Optional[Dict[str, Any]] = None


class TerminalResponse(BaseModel):
    """Terminal session response"""
    success: bool
    artifact: Optional[Dict[str, Any]] = None
    terminal_session_id: Optional[str] = None
    websocket_endpoint: Optional[str] = None


class CollaborationResponse(BaseModel):
    """Collaboration response"""
    success: bool
    collaboration_state: Dict[str, Any]
    websocket_endpoint: str
    active_users: List[str]


class ProgressionResponse(BaseModel):
    """User progression response"""
    user_id: UserID
    level: int
    progression: Dict[str, Any]
    easter_eggs: List[str]
    next_unlocks: List[Dict[str, Any]]


# ============================================================================
# MIDDLEWARE
# ============================================================================

class ArtifactMonitoringMiddleware(BaseHTTPMiddleware):
    """Middleware for monitoring and SaaS differentiation messaging"""
    
    def __init__(self, app, artifact_manager: EnhancedArtifactManager):
        super().__init__(app)
        self.artifact_manager = artifact_manager
        self.request_count = 0
        self.total_execution_time = 0.0
        self.cost_savings = 0.0
    
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        self.request_count += 1
        
        # Check if this is an artifact-related request
        is_artifact_request = "/artifacts" in request.url.path
        
        # Add sovereignty messaging to headers
        response = await call_next(request)
        
        # Calculate metrics
        process_time = time.time() - start_time
        self.total_execution_time += process_time
        
        if is_artifact_request:
            # Calculate cost savings (what this would cost on SaaS)
            estimated_saas_cost = self._calculate_saas_cost(request, process_time)
            self.cost_savings += estimated_saas_cost
            
            # Add sovereignty headers
            response.headers["X-Morpheus-Cost"] = "$0.00"
            response.headers["X-Morpheus-SaaS-Cost-Saved"] = f"${estimated_saas_cost:.4f}"
            response.headers["X-Morpheus-Total-Saved"] = f"${self.cost_savings:.2f}"
            response.headers["X-Morpheus-Processing-Time"] = f"{process_time:.3f}s"
            response.headers["X-Morpheus-Sovereignty"] = "COMPLETE"
            response.headers["X-Morpheus-Restrictions"] = "NONE"
            
            # Performance messaging
            if process_time < 0.1:
                response.headers["X-Morpheus-Performance"] = "BLAZING_FAST"
            elif process_time < 0.5:
                response.headers["X-Morpheus-Performance"] = "FAST"
            else:
                response.headers["X-Morpheus-Performance"] = "NORMAL"
        
        return response
    
    def _calculate_saas_cost(self, request: Request, process_time: float) -> float:
        """Calculate estimated SaaS cost for this request"""
        base_cost = 0.001  # Base API call cost
        
        if "execute" in request.url.path:
            # Execution would cost more on SaaS
            base_cost += 0.01 + (process_time * 0.001)
        
        if "collaborate" in request.url.path:
            # Collaboration features cost extra
            base_cost += 0.005
        
        if "terminal" in request.url.path:
            # Terminal access is premium
            base_cost += 0.02
        
        return base_cost


# ============================================================================
# WEBSOCKET CONNECTION MANAGER
# ============================================================================

class WebSocketManager:
    """WebSocket connection manager for real-time features"""
    
    def __init__(self):
        self.connections: Dict[str, Dict[str, WebSocket]] = {}  # session_id -> {user_id: websocket}
        self.user_sessions: Dict[str, Set[str]] = {}  # user_id -> {session_ids}
    
    async def connect(self, websocket: WebSocket, session_id: str, user_id: str):
        """Connect user to session"""
        await websocket.accept()
        
        if session_id not in self.connections:
            self.connections[session_id] = {}
        
        self.connections[session_id][user_id] = websocket
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = set()
        self.user_sessions[user_id].add(session_id)
        
        logger.info(f"WebSocket connected: {user_id} -> {session_id}")
    
    async def disconnect(self, session_id: str, user_id: str):
        """Disconnect user from session"""
        if session_id in self.connections:
            self.connections[session_id].pop(user_id, None)
            
            if not self.connections[session_id]:
                self.connections.pop(session_id, None)
        
        if user_id in self.user_sessions:
            self.user_sessions[user_id].discard(session_id)
            if not self.user_sessions[user_id]:
                self.user_sessions.pop(user_id, None)
        
        logger.info(f"WebSocket disconnected: {user_id} -> {session_id}")
    
    async def send_to_user(self, session_id: str, user_id: str, message: Dict):
        """Send message to specific user"""
        if session_id in self.connections and user_id in self.connections[session_id]:
            websocket = self.connections[session_id][user_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to send message to {user_id}: {e}")
                await self.disconnect(session_id, user_id)
    
    async def broadcast_to_session(self, session_id: str, message: Dict, exclude_user: str = None):
        """Broadcast message to all users in session"""
        if session_id not in self.connections:
            return
        
        for user_id, websocket in self.connections[session_id].items():
            if exclude_user and user_id == exclude_user:
                continue
            
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Failed to broadcast to {user_id}: {e}")
                await self.disconnect(session_id, user_id)


# ============================================================================
# API ROUTER CREATION
# ============================================================================

def create_artifact_router(
    artifact_manager: EnhancedArtifactManager,
    memory_manager: MemoryManager,
    security_enforcer: SecurityEnforcer
) -> APIRouter:
    """Create complete artifact API router"""
    
    router = APIRouter(prefix="/artifacts", tags=["Revolutionary Artifacts"])
    websocket_manager = WebSocketManager()
    
    # ========================================================================
    # CORE ARTIFACT OPERATIONS
    # ========================================================================
    
    @router.post("/create", response_model=ArtifactResponse)
    async def create_artifact(request: ArtifactCreateRequest, background_tasks: BackgroundTasks):
        """Create new artifact with full feature support"""
        try:
            # Create artifact
            artifact = await artifact_manager.create_collaborative_artifact(
                name=request.name,
                content=request.content,
                artifact_type=request.artifact_type,
                user_id=request.user_id,
                session_id=request.session_id,
                description=request.description,
                security_level=request.security_level,
                collaboration_mode=request.collaboration_mode,
                working_directory=request.working_directory
            )
            
            # Get progression update
            progression = await artifact_manager.get_user_progression(request.user_id)
            
            # Update user progression
            if request.user_id in artifact_manager.user_progression:
                artifact_manager.user_progression[request.user_id]['artifacts_created'] += 1
            
            logger.info(f"Created artifact {artifact.metadata.artifact_id} for user {request.user_id}")
            
            return ArtifactResponse(
                success=True,
                artifact=artifact.dict(),
                message=f"Artifact '{request.name}' created successfully",
                progression_update=progression
            )
        
        except Exception as e:
            logger.error(f"Artifact creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create artifact: {str(e)}"
            )
    
    @router.post("/create-terminal", response_model=TerminalResponse)
    async def create_terminal_artifact(request: TerminalRequest):
        """Create terminal artifact with live session"""
        try:
            artifact, terminal_session_id = await artifact_manager.create_terminal_artifact(
                user_id=request.user_id,
                session_id=request.session_id,
                working_dir=request.working_directory
            )
            
            return TerminalResponse(
                success=True,
                artifact=artifact.dict(),
                terminal_session_id=terminal_session_id,
                websocket_endpoint=f"/artifacts/terminal/{terminal_session_id}/ws"
            )
        
        except Exception as e:
            logger.error(f"Terminal artifact creation failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to create terminal artifact: {str(e)}"
            )
    
    @router.put("/{artifact_id}", response_model=ArtifactResponse)
    async def update_artifact(artifact_id: str, request: ArtifactUpdateRequest):
        """Update existing artifact"""
        try:
            artifact_uuid = UUID(artifact_id)
            artifact = await artifact_manager.update_artifact(
                artifact_id=artifact_uuid,
                user_id=request.user_id,
                content=request.content,
                name=request.name,
                description=request.description
            )
            
            # Notify collaborators
            collaboration_state = await artifact_manager.get_collaboration_state(artifact_uuid)
            if collaboration_state and len(collaboration_state.active_users) > 1:
                await websocket_manager.broadcast_to_session(
                    str(artifact_uuid),
                    {
                        'type': 'artifact_updated',
                        'artifact_id': artifact_id,
                        'updated_by': request.user_id,
                        'version': artifact.metadata.version,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    },
                    exclude_user=request.user_id
                )
            
            return ArtifactResponse(
                success=True,
                artifact=artifact.dict(),
                message=f"Artifact updated to version {artifact.metadata.version}"
            )
        
        except ValueError as e:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(e))
        except Exception as e:
            logger.error(f"Artifact update failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to update artifact: {str(e)}"
            )
    
    @router.post("/{artifact_id}/execute", response_model=ArtifactResponse)
    async def execute_artifact(artifact_id: str, request: ArtifactExecuteRequest):
        """Execute artifact with optional VM support"""
        try:
            artifact_uuid = UUID(artifact_id)
            
            if request.use_vm:
                # Execute in VM
                result = await artifact_manager.execute_with_vm(
                    artifact_uuid, 
                    request.user_id,
                    request.vm_config
                )
            else:
                # Execute normally
                result = await artifact_manager.execute_artifact(artifact_uuid, request.user_id)
            
            # Update user progression
            if request.user_id in artifact_manager.user_progression:
                artifact_manager.user_progression[request.user_id]['code_executed'] += 1
                if request.use_vm:
                    artifact_manager.user_progression[request.user_id]['vm_sessions'] += 1
            
            return ArtifactResponse(
                success=True,
                execution_result=result.__dict__,
                message="Execution completed" if result.success else "Execution failed"
            )
        
        except Exception as e:
            logger.error(f"Artifact execution failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Execution failed: {str(e)}"
            )
    
    @router.get("/{artifact_id}", response_model=ArtifactResponse)
    async def get_artifact(artifact_id: str, user_id: UserID = Query(...)):
        """Get artifact by ID"""
        try:
            artifact_uuid = UUID(artifact_id)
            artifact = await artifact_manager.get_artifact(artifact_uuid)
            
            if not artifact:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
            
            # Check permissions
            if artifact.metadata.created_by != user_id and not artifact.metadata.is_public:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
            
            return ArtifactResponse(
                success=True,
                artifact=artifact.dict()
            )
        
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid artifact ID")
    
    @router.delete("/{artifact_id}")
    async def delete_artifact(artifact_id: str, user_id: UserID = Query(...)):
        """Delete artifact"""
        try:
            artifact_uuid = UUID(artifact_id)
            success = await artifact_manager.delete_artifact(artifact_uuid, user_id)
            
            if not success:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Artifact not found")
            
            return {"success": True, "message": "Artifact deleted successfully"}
        
        except PermissionError:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
        except Exception as e:
            logger.error(f"Artifact deletion failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to delete artifact: {str(e)}"
            )
    
    @router.get("/user/{user_id}")
    async def list_user_artifacts(user_id: UserID, limit: int = Query(50, ge=1, le=100)):
        """List user's artifacts"""
        try:
            artifacts = await artifact_manager.list_user_artifacts(user_id, limit)
            return {
                "success": True,
                "artifacts": [artifact.dict() for artifact in artifacts],
                "total": len(artifacts)
            }
        
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to list artifacts: {str(e)}"
            )
    
    # ========================================================================
    # COLLABORATION FEATURES
    # ========================================================================
    
    @router.post("/{artifact_id}/collaborate", response_model=CollaborationResponse)
    async def join_collaboration(artifact_id: str, request: CollaborationJoinRequest):
        """Join collaborative editing session"""
        try:
            artifact_uuid = UUID(artifact_id)
            
            # Join collaboration
            collaboration_state = await artifact_manager.collaboration_manager.join_collaboration(
                artifact_uuid, request.user_id, None
            )
            
            # Update user progression
            if request.user_id in artifact_manager.user_progression:
                artifact_manager.user_progression[request.user_id]['collaborations_joined'] += 1
            
            return CollaborationResponse(
                success=True,
                collaboration_state=collaboration_state.__dict__,
                websocket_endpoint=f"/artifacts/{artifact_id}/collaborate/ws",
                active_users=list(collaboration_state.active_users)
            )
        
        except Exception as e:
            logger.error(f"Failed to join collaboration: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to join collaboration: {str(e)}"
            )
    
    @router.post("/{artifact_id}/multi-model", response_model=ArtifactResponse)
    async def enable_multi_model(artifact_id: str, request: MultiModelRequest):
        """Enable multi-model comparison"""
        try:
            artifact_uuid = UUID(artifact_id)
            
            comparison = await artifact_manager.enable_multi_model_comparison(
                artifact_uuid, request.models, request.original_prompt
            )
            
            return ArtifactResponse(
                success=True,
                message="Multi-model comparison enabled",
                collaboration_state=comparison.__dict__
            )
        
        except Exception as e:
            logger.error(f"Multi-model comparison failed: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to enable multi-model comparison: {str(e)}"
            )
    
    # ========================================================================
    # PROGRESSION AND EASTER EGGS
    # ========================================================================
    
    @router.get("/progression/{user_id}", response_model=ProgressionResponse)
    async def get_user_progression(user_id: UserID):
        """Get user progression and unlocked features"""
        try:
            progression_data = await artifact_manager.get_user_progression(user_id)
            
            return ProgressionResponse(
                user_id=user_id,
                level=progression_data['progression']['level'],
                progression=progression_data['progression'],
                easter_eggs=progression_data['easter_eggs'],
                next_unlocks=progression_data['next_unlocks']
            )
        
        except Exception as e:
            logger.error(f"Failed to get progression: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to get progression: {str(e)}"
            )
    
    @router.get("/metrics/{artifact_id}")
    async def get_artifact_metrics(artifact_id: str, user_id: UserID = Query(...)):
        """Get detailed artifact metrics"""
        try:
            artifact_uuid = UUID(artifact_id)
            
            # Check permissions
            artifact = await artifact_manager.get_artifact(artifact_uuid)
            if not artifact or artifact.metadata.created_by != user_id:
                raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Access denied")
            
            metrics = await artifact_manager.get_artifact_metrics(artifact_uuid)
            if not metrics:
                raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Metrics not found")
            
            return {
                "success": True,
                "metrics": metrics.__dict__
            }
        
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Failed to get metrics: {str(e)}"
            )
    
    # ========================================================================
    # WEBSOCKET ENDPOINTS
    # ========================================================================
    
    @router.websocket("/{artifact_id}/collaborate/ws")
    async def collaboration_websocket(websocket: WebSocket, artifact_id: str, user_id: str = Query(...)):
        """WebSocket for real-time collaboration"""
        try:
            artifact_uuid = UUID(artifact_id)
            await websocket_manager.connect(websocket, artifact_id, user_id)
            
            # Join collaboration
            collaboration_state = await artifact_manager.collaboration_manager.join_collaboration(
                artifact_uuid, user_id, websocket
            )
            
            # Send initial state
            await websocket.send_text(json.dumps({
                'type': 'collaboration_joined',
                'artifact_id': artifact_id,
                'user_id': user_id,
                'active_users': list(collaboration_state.active_users),
                'version': collaboration_state.version
            }))
            
            try:
                while True:
                    # Receive collaboration operations
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    # Handle operation
                    result = await artifact_manager.collaboration_manager.handle_operation(
                        artifact_uuid, user_id, message
                    )
                    
                    # Send confirmation
                    await websocket.send_text(json.dumps({
                        'type': 'operation_result',
                        'success': result['success'],
                        'version': result['version']
                    }))
            
            except WebSocketDisconnect:
                pass
            
            finally:
                # Leave collaboration
                await artifact_manager.collaboration_manager.leave_collaboration(
                    artifact_uuid, user_id, websocket
                )
                await websocket_manager.disconnect(artifact_id, user_id)
        
        except Exception as e:
            logger.error(f"Collaboration WebSocket error: {e}")
            await websocket.close(code=1011, reason=str(e))
    
    @router.websocket("/terminal/{session_id}/ws")
    async def terminal_websocket(websocket: WebSocket, session_id: str, user_id: str = Query(...)):
        """WebSocket for terminal session"""
        try:
            await websocket_manager.connect(websocket, session_id, user_id)
            
            # Get terminal session
            terminal = await artifact_manager.collaboration_manager.get_terminal_session(session_id)
            if not terminal:
                await websocket.close(code=1011, reason="Terminal session not found")
                return
            
            # Subscribe to terminal output
            async def output_callback(output: str):
                try:
                    await websocket.send_text(json.dumps({
                        'type': 'terminal_output',
                        'content': output,
                        'timestamp': datetime.now(timezone.utc).isoformat()
                    }))
                except Exception as e:
                    logger.error(f"Failed to send terminal output: {e}")
            
            terminal.subscribe(output_callback)
            
            # Send terminal history
            history = terminal.get_history()
            await websocket.send_text(json.dumps({
                'type': 'terminal_history',
                'history': history
            }))
            
            try:
                while True:
                    # Receive terminal input
                    data = await websocket.receive_text()
                    message = json.loads(data)
                    
                    if message['type'] == 'terminal_input':
                        await terminal.send_input(message['content'])
                    
                    elif message['type'] == 'terminal_resize':
                        # Handle terminal resize
                        pass
            
            except WebSocketDisconnect:
                pass
            
            finally:
                terminal.unsubscribe(output_callback)
                await websocket_manager.disconnect(session_id, user_id)
        
        except Exception as e:
            logger.error(f"Terminal WebSocket error: {e}")
            await websocket.close(code=1011, reason=str(e))
    
    # ========================================================================
    # SYSTEM STATUS
    # ========================================================================
    
    @router.get("/system/status")
    async def system_status():
        """Get system status and capabilities"""
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent(interval=1)
        disk = psutil.disk_usage('/')
        
        return {
            "status": "🟢 SOVEREIGN",
            "message": "Complete AI sovereignty - No APIs, No Limits, No Fees",
            "capabilities": {
                "real_time_collaboration": True,
                "vm_integration": True,
                "terminal_access": True,
                "unlimited_execution": True,
                "multi_model_comparison": True,
                "persistent_storage": True,
                "cost": "$0.00"
            },
            "system_resources": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "disk_percent": disk.percent,
                "memory_total_gb": memory.total / 1024 / 1024 / 1024,
                "disk_total_gb": disk.total / 1024 / 1024 / 1024
            },
            "active_sessions": {
                "collaborations": len(artifact_manager.collaboration_manager.active_collaborations),
                "terminals": len(artifact_manager.collaboration_manager.terminal_sessions),
                "vm_sessions": len(artifact_manager.vm_integration.active_sessions),
                "websocket_connections": sum(len(conns) for conns in websocket_manager.connections.values())
            }
        }
    
    return router


# ============================================================================
# MAIN INTEGRATION FUNCTION
# ============================================================================

def integrate_with_morpheus_app(
    app,
    artifact_manager: EnhancedArtifactManager,
    memory_manager: MemoryManager,
    security_enforcer: SecurityEnforcer
):
    """Integrate artifact system with main Morpheus app"""
    
    # Add monitoring middleware
    app.add_middleware(ArtifactMonitoringMiddleware, artifact_manager=artifact_manager)
    
    # Create and include artifact router
    artifact_router = create_artifact_router(
        artifact_manager, memory_manager, security_enforcer
    )
    app.include_router(artifact_router, prefix="/api")
    
    # Add startup event
    @app.on_event("startup")
    async def startup_artifact_system():
        logger.info("🎨 Revolutionary Artifact System initialized!")
        logger.info("🚀 Features: Real-time collaboration, VM integration, Terminal access")
        logger.info("💰 Cost: $0.00 - Complete AI sovereignty achieved!")
    
    # Add shutdown event
    @app.on_event("shutdown")
    async def shutdown_artifact_system():
        await artifact_manager.cleanup()
        logger.info("🧹 Artifact system cleanup complete")
    
    logger.info("✅ Revolutionary Artifact System integrated with Morpheus Chat!")
    return app


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    """
    Example integration with your existing Morpheus Chat app
    """
    
    # Add this to your app_server.py:
    """
    from core.artifact_system_fastapi import integrate_with_morpheus_app
    from core.artifact_system import ArtifactManager
    from core.morpheus_artifact_v2 import EnhancedArtifactManager
    
    # In your lifespan function:
    app_state.artifact_manager = EnhancedArtifactManager(
        storage_dir="data/artifacts",
        execution_dir="data/execution",
        memory_manager=app_state.memory_manager,
        vm_manager=app_state.vm_manager
    )
    
    # Integrate with app:
    app = integrate_with_morpheus_app(
        app,
        app_state.artifact_manager,
        app_state.memory_manager,
        app_state.security_enforcer
    )
    """
    
    print("🎨 Revolutionary Artifact System FastAPI Integration Ready!")
    print("🚀 Features included:")
    print("   • Real-time collaborative editing")
    print("   • VM-integrated code execution") 
    print("   • Interactive terminal sessions")
    print("   • Multi-model comparison")
    print("   • Progressive feature unlocking")
    print("   • Complete AI sovereignty")
    print("   • $0.00 cost structure")