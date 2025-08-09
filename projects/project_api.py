"""
MORPHEUS CHAT - Project API Routes
FastAPI endpoints for project management system

Revolutionary Features:
- Complete project lifecycle management via API
- Real-time WebSocket updates for project activities
- File upload and processing endpoints
- Collaboration and automation controls
- VM management and artifact execution
- Knowledge base search and synthesis
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect, UploadFile, File, Depends, BackgroundTasks, Query, Body
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field, validator
from starlette.websockets import WebSocketState

from .project_core import ProjectManager, ProjectMetadata, ProjectSpecs, ProjectType
from .project_collaboration import ProjectCollaborationManager
from .project_automation import ProjectAutomationEngine, AutomationRule
from .project_artifacts import ProjectArtifactManager
from .project_knowledge import ProjectKnowledgeBase

logger = logging.getLogger(__name__)


# ============================================================================
# REQUEST/RESPONSE MODELS
# ============================================================================

class ProjectCreateRequest(BaseModel):
    """Request to create new project"""
    name: str = Field(..., min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    project_type: Optional[ProjectType] = None
    specs: Optional[ProjectSpecs] = None
    user_id: str = Field(..., description="User creating the project")
    enable_collaboration: bool = Field(default=False)
    enable_automation: bool = Field(default=True)


class ProjectUpdateRequest(BaseModel):
    """Request to update project"""
    name: Optional[str] = Field(None, min_length=1, max_length=100)
    description: Optional[str] = Field(None, max_length=500)
    enable_collaboration: Optional[bool] = None
    enable_automation: Optional[bool] = None


class ProjectResponse(BaseModel):
    """Project information response"""
    project_id: str
    name: str
    description: Optional[str]
    project_type: str
    status: str
    specs: Dict[str, Any]
    vm_connection: Dict[str, Any]
    created_at: str
    last_active: str
    
    # Enhanced project info
    files_processed: int
    knowledge_items: int
    collaboration_enabled: bool
    automation_enabled: bool
    artifacts_count: int


class FileUploadResponse(BaseModel):
    """File upload response"""
    file_id: str
    filename: str
    size_bytes: int
    processing_status: str
    analysis_completed: bool
    knowledge_items_created: int


class KnowledgeSearchRequest(BaseModel):
    """Knowledge search request"""
    query: str = Field(..., min_length=1)
    limit: int = Field(default=10, ge=1, le=50)
    item_types: Optional[List[str]] = None


class CollaborationTaskRequest(BaseModel):
    """Collaboration task request"""
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    task_type: str = Field(default="general")
    project_files: Optional[List[str]] = None


class AutomationRuleRequest(BaseModel):
    """Automation rule creation request"""
    name: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    trigger_type: str
    trigger_config: Dict[str, Any]
    action_type: str
    action_config: Dict[str, Any]
    cron_expression: Optional[str] = None
    enabled: bool = Field(default=True)


class ArtifactCreateRequest(BaseModel):
    """Artifact creation request"""
    title: str = Field(..., min_length=1)
    content: str = Field(..., min_length=1)
    artifact_type: str
    enhance_with_project: bool = Field(default=True)
    enable_collaboration: bool = Field(default=False)


# ============================================================================
# WEBSOCKET MANAGER
# ============================================================================

class ProjectWebSocketManager:
    """Manages WebSocket connections for real-time project updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.project_subscribers: Dict[str, Set[str]] = {}
    
    async def connect(self, websocket: WebSocket, project_id: str, user_id: str):
        """Connect user to project WebSocket"""
        await websocket.accept()
        
        if project_id not in self.active_connections:
            self.active_connections[project_id] = []
            self.project_subscribers[project_id] = set()
        
        self.active_connections[project_id].append(websocket)
        self.project_subscribers[project_id].add(user_id)
        
        # Send initial project status
        await self.send_to_connection(websocket, {
            "type": "connection_established",
            "project_id": project_id,
            "user_id": user_id,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def disconnect(self, websocket: WebSocket, project_id: str, user_id: str):
        """Disconnect user from project WebSocket"""
        if project_id in self.active_connections:
            try:
                self.active_connections[project_id].remove(websocket)
                self.project_subscribers[project_id].discard(user_id)
                
                # Cleanup empty project connections
                if not self.active_connections[project_id]:
                    del self.active_connections[project_id]
                    del self.project_subscribers[project_id]
            except ValueError:
                pass
    
    async def send_to_connection(self, websocket: WebSocket, data: Dict[str, Any]):
        """Send data to specific WebSocket connection"""
        try:
            if websocket.client_state == WebSocketState.CONNECTED:
                await websocket.send_json(data)
        except:
            pass
    
    async def broadcast_to_project(self, project_id: str, data: Dict[str, Any]):
        """Broadcast data to all connections for a project"""
        if project_id in self.active_connections:
            disconnected = []
            
            for websocket in self.active_connections[project_id]:
                try:
                    if websocket.client_state == WebSocketState.CONNECTED:
                        await websocket.send_json(data)
                    else:
                        disconnected.append(websocket)
                except:
                    disconnected.append(websocket)
            
            # Remove disconnected websockets
            for ws in disconnected:
                try:
                    self.active_connections[project_id].remove(ws)
                except ValueError:
                    pass


# ============================================================================
# API ROUTER FACTORY
# ============================================================================

def create_project_api_routes(
    project_manager: ProjectManager,
    vm_manager,
    model_loader,
    memory_manager,
    artifact_manager
) -> APIRouter:
    """Create FastAPI router for project management"""
    
    router = APIRouter(prefix="/api/projects", tags=["projects"])
    websocket_manager = ProjectWebSocketManager()
    
    # Store active project components
    active_project_components: Dict[str, Dict[str, Any]] = {}
    
    # ========================================================================
    # PROJECT LIFECYCLE ENDPOINTS
    # ========================================================================
    
    @router.post("/create", response_model=ProjectResponse)
    async def create_project(request: ProjectCreateRequest, background_tasks: BackgroundTasks):
        """Create new project with VM and intelligence systems"""
        
        try:
            # Create project
            project = await project_manager.create_project(
                name=request.name,
                description=request.description,
                specs=request.specs,
                user_id=request.user_id
            )
            
            # Initialize project components
            project_components = await _initialize_project_components(
                project, request, background_tasks
            )
            
            active_project_components[str(project.project_id)] = project_components
            
            # Broadcast project creation
            await websocket_manager.broadcast_to_project(str(project.project_id), {
                "type": "project_created",
                "project_id": str(project.project_id),
                "name": project.name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return _format_project_response(project, project_components)
            
        except Exception as e:
            logger.error(f"Failed to create project: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{project_id}", response_model=ProjectResponse)
    async def get_project(project_id: str):
        """Get project information"""
        
        try:
            project_uuid = UUID(project_id)
            project = await project_manager.get_project(project_uuid)
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            project_components = active_project_components.get(project_id, {})
            return _format_project_response(project, project_components)
            
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid project ID")
        except Exception as e:
            logger.error(f"Failed to get project: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.put("/{project_id}", response_model=ProjectResponse)
    async def update_project(project_id: str, request: ProjectUpdateRequest):
        """Update project configuration"""
        
        try:
            project_uuid = UUID(project_id)
            project = await project_manager.get_project(project_uuid)
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            # Update project metadata
            if request.name:
                project.name = request.name
            if request.description is not None:
                project.description = request.description
            
            # Handle collaboration toggle
            if request.enable_collaboration is not None:
                project_components = active_project_components.get(project_id, {})
                if 'collaboration' in project_components:
                    collab_manager = project_components['collaboration']
                    if request.enable_collaboration and not project.collaboration_enabled:
                        await collab_manager.enable_collaboration()
                        project.collaboration_enabled = True
                    elif not request.enable_collaboration and project.collaboration_enabled:
                        await collab_manager.disable_collaboration()
                        project.collaboration_enabled = False
            
            # Broadcast project update
            await websocket_manager.broadcast_to_project(project_id, {
                "type": "project_updated",
                "project_id": project_id,
                "changes": request.dict(exclude_unset=True),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            project_components = active_project_components.get(project_id, {})
            return _format_project_response(project, project_components)
            
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid project ID")
        except Exception as e:
            logger.error(f"Failed to update project: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/{project_id}")
    async def delete_project(project_id: str):
        """Delete project and cleanup all resources"""
        
        try:
            project_uuid = UUID(project_id)
            
            # Cleanup project components
            if project_id in active_project_components:
                project_components = active_project_components[project_id]
                
                # Cleanup each component
                for component_name, component in project_components.items():
                    try:
                        if hasattr(component, 'cleanup'):
                            await component.cleanup()
                    except Exception as e:
                        logger.error(f"Error cleaning up {component_name}: {e}")
                
                del active_project_components[project_id]
            
            # Delete project
            success = await project_manager.delete_project(project_uuid)
            
            if not success:
                raise HTTPException(status_code=404, detail="Project not found")
            
            # Broadcast project deletion
            await websocket_manager.broadcast_to_project(project_id, {
                "type": "project_deleted",
                "project_id": project_id,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {"success": True, "message": "Project deleted successfully"}
            
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid project ID")
        except Exception as e:
            logger.error(f"Failed to delete project: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/", response_model=List[ProjectResponse])
    async def list_projects(user_id: Optional[str] = Query(None)):
        """List all projects for user"""
        
        try:
            projects = await project_manager.list_projects(user_id)
            
            response_list = []
            for project in projects:
                project_components = active_project_components.get(str(project.project_id), {})
                response_list.append(_format_project_response(project, project_components))
            
            return response_list
            
        except Exception as e:
            logger.error(f"Failed to list projects: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # FILE MANAGEMENT ENDPOINTS
    # ========================================================================
    
    @router.post("/{project_id}/files/upload", response_model=FileUploadResponse)
    async def upload_file(
        project_id: str,
        file: UploadFile = File(...),
        background_tasks: BackgroundTasks = BackgroundTasks()
    ):
        """Upload file to project for processing"""
        
        try:
            project_uuid = UUID(project_id)
            project = await project_manager.get_project(project_uuid)
            
            if not project:
                raise HTTPException(status_code=404, detail="Project not found")
            
            # Read file content
            file_content = await file.read()
            
            # Get project components
            project_components = active_project_components.get(project_id, {})
            project_vm = project_components.get('vm')
            project_intelligence = project_components.get('intelligence')
            
            if not project_vm:
                raise HTTPException(status_code=500, detail="Project VM not available")
            
            # Save file to project VM
            file_path = f"/project/files/{file.filename}"
            await project_vm.write_file_to_vm(project_vm, file_path, file_content.decode('utf-8', errors='ignore'))
            
            # Schedule background processing
            background_tasks.add_task(
                _process_uploaded_file,
                project_id, file.filename, project_intelligence, websocket_manager
            )
            
            # Broadcast file upload
            await websocket_manager.broadcast_to_project(project_id, {
                "type": "file_uploaded",
                "project_id": project_id,
                "filename": file.filename,
                "size": len(file_content),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return FileUploadResponse(
                file_id=str(uuid4()),
                filename=file.filename,
                size_bytes=len(file_content),
                processing_status="queued",
                analysis_completed=False,
                knowledge_items_created=0
            )
            
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid project ID")
        except Exception as e:
            logger.error(f"Failed to upload file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{project_id}/files")
    async def list_project_files(project_id: str):
        """List all files in project"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            project_vm = project_components.get('vm')
            
            if not project_vm:
                raise HTTPException(status_code=404, detail="Project not found")
            
            # List files in project
            result = await project_vm.execute_command_in_vm(
                project_vm,
                "find /project/files -type f -printf '%p %s %T@\\n' | head -100"
            )
            
            files = []
            for line in result.strip().split('\n'):
                if line:
                    parts = line.rsplit(' ', 2)
                    if len(parts) == 3:
                        file_path, size, timestamp = parts
                        files.append({
                            "path": file_path.replace('/project/files/', ''),
                            "size_bytes": int(size),
                            "modified_at": datetime.fromtimestamp(float(timestamp), timezone.utc).isoformat()
                        })
            
            return {"files": files, "total_count": len(files)}
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # KNOWLEDGE BASE ENDPOINTS
    # ========================================================================
    
    @router.post("/{project_id}/knowledge/search")
    async def search_knowledge(project_id: str, request: KnowledgeSearchRequest):
        """Search project knowledge base"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            knowledge_base = project_components.get('knowledge')
            
            if not knowledge_base:
                raise HTTPException(status_code=404, detail="Project knowledge base not found")
            
            results = await knowledge_base.search_knowledge(
                query=request.query,
                limit=request.limit,
                item_types=request.item_types
            )
            
            return {
                "query": request.query,
                "results": results,
                "total_found": len(results)
            }
            
        except Exception as e:
            logger.error(f"Failed to search knowledge: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{project_id}/knowledge/summary")
    async def get_knowledge_summary(project_id: str):
        """Get project knowledge base summary"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            knowledge_base = project_components.get('knowledge')
            
            if not knowledge_base:
                raise HTTPException(status_code=404, detail="Project knowledge base not found")
            
            summary = await knowledge_base.get_knowledge_summary()
            return summary
            
        except Exception as e:
            logger.error(f"Failed to get knowledge summary: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/{project_id}/knowledge/synthesize")
    async def synthesize_knowledge(project_id: str, topic: str = Body(..., embed=True)):
        """Synthesize knowledge about specific topic"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            knowledge_base = project_components.get('knowledge')
            
            if not knowledge_base:
                raise HTTPException(status_code=404, detail="Project knowledge base not found")
            
            synthesis = await knowledge_base.synthesize_knowledge(topic)
            return synthesis
            
        except Exception as e:
            logger.error(f"Failed to synthesize knowledge: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # COLLABORATION ENDPOINTS
    # ========================================================================
    
    @router.post("/{project_id}/collaboration/enable")
    async def enable_collaboration(project_id: str, team_size: int = Body(3, embed=True)):
        """Enable multi-agent collaboration for project"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            collaboration_manager = project_components.get('collaboration')
            
            if not collaboration_manager:
                raise HTTPException(status_code=404, detail="Project collaboration not available")
            
            result = await collaboration_manager.enable_collaboration(team_size)
            
            # Broadcast collaboration enabled
            await websocket_manager.broadcast_to_project(project_id, {
                "type": "collaboration_enabled",
                "project_id": project_id,
                "team_size": result.get('team_size', 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to enable collaboration: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/{project_id}/collaboration/task")
    async def execute_collaborative_task(project_id: str, request: CollaborationTaskRequest):
        """Execute task using project collaboration team"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            collaboration_manager = project_components.get('collaboration')
            
            if not collaboration_manager:
                raise HTTPException(status_code=404, detail="Project collaboration not available")
            
            result = await collaboration_manager.execute_collaborative_task(
                title=request.title,
                description=request.description,
                task_type=request.task_type,
                project_files=request.project_files
            )
            
            # Broadcast task completion
            await websocket_manager.broadcast_to_project(project_id, {
                "type": "collaborative_task_completed",
                "project_id": project_id,
                "task_title": request.title,
                "participating_agents": result.get('participating_agents', 0),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to execute collaborative task: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{project_id}/collaboration/status")
    async def get_collaboration_status(project_id: str):
        """Get current collaboration status"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            collaboration_manager = project_components.get('collaboration')
            
            if not collaboration_manager:
                return {"collaboration_enabled": False}
            
            status = await collaboration_manager.get_collaboration_status()
            return status
            
        except Exception as e:
            logger.error(f"Failed to get collaboration status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # AUTOMATION ENDPOINTS
    # ========================================================================
    
    @router.post("/{project_id}/automation/rules")
    async def create_automation_rule(project_id: str, request: AutomationRuleRequest):
        """Create new automation rule"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            automation_engine = project_components.get('automation')
            
            if not automation_engine:
                raise HTTPException(status_code=404, detail="Project automation not available")
            
            # Create automation rule
            from .project_automation import AutomationRule, AutomationTrigger, AutomationAction
            
            rule = AutomationRule(
                name=request.name,
                description=request.description,
                trigger_type=AutomationTrigger(request.trigger_type),
                trigger_config=request.trigger_config,
                action_type=AutomationAction(request.action_type),
                action_config=request.action_config,
                cron_expression=request.cron_expression,
                enabled=request.enabled
            )
            
            rule_id = await automation_engine.create_automation_rule(rule)
            
            return {
                "rule_id": str(rule_id),
                "name": request.name,
                "created": True
            }
            
        except Exception as e:
            logger.error(f"Failed to create automation rule: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{project_id}/automation/status")
    async def get_automation_status(project_id: str):
        """Get automation system status"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            automation_engine = project_components.get('automation')
            
            if not automation_engine:
                return {"automation_enabled": False}
            
            status = await automation_engine.get_automation_status()
            return status
            
        except Exception as e:
            logger.error(f"Failed to get automation status: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # ARTIFACT ENDPOINTS
    # ========================================================================
    
    @router.post("/{project_id}/artifacts")
    async def create_artifact(project_id: str, request: ArtifactCreateRequest):
        """Create new project artifact"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            artifact_manager = project_components.get('artifacts')
            
            if not artifact_manager:
                raise HTTPException(status_code=404, detail="Project artifacts not available")
            
            from ..morpheus_artifact_v2 import ArtifactType, CollaborationMode
            
            artifact = await artifact_manager.create_project_artifact(
                title=request.title,
                content=request.content,
                artifact_type=ArtifactType(request.artifact_type),
                user_id="api_user",  # TODO: Get from authentication
                enhance_with_project=request.enhance_with_project,
                collaboration_mode=CollaborationMode.LIVE_COLLABORATION if request.enable_collaboration else CollaborationMode.HUMAN_ONLY
            )
            
            # Broadcast artifact creation
            await websocket_manager.broadcast_to_project(project_id, {
                "type": "artifact_created",
                "project_id": project_id,
                "artifact_id": artifact.id,
                "title": request.title,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return {
                "artifact_id": artifact.id,
                "title": artifact.title,
                "type": artifact.artifact_type,
                "status": artifact.status,
                "vm_enabled": artifact.uses_project_vm,
                "created_at": artifact.created_at.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to create artifact: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/{project_id}/artifacts/{artifact_id}/execute")
    async def execute_artifact(project_id: str, artifact_id: str):
        """Execute artifact in project context"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            artifact_manager = project_components.get('artifacts')
            
            if not artifact_manager:
                raise HTTPException(status_code=404, detail="Project artifacts not available")
            
            execution_result = await artifact_manager.execute_artifact_in_project(artifact_id)
            
            # Broadcast artifact execution
            await websocket_manager.broadcast_to_project(project_id, {
                "type": "artifact_executed",
                "project_id": project_id,
                "artifact_id": artifact_id,
                "status": execution_result.get('status'),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
            return execution_result
            
        except Exception as e:
            logger.error(f"Failed to execute artifact: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{project_id}/artifacts")
    async def list_artifacts(project_id: str):
        """List project artifacts"""
        
        try:
            project_components = active_project_components.get(project_id, {})
            artifact_manager = project_components.get('artifacts')
            
            if not artifact_manager:
                return {"artifacts": [], "total_count": 0}
            
            summary = await artifact_manager.get_project_artifacts_summary()
            return summary
            
        except Exception as e:
            logger.error(f"Failed to list artifacts: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    # ========================================================================
    # WEBSOCKET ENDPOINT
    # ========================================================================
    
    @router.websocket("/{project_id}/ws")
    async def project_websocket(websocket: WebSocket, project_id: str, user_id: str = Query("anonymous")):
        """WebSocket endpoint for real-time project updates"""
        
        await websocket_manager.connect(websocket, project_id, user_id)
        
        try:
            while True:
                # Listen for client messages
                data = await websocket.receive_json()
                
                # Handle different message types
                if data.get("type") == "ping":
                    await websocket_manager.send_to_connection(websocket, {
                        "type": "pong",
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
                elif data.get("type") == "subscribe_to_updates":
                    # Client requesting specific update types
                    await websocket_manager.send_to_connection(websocket, {
                        "type": "subscribed",
                        "update_types": data.get("update_types", ["all"]),
                        "timestamp": datetime.now(timezone.utc).isoformat()
                    })
                
        except WebSocketDisconnect:
            websocket_manager.disconnect(websocket, project_id, user_id)
        except Exception as e:
            logger.error(f"WebSocket error for project {project_id}: {e}")
            websocket_manager.disconnect(websocket, project_id, user_id)
    
    # ========================================================================
    # HELPER FUNCTIONS
    # ========================================================================
    
    async def _initialize_project_components(
        project: ProjectMetadata,
        request: ProjectCreateRequest,
        background_tasks: BackgroundTasks
    ) -> Dict[str, Any]:
        """Initialize all project components"""
        
        project_id = str(project.project_id)
        project_vm = None
        
        # Get project VM
        if project.vm_id:
            # TODO: Get VM instance from vm_manager
            project_vm = "vm_placeholder"  # Would be actual VM instance
        
        components = {
            'vm': project_vm,
            'project': project
        }
        
        # Initialize components based on request
        if project_vm:
            # Initialize intelligence
            from .project_intelligence import ProjectIntelligenceEngine
            intelligence = ProjectIntelligenceEngine(project_id, project_vm)
            await intelligence.initialize()
            components['intelligence'] = intelligence
            
            # Initialize knowledge base
            from .project_knowledge import ProjectKnowledgeBase
            knowledge = ProjectKnowledgeBase(project_id, project_vm, Path(f"/tmp/project_{project_id}_knowledge"))
            await knowledge.initialize()
            components['knowledge'] = knowledge
            
            # Initialize collaboration if requested
            if request.enable_collaboration:
                collaboration = ProjectCollaborationManager(
                    project_id, project_vm, knowledge, model_loader, memory_manager
                )
                components['collaboration'] = collaboration
            
            # Initialize automation if requested
            if request.enable_automation:
                from .project_automation import ProjectAutomationEngine
                automation = ProjectAutomationEngine(project_id, project_vm, knowledge, intelligence)
                await automation.initialize()
                components['automation'] = automation
            
            # Initialize artifacts
            artifacts = ProjectArtifactManager(
                project_id, project_vm, knowledge, intelligence, artifact_manager
            )
            await artifacts.initialize()
            components['artifacts'] = artifacts
            
            # Start background intelligence monitoring
            background_tasks.add_task(intelligence.monitor_and_process_files)
        
        return components
    
    def _format_project_response(project: ProjectMetadata, components: Dict[str, Any]) -> ProjectResponse:
        """Format project response"""
        
        return ProjectResponse(
            project_id=str(project.project_id),
            name=project.name,
            description=project.description,
            project_type=project.project_type.value,
            status=project.status.value,
            specs=project.specs.__dict__,
            vm_connection={
                "vm_id": str(project.vm_id) if project.vm_id else None,
                "ssh_port": getattr(components.get('vm'), 'ssh_port', None),
                "web_port": getattr(components.get('vm'), 'web_port', None)
            },
            created_at=project.created_at.isoformat(),
            last_active=project.last_active.isoformat(),
            files_processed=project.files_processed,
            knowledge_items=project.knowledge_items,
            collaboration_enabled=project.collaboration_enabled,
            automation_enabled=bool(components.get('automation')),
            artifacts_count=len(getattr(components.get('artifacts'), 'project_artifacts', {}))
        )
    
    async def _process_uploaded_file(
        project_id: str,
        filename: str,
        intelligence_engine,
        websocket_manager: ProjectWebSocketManager
    ):
        """Background task to process uploaded file"""
        
        try:
            if intelligence_engine:
                # Trigger file processing
                await intelligence_engine._scan_for_new_files()
                
                # Broadcast processing complete
                await websocket_manager.broadcast_to_project(project_id, {
                    "type": "file_processed",
                    "project_id": project_id,
                    "filename": filename,
                    "timestamp": datetime.now(timezone.utc).isoformat()
                })
        
        except Exception as e:
            logger.error(f"Failed to process uploaded file {filename}: {e}")
            
            # Broadcast processing error
            await websocket_manager.broadcast_to_project(project_id, {
                "type": "file_processing_error",
                "project_id": project_id,
                "filename": filename,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
    
    return router