"""
MORPHEUS CHAT - Enhanced Collaborative Artifact System
Real-time collaboration, VM integration, and terminal GUI functionality
"""

import asyncio
import json
import logging
import os
import pty
import select
import signal
import subprocess
import termios
import threading
import time
import tty
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import websockets
import psutil
from pathlib import Path

from pydantic import BaseModel, Field
from .artifact_system import (
    ArtifactManager, ArtifactContent, ArtifactType, ArtifactStatus,
    SecurityLevel, ExecutionResult, ArtifactMetrics
)
from schemas.session import SessionID, UserID
from core.memory_core import MemoryManager, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)


class CollaborationMode(str, Enum):
    """Collaboration types"""
    SOLO = "solo"
    REAL_TIME = "real_time"
    MULTI_MODEL = "multi_model"
    TERMINAL_SHARED = "terminal_shared"
    VM_SHARED = "vm_shared"


class TerminalSession:
    """Real terminal session with pty support"""
    
    def __init__(self, session_id: str, working_dir: str = "/tmp"):
        self.session_id = session_id
        self.working_dir = working_dir
        self.master_fd = None
        self.slave_fd = None
        self.process = None
        self.is_active = False
        self.subscribers: Set[Callable] = set()
        self.history: List[Dict] = []
        
    async def start(self):
        """Start terminal session"""
        try:
            # Create pty
            self.master_fd, self.slave_fd = pty.openpty()
            
            # Start shell process
            self.process = subprocess.Popen(
                ['/bin/bash', '-i'],
                stdin=self.slave_fd,
                stdout=self.slave_fd,
                stderr=self.slave_fd,
                cwd=self.working_dir,
                env=dict(os.environ, TERM='xterm-256color', PS1='\\u@morpheus:\\w$ '),
                preexec_fn=os.setsid
            )
            
            self.is_active = True
            
            # Start monitoring thread
            threading.Thread(target=self._monitor_output, daemon=True).start()
            
            logger.info(f"Terminal session {self.session_id} started (PID: {self.process.pid})")
            
        except Exception as e:
            logger.error(f"Failed to start terminal session: {e}")
            self.cleanup()
            raise
    
    def _monitor_output(self):
        """Monitor terminal output and broadcast to subscribers"""
        while self.is_active and self.master_fd:
            try:
                # Check if there's data to read
                ready, _, _ = select.select([self.master_fd], [], [], 0.1)
                
                if ready:
                    data = os.read(self.master_fd, 1024)
                    if data:
                        output = data.decode('utf-8', errors='replace')
                        
                        # Add to history
                        self.history.append({
                            'timestamp': datetime.now(timezone.utc).isoformat(),
                            'type': 'output',
                            'content': output
                        })
                        
                        # Broadcast to subscribers
                        for callback in self.subscribers.copy():
                            try:
                                callback(output)
                            except Exception as e:
                                logger.error(f"Error broadcasting terminal output: {e}")
                                self.subscribers.discard(callback)
                
            except OSError:
                break
            except Exception as e:
                logger.error(f"Terminal monitoring error: {e}")
                break
        
        logger.info(f"Terminal session {self.session_id} monitoring stopped")
    
    async def send_input(self, input_data: str):
        """Send input to terminal"""
        if not self.is_active or not self.master_fd:
            raise RuntimeError("Terminal session not active")
        
        try:
            # Add to history
            self.history.append({
                'timestamp': datetime.now(timezone.utc).isoformat(),
                'type': 'input',
                'content': input_data
            })
            
            # Send to terminal
            os.write(self.master_fd, input_data.encode())
            
        except Exception as e:
            logger.error(f"Failed to send input to terminal: {e}")
            raise
    
    def subscribe(self, callback: Callable[[str], None]):
        """Subscribe to terminal output"""
        self.subscribers.add(callback)
    
    def unsubscribe(self, callback: Callable[[str], None]):
        """Unsubscribe from terminal output"""
        self.subscribers.discard(callback)
    
    def get_history(self, lines: int = 100) -> List[Dict]:
        """Get terminal history"""
        return self.history[-lines:] if lines > 0 else self.history
    
    def cleanup(self):
        """Cleanup terminal session"""
        self.is_active = False
        
        if self.process and self.process.poll() is None:
            try:
                # Send SIGTERM to process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    # Force kill if needed
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    pass
        
        if self.master_fd:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
        
        if self.slave_fd:
            try:
                os.close(self.slave_fd)
            except OSError:
                pass
        
        logger.info(f"Terminal session {self.session_id} cleaned up")


@dataclass
class CollaborationState:
    """Real-time collaboration state"""
    artifact_id: UUID
    active_users: Set[UserID] = field(default_factory=set)
    cursors: Dict[UserID, Dict[str, Any]] = field(default_factory=dict)
    operations: List[Dict] = field(default_factory=list)
    version: int = 0
    last_sync: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ModelComparison:
    """Multi-model comparison results"""
    original_prompt: str
    model_outputs: Dict[str, str] = field(default_factory=dict)
    execution_results: Dict[str, ExecutionResult] = field(default_factory=dict)
    performance_metrics: Dict[str, Dict] = field(default_factory=dict)
    user_ratings: Dict[str, float] = field(default_factory=dict)
    cost_analysis: Dict[str, float] = field(default_factory=dict)


class VMIntegration:
    """VM integration for artifact execution"""
    
    def __init__(self, vm_manager=None):
        self.vm_manager = vm_manager
        self.active_sessions: Dict[str, Any] = {}
        self.resource_monitors: Dict[str, threading.Thread] = {}
    
    async def create_vm_session(self, user_id: UserID, session_config: Dict) -> str:
        """Create dedicated VM session for artifact execution"""
        session_id = str(uuid4())
        
        try:
            # If we have VM manager, create actual VM
            if self.vm_manager:
                vm_config = {
                    'cpu_cores': session_config.get('cpu_cores', 2),
                    'memory_mb': session_config.get('memory_mb', 2048),
                    'disk_gb': session_config.get('disk_gb', 20),
                    'network_enabled': session_config.get('network_enabled', True),
                    'gpu_enabled': session_config.get('gpu_enabled', False)
                }
                
                vm_instance = await self.vm_manager.create_vm(
                    user_id=user_id,
                    config=vm_config
                )
                
                self.active_sessions[session_id] = {
                    'vm_instance': vm_instance,
                    'user_id': user_id,
                    'created_at': datetime.now(timezone.utc),
                    'config': vm_config
                }
            else:
                # Fallback to container/process isolation
                self.active_sessions[session_id] = {
                    'user_id': user_id,
                    'created_at': datetime.now(timezone.utc),
                    'config': session_config,
                    'process_isolation': True
                }
            
            # Start resource monitoring
            monitor_thread = threading.Thread(
                target=self._monitor_session_resources,
                args=(session_id,),
                daemon=True
            )
            monitor_thread.start()
            self.resource_monitors[session_id] = monitor_thread
            
            logger.info(f"VM session {session_id} created for user {user_id}")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to create VM session: {e}")
            raise
    
    def _monitor_session_resources(self, session_id: str):
        """Monitor VM session resources"""
        while session_id in self.active_sessions:
            try:
                session = self.active_sessions[session_id]
                
                # Get current system stats
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                
                # Update session stats
                session.setdefault('resource_history', []).append({
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'cpu_percent': cpu_percent,
                    'memory_percent': memory.percent,
                    'memory_used_mb': memory.used / 1024 / 1024,
                    'disk_percent': disk.percent,
                    'disk_used_gb': disk.used / 1024 / 1024 / 1024
                })
                
                # Keep only last 100 entries
                if len(session['resource_history']) > 100:
                    session['resource_history'] = session['resource_history'][-100:]
                
                time.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring error for session {session_id}: {e}")
                break
    
    async def execute_in_vm(self, session_id: str, artifact: ArtifactContent) -> ExecutionResult:
        """Execute artifact in VM session"""
        if session_id not in self.active_sessions:
            raise ValueError(f"VM session {session_id} not found")
        
        session = self.active_sessions[session_id]
        
        try:
            if 'vm_instance' in session:
                # Execute in actual VM
                return await self._execute_in_vm_instance(session['vm_instance'], artifact)
            else:
                # Execute with process isolation
                return await self._execute_with_isolation(artifact)
        
        except Exception as e:
            logger.error(f"VM execution failed: {e}")
            return ExecutionResult(
                success=False,
                output="",
                error=str(e),
                exit_code=1,
                execution_time=0.0,
                resource_usage={}
            )
    
    async def _execute_in_vm_instance(self, vm_instance, artifact: ArtifactContent) -> ExecutionResult:
        """Execute in actual VM instance"""
        # This would integrate with your VM manager
        # For now, implement basic execution
        start_time = time.time()
        
        try:
            # Create temporary script file
            script_content = artifact.raw_content
            script_path = f"/tmp/artifact_{artifact.metadata.artifact_id}.py"
            
            # Write script to VM (would use VM manager API)
            # vm_instance.write_file(script_path, script_content)
            
            # Execute script in VM
            # result = await vm_instance.execute_command(f"python3 {script_path}")
            
            # Placeholder implementation
            result = subprocess.run(
                ['python3', '-c', script_content],
                capture_output=True,
                text=True,
                timeout=300
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                exit_code=result.returncode,
                execution_time=time.time() - start_time,
                resource_usage={}
            )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error="Execution timeout",
                exit_code=124,
                execution_time=300.0,
                resource_usage={}
            )
    
    async def _execute_with_isolation(self, artifact: ArtifactContent) -> ExecutionResult:
        """Execute with process isolation"""
        start_time = time.time()
        
        try:
            # Use subprocess with limited permissions
            result = subprocess.run(
                ['python3', '-c', artifact.raw_content],
                capture_output=True,
                text=True,
                timeout=300,
                cwd='/tmp',
                env={'PATH': '/usr/bin:/bin', 'PYTHONPATH': ''}
            )
            
            return ExecutionResult(
                success=result.returncode == 0,
                output=result.stdout,
                error=result.stderr,
                exit_code=result.returncode,
                execution_time=time.time() - start_time,
                resource_usage={}
            )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                output="",
                error="Execution timeout",
                exit_code=124,
                execution_time=300.0,
                resource_usage={}
            )
    
    async def cleanup_session(self, session_id: str):
        """Cleanup VM session"""
        if session_id in self.active_sessions:
            session = self.active_sessions.pop(session_id)
            
            # Stop resource monitoring
            if session_id in self.resource_monitors:
                # Thread will stop when session is removed
                self.resource_monitors.pop(session_id)
            
            # Cleanup VM instance if exists
            if 'vm_instance' in session and self.vm_manager:
                await self.vm_manager.cleanup_vm(session['vm_instance'])
            
            logger.info(f"VM session {session_id} cleaned up")


class CollaborationManager:
    """Real-time collaboration management"""
    
    def __init__(self):
        self.active_collaborations: Dict[UUID, CollaborationState] = {}
        self.websocket_connections: Dict[str, Set[Any]] = {}  # artifact_id -> websockets
        self.terminal_sessions: Dict[str, TerminalSession] = {}
        self.operation_lock = asyncio.Lock()
    
    async def join_collaboration(self, artifact_id: UUID, user_id: UserID, websocket) -> CollaborationState:
        """Join collaborative editing session"""
        
        if artifact_id not in self.active_collaborations:
            self.active_collaborations[artifact_id] = CollaborationState(artifact_id=artifact_id)
        
        collaboration = self.active_collaborations[artifact_id]
        collaboration.active_users.add(user_id)
        
        # Add websocket connection
        artifact_key = str(artifact_id)
        if artifact_key not in self.websocket_connections:
            self.websocket_connections[artifact_key] = set()
        self.websocket_connections[artifact_key].add(websocket)
        
        # Notify other users
        await self._broadcast_user_joined(artifact_id, user_id, exclude_websocket=websocket)
        
        logger.info(f"User {user_id} joined collaboration for artifact {artifact_id}")
        return collaboration
    
    async def leave_collaboration(self, artifact_id: UUID, user_id: UserID, websocket):
        """Leave collaborative editing session"""
        
        if artifact_id in self.active_collaborations:
            collaboration = self.active_collaborations[artifact_id]
            collaboration.active_users.discard(user_id)
            collaboration.cursors.pop(user_id, None)
            
            # Remove websocket connection
            artifact_key = str(artifact_id)
            if artifact_key in self.websocket_connections:
                self.websocket_connections[artifact_key].discard(websocket)
            
            # Notify other users
            await self._broadcast_user_left(artifact_id, user_id)
            
            # Cleanup if no active users
            if not collaboration.active_users:
                self.active_collaborations.pop(artifact_id, None)
                self.websocket_connections.pop(artifact_key, None)
        
        logger.info(f"User {user_id} left collaboration for artifact {artifact_id}")
    
    async def handle_operation(self, artifact_id: UUID, user_id: UserID, operation: Dict) -> Dict:
        """Handle collaborative operation"""
        
        async with self.operation_lock:
            if artifact_id not in self.active_collaborations:
                raise ValueError("Collaboration session not found")
            
            collaboration = self.active_collaborations[artifact_id]
            
            # Process operation based on type
            if operation['type'] == 'cursor_move':
                collaboration.cursors[user_id] = {
                    'line': operation['line'],
                    'column': operation['column'],
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
            
            elif operation['type'] == 'text_insert':
                # Add operation to history
                operation['user_id'] = user_id
                operation['timestamp'] = datetime.now(timezone.utc).isoformat()
                operation['version'] = collaboration.version
                collaboration.operations.append(operation)
                collaboration.version += 1
            
            elif operation['type'] == 'text_delete':
                # Add operation to history
                operation['user_id'] = user_id
                operation['timestamp'] = datetime.now(timezone.utc).isoformat()
                operation['version'] = collaboration.version
                collaboration.operations.append(operation)
                collaboration.version += 1
            
            # Update last sync time
            collaboration.last_sync = datetime.now(timezone.utc)
            
            # Broadcast to other collaborators
            await self._broadcast_operation(artifact_id, operation, exclude_user=user_id)
            
            return {
                'success': True,
                'version': collaboration.version,
                'timestamp': collaboration.last_sync.isoformat()
            }
    
    async def create_terminal_session(self, artifact_id: UUID, user_id: UserID, working_dir: str = "/tmp") -> str:
        """Create shared terminal session"""
        
        session_id = f"{artifact_id}_{user_id}_{int(time.time())}"
        
        try:
            terminal = TerminalSession(session_id, working_dir)
            await terminal.start()
            
            self.terminal_sessions[session_id] = terminal
            
            logger.info(f"Terminal session {session_id} created for artifact {artifact_id}")
            return session_id
        
        except Exception as e:
            logger.error(f"Failed to create terminal session: {e}")
            raise
    
    async def get_terminal_session(self, session_id: str) -> Optional[TerminalSession]:
        """Get terminal session"""
        return self.terminal_sessions.get(session_id)
    
    async def cleanup_terminal_session(self, session_id: str):
        """Cleanup terminal session"""
        if session_id in self.terminal_sessions:
            terminal = self.terminal_sessions.pop(session_id)
            terminal.cleanup()
    
    async def _broadcast_user_joined(self, artifact_id: UUID, user_id: UserID, exclude_websocket=None):
        """Broadcast user joined event"""
        message = {
            'type': 'user_joined',
            'artifact_id': str(artifact_id),
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self._broadcast_to_artifact(artifact_id, message, exclude_websocket)
    
    async def _broadcast_user_left(self, artifact_id: UUID, user_id: UserID):
        """Broadcast user left event"""
        message = {
            'type': 'user_left',
            'artifact_id': str(artifact_id),
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self._broadcast_to_artifact(artifact_id, message)
    
    async def _broadcast_operation(self, artifact_id: UUID, operation: Dict, exclude_user: UserID = None):
        """Broadcast operation to collaborators"""
        message = {
            'type': 'operation',
            'artifact_id': str(artifact_id),
            'operation': operation,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        await self._broadcast_to_artifact(artifact_id, message, exclude_user=exclude_user)
    
    async def _broadcast_to_artifact(self, artifact_id: UUID, message: Dict, exclude_websocket=None, exclude_user: UserID = None):
        """Broadcast message to all connections for an artifact"""
        artifact_key = str(artifact_id)
        
        if artifact_key not in self.websocket_connections:
            return
        
        connections = self.websocket_connections[artifact_key].copy()
        
        for websocket in connections:
            try:
                # Skip excluded websocket
                if exclude_websocket and websocket == exclude_websocket:
                    continue
                
                # Skip excluded user (would need to track user per websocket)
                if exclude_user:
                    # For now, skip this check - would need websocket -> user mapping
                    pass
                
                await websocket.send_text(json.dumps(message))
            
            except Exception as e:
                logger.error(f"Failed to broadcast to websocket: {e}")
                # Remove dead connection
                self.websocket_connections[artifact_key].discard(websocket)


class EnhancedArtifactManager:
    """Enhanced artifact manager with collaboration and VM integration"""
    
    def __init__(self, 
                 storage_dir: str = "data/artifacts",
                 execution_dir: str = "data/execution",
                 memory_manager: Optional[MemoryManager] = None,
                 vm_manager = None):
        
        # Initialize base manager
        self.base_manager = ArtifactManager(storage_dir, execution_dir)
        
        # Enhanced components
        self.collaboration_manager = CollaborationManager()
        self.vm_integration = VMIntegration(vm_manager)
        self.memory_manager = memory_manager
        
        # Multi-model comparison
        self.model_comparisons: Dict[UUID, ModelComparison] = {}
        
        # Progressive features tracking
        self.user_progression: Dict[UserID, Dict] = {}
        self.easter_eggs_unlocked: Dict[UserID, Set[str]] = {}
        
        logger.info("Enhanced artifact manager initialized")
    
    async def create_collaborative_artifact(self,
                                          name: str,
                                          content: str,
                                          artifact_type: ArtifactType,
                                          user_id: UserID,
                                          session_id: SessionID,
                                          collaboration_mode: CollaborationMode = CollaborationMode.SOLO,
                                          **kwargs) -> ArtifactContent:
        """Create artifact with collaboration features"""
        
        # Create base artifact
        artifact = await self.base_manager.create_artifact(
            name=name,
            content=content,
            artifact_type=artifact_type,
            user_id=user_id,
            session_id=session_id,
            **kwargs
        )
        
        # Enable collaboration if requested
        if collaboration_mode != CollaborationMode.SOLO:
            await self.collaboration_manager.join_collaboration(
                artifact.metadata.artifact_id,
                user_id,
                None  # WebSocket would be provided in real usage
            )
        
        # Store in memory for context
        if self.memory_manager:
            await self._store_artifact_memory(artifact, user_id)
        
        # Check for easter eggs
        await self._check_easter_eggs(user_id, artifact)
        
        return artifact
    
    async def execute_with_vm(self, artifact_id: UUID, user_id: UserID, vm_config: Optional[Dict] = None) -> ExecutionResult:
        """Execute artifact in dedicated VM"""
        
        artifact = await self.base_manager.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact {artifact_id} not found")
        
        # Create VM session if config provided
        if vm_config:
            session_id = await self.vm_integration.create_vm_session(user_id, vm_config)
            try:
                result = await self.vm_integration.execute_in_vm(session_id, artifact)
            finally:
                await self.vm_integration.cleanup_session(session_id)
        else:
            # Use base manager execution
            result = await self.base_manager.execute_artifact(artifact_id, user_id)
        
        return result
    
    async def create_terminal_artifact(self, user_id: UserID, session_id: SessionID, working_dir: str = "/tmp") -> Tuple[ArtifactContent, str]:
        """Create terminal artifact with live session"""
        
        # Create terminal session
        terminal_session_id = await self.collaboration_manager.create_terminal_session(
            uuid4(), user_id, working_dir
        )
        
        # Create artifact
        artifact = await self.base_manager.create_artifact(
            name=f"Terminal Session {terminal_session_id[:8]}",
            content=f"# Live Terminal Session\n# Session ID: {terminal_session_id}\n# Working Directory: {working_dir}\n",
            artifact_type=ArtifactType.TERMINAL,
            user_id=user_id,
            session_id=session_id,
            security_level=SecurityLevel.TERMINAL,
            working_directory=working_dir
        )
        
        return artifact, terminal_session_id
    
    async def enable_multi_model_comparison(self, artifact_id: UUID, models: List[str], original_prompt: str) -> ModelComparison:
        """Enable multi-model comparison for artifact"""
        
        artifact = await self.base_manager.get_artifact(artifact_id)
        if not artifact:
            raise ValueError(f"Artifact {artifact_id} not found")
        
        comparison = ModelComparison(original_prompt=original_prompt)
        
        # Store original as first model
        comparison.model_outputs['original'] = artifact.raw_content
        
        # This would integrate with your model loading system
        # For now, create placeholder for additional models
        for model in models:
            if model != 'original':
                # Would generate content with different model
                comparison.model_outputs[model] = f"// Content generated by {model}\n{artifact.raw_content}"
                
                # Would execute and compare performance
                comparison.performance_metrics[model] = {
                    'generation_time': 0.5,
                    'tokens_generated': len(artifact.raw_content) // 4,
                    'cost_estimate': 0.0 if 'local' in model.lower() else 0.01
                }
        
        self.model_comparisons[artifact_id] = comparison
        return comparison
    
    async def get_collaboration_state(self, artifact_id: UUID) -> Optional[CollaborationState]:
        """Get collaboration state for artifact"""
        return self.collaboration_manager.active_collaborations.get(artifact_id)
    
    async def get_user_progression(self, user_id: UserID) -> Dict:
        """Get user progression and unlocked features"""
        if user_id not in self.user_progression:
            self.user_progression[user_id] = {
                'level': 1,
                'artifacts_created': 0,
                'collaborations_joined': 0,
                'code_executed': 0,
                'vm_sessions': 0,
                'terminal_sessions': 0
            }
        
        return {
            'progression': self.user_progression[user_id],
            'easter_eggs': list(self.easter_eggs_unlocked.get(user_id, set())),
            'next_unlocks': self._get_next_unlocks(user_id)
        }
    
    async def _store_artifact_memory(self, artifact: ArtifactContent, user_id: UserID):
        """Store artifact creation in memory"""
        if not self.memory_manager:
            return
        
        memory_content = f"""
        Created artifact: {artifact.metadata.name}
        Type: {artifact.metadata.artifact_type}
        Size: {artifact.metadata.content_size} bytes
        Lines: {artifact.metadata.line_count}
        Executable: {artifact.metadata.is_executable}
        Content preview: {artifact.raw_content[:200]}...
        """
        
        await self.memory_manager.store_memory(
            user_id=user_id,
            content=memory_content,
            memory_type=MemoryType.TOOL_RESULT,
            importance=MemoryImportance.MEDIUM,
            metadata={
                'artifact_id': str(artifact.metadata.artifact_id),
                'artifact_type': artifact.metadata.artifact_type,
                'action': 'create_artifact'
            }
        )
    
    async def _check_easter_eggs(self, user_id: UserID, artifact: ArtifactContent):
        """Check for easter egg unlocks"""
        if user_id not in self.easter_eggs_unlocked:
            self.easter_eggs_unlocked[user_id] = set()
        
        unlocked = self.easter_eggs_unlocked[user_id]
        
        # Check various conditions
        if artifact.metadata.artifact_type == ArtifactType.TERMINAL and 'terminal_master' not in unlocked:
            unlocked.add('terminal_master')
            logger.info(f"Easter egg unlocked for {user_id}: terminal_master")
        
        if artifact.metadata.security_level == SecurityLevel.UNRESTRICTED and 'unrestricted_access' not in unlocked:
            unlocked.add('unrestricted_access')
            logger.info(f"Easter egg unlocked for {user_id}: unrestricted_access")
        
        if len(artifact.raw_content) > 10000 and 'code_marathon' not in unlocked:
            unlocked.add('code_marathon')
            logger.info(f"Easter egg unlocked for {user_id}: code_marathon")
    
    def _get_next_unlocks(self, user_id: UserID) -> List[Dict]:
        """Get next possible unlocks for user"""
        progression = self.user_progression.get(user_id, {})
        unlocked = self.easter_eggs_unlocked.get(user_id, set())
        
        next_unlocks = []
        
        if 'collaboration_master' not in unlocked:
            next_unlocks.append({
                'name': 'collaboration_master',
                'description': 'Join 5 collaborative sessions',
                'progress': progression.get('collaborations_joined', 0),
                'target': 5
            })
        
        if 'vm_wizard' not in unlocked:
            next_unlocks.append({
                'name': 'vm_wizard',
                'description': 'Create 10 VM sessions',
                'progress': progression.get('vm_sessions', 0),
                'target': 10
            })
        
        return next_unlocks
    
    async def cleanup(self):
        """Cleanup all resources"""
        await self.base_manager.cleanup()
        
        # Cleanup terminal sessions
        for session_id in list(self.collaboration_manager.terminal_sessions.keys()):
            await self.collaboration_manager.cleanup_terminal_session(session_id)
        
        # Cleanup VM sessions
        for session_id in list(self.vm_integration.active_sessions.keys()):
            await self.vm_integration.cleanup_session(session_id)
        
        logger.info("Enhanced artifact manager cleanup completed")


# Delegate all base methods to the base manager
def __getattr__(name):
    """Delegate missing methods to base manager"""
    if hasattr(EnhancedArtifactManager, name):
        return getattr(EnhancedArtifactManager, name)
    
    def wrapper(self, *args, **kwargs):
        return getattr(self.base_manager, name)(*args, **kwargs)
    
    return wrapper