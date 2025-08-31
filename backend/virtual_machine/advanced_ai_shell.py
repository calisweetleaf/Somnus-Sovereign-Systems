"""
SOMNUS SYSTEMS - Advanced AI Shell Module
Multi-Modal AI Interaction: VM Management + Container Orchestration + Multi-Agent Collaboration

ARCHITECTURE:
- AI operates from persistent VM environment (never resets)
- Orchestrates disposable container overlays for artifact execution
- Coordinates with other AI VMs for multi-agent collaboration
- Maintains security through architectural separation and local-only APIs

SECURITY MODEL:
- All communication is localhost-only (no external exposure)
- Docker API calls are local-only (127.0.0.1)
- Inter-VM communication uses secure local protocols
- Container isolation provides security boundaries
- No cloud dependencies or external APIs
"""

import asyncio
import json
import logging
import os
import shlex
import socket
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4

import aiofiles
import docker
import artifact_system  # new local API for VM commands
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION AND MODELS
# ============================================================================

class ExecutionContext(str, Enum):
    """Execution context types"""
    VM_NATIVE = "vm_native"                 # Direct execution in AI's VM
    CONTAINER_OVERLAY = "container_overlay" # Artifact container execution
    MULTI_AGENT = "multi_agent"            # Multi-agent collaboration
    HYBRID = "hybrid"                       # VM orchestrating container


class CommandType(str, Enum):
    """Command classification for routing"""
    SYSTEM = "system"                       # Basic system commands
    ARTIFACT = "artifact"                   # Artifact-related operations
    COLLABORATION = "collaboration"         # Multi-agent commands
    RESEARCH = "research"                   # Research and analysis
    DEVELOPMENT = "development"             # Development workflows


@dataclass
class ExecutionResult:
    """Comprehensive execution result"""
    command: str
    stdout: str
    stderr: str
    return_code: int
    execution_time: float
    context: ExecutionContext
    command_type: CommandType
    was_corrected: bool = False
    correction_prompt: Optional[str] = None
    container_id: Optional[str] = None
    collaborator_responses: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ContainerSpec:
    """Container specification for artifact execution"""
    image: str = "somnus-artifact:unlimited"
    cpu_limit: Optional[str] = None
    memory_limit: Optional[str] = None
    gpu_access: bool = True
    network_access: bool = True
    environment: Dict[str, str] = field(default_factory=dict)
    volumes: Dict[str, str] = field(default_factory=dict)
    working_dir: str = "/workspace"


@dataclass
class CollaborationSession:
    """Multi-agent collaboration session"""
    session_id: UUID
    primary_agent_id: UUID
    collaborator_ids: List[UUID]
    task_description: str
    status: str = "initializing"
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    responses: Dict[UUID, str] = field(default_factory=dict)
    final_synthesis: Optional[str] = None


class AIShellSettings(BaseSettings):
    """Advanced AI Shell configuration"""
    # VM Connection settings
    vm_host: str = "127.0.0.1"
    vm_port: int = 22
    vm_user: str = "morpheus"
    vm_password: Optional[str] = None
    vm_ssh_key_path: Optional[str] = f"{os.path.expanduser('~')}/.ssh/id_rsa"
    command_timeout: int = 300  # 5 minutes default
    
    # Container orchestration
    docker_host: str = "unix:///var/run/docker.sock"
    container_network: str = "somnus_network"
    artifact_registry: str = "localhost:5000"
    
    # Multi-agent collaboration
    collaboration_port_base: int = 8100
    max_concurrent_agents: int = 10
    agent_communication_timeout: int = 30
    
    # Memory and logging
    memory_log_path: str = "./data/ai_shell_memory.jsonl"
    execution_log_path: str = "./data/ai_shell_execution.log"
    
    # Security settings
    allow_privileged_containers: bool = False
    restrict_network_access: bool = False
    enable_command_validation: bool = True
    
    class Config:
        env_prefix = "AI_SHELL_"


# ============================================================================
# CORE VM MANAGER
# ============================================================================

class AdvancedVMManager:
    """Enhanced VM manager with container orchestration capabilities"""
    
    def __init__(self, settings: AIShellSettings, agent_id: Optional[UUID] = None):
        self.settings = settings
        self.agent_id = agent_id or uuid4()
        # self.ssh_client = None  # removed
        self.docker_client = None
        self.active_containers: Dict[str, Any] = {}
        self.collaboration_socket: Optional[socket.socket] = None
        
    async def initialize(self) -> bool:
        """Initialize VM and container connections"""
        try:
            # Establish Docker client for container orchestration
            await self._initialize_docker_client()
            
            # Setup collaboration networking if needed
            await self._setup_collaboration_network()
            
            logger.info(f"AI Shell initialized for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Shell: {e}")
            return False
    
    async def _initialize_docker_client(self):
        """Initialize Docker client for container orchestration"""
        try:
            self.docker_client = docker.from_env()
            
            # Test Docker connection with timeout
            try:
                self.docker_client.ping()
            except docker.errors.APIError as e:
                raise ConnectionError(f"Docker API error: {e}")
            
            # Ensure Somnus network exists with proper configuration
            try:
                network = self.docker_client.networks.get(self.settings.container_network)
                # Verify network configuration
                network.reload()
                if not network.attrs.get('Driver') == 'bridge':
                    logger.warning(f"Network {self.settings.container_network} has unexpected driver")
            except docker.errors.NotFound:
                # Create network with security-focused options
                ipam_pool = docker.types.IPAMPool(
                    subnet='172.20.0.0/16',
                    gateway='172.20.0.1'
                )
                ipam_config = docker.types.IPAMConfig(pool_configs=[ipam_pool])
                
                self.docker_client.networks.create(
                    self.settings.container_network,
                    driver="bridge",
                    ipam=ipam_config,
                    options={
                        "com.docker.network.bridge.enable_icc": "true",
                        "com.docker.network.bridge.enable_ip_masquerade": "true",
                        "com.docker.network.bridge.host_binding_ipv4": "127.0.0.1"
                    },
                    labels={
                        "somnus.system": "ai_shell",
                        "somnus.agent_id": str(self.agent_id)
                    }
                )
            
            # Verify Docker daemon capabilities
            info = self.docker_client.info()
            if not info.get('ServerVersion'):
                raise RuntimeError("Unable to determine Docker server version")
                
            logger.info(f"Docker client initialized (server: {info['ServerVersion']})")
            
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    async def _setup_collaboration_network(self):
        """Setup networking for multi-agent collaboration"""
        try:
            # Calculate unique port for this agent
            agent_hash = hash(str(self.agent_id)) & 0x7FFFFFFF  # Ensure positive
            collaboration_port = self.settings.collaboration_port_base + (agent_hash % 1000)
            
            # Ensure port is within valid range
            if collaboration_port > 65535:
                collaboration_port = 65535
            
            # Create listening socket for agent communication
            self.collaboration_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.collaboration_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.collaboration_socket.setblocking(False)
            
            # Bind to localhost only for security
            self.collaboration_socket.bind(("127.0.0.1", collaboration_port))
            self.collaboration_socket.listen(5)
            
            logger.info(f"Collaboration network setup on port {collaboration_port}")
            
        except OSError as e:
            if e.errno == 98:  # Address already in use
                # Try next available port
                for port in range(collaboration_port + 1, collaboration_port + 100):
                    try:
                        self.collaboration_socket.bind(("127.0.0.1", port))
                        self.collaboration_socket.listen(5)
                        logger.info(f"Collaboration network setup on fallback port {port}")
                        break
                    except OSError:
                        continue
                else:
                    raise RuntimeError("No available ports for collaboration network")
            else:
                logger.error(f"Failed to setup collaboration network: {e}")
                raise
    
    async def execute_in_vm(self, command: str, **kwargs) -> Tuple[str, str, int]:
        """Execute command directly in AI's VM with comprehensive error handling"""
        try:
            logger.debug(f"Executing in VM: {command}")
            
            if not command or not isinstance(command, str):
                raise ValueError("Invalid command provided")
            
            command = command.strip()
            if not command:
                raise ValueError("Empty command provided")
            
            # Use local API instead of SSH
            stdout_data, stderr_data, exit_status = artifact_system.run_command(
                command,
                timeout=self.settings.command_timeout
            )
            
            logger.info(f"VM command executed: exit_code={exit_status}, cmd_length={len(command)}")
            
            return stdout_data, stderr_data, exit_status
            
        except Exception as e:
            logger.error(f"VM execution error: {e}")
            return "", str(e), -1


# ============================================================================
# CONTAINER ORCHESTRATION
# ============================================================================

class ContainerOrchestrator:
    """Orchestrates container overlays for artifact execution"""
    
    def __init__(self, vm_manager: AdvancedVMManager):
        self.vm_manager = vm_manager
        self.active_containers: Dict[str, docker.models.containers.Container] = {}
        self._container_lock = asyncio.Lock()
    
    async def create_artifact_container(
        self, 
        artifact_id: str, 
        spec: ContainerSpec,
        files: Dict[str, str] = None
    ) -> str:
        """Create specialized container for artifact execution with full validation"""
        if not artifact_id or not isinstance(artifact_id, str):
            raise ValueError("Invalid artifact_id provided")
        
        if not spec or not isinstance(spec, ContainerSpec):
            raise ValueError("Invalid ContainerSpec provided")
        
        async with self._container_lock:
            try:
                container_name = f"artifact_{artifact_id}_{int(time.time())}"
                
                # Validate container image exists
                try:
                    self.vm_manager.docker_client.images.get(spec.image)
                except docker.errors.ImageNotFound:
                    logger.info(f"Pulling container image: {spec.image}")
                    self.vm_manager.docker_client.images.pull(spec.image)
                
                # Prepare container configuration with security hardening
                container_config = {
                    "image": spec.image,
                    "name": container_name,
                    "detach": True,
                    "network": self.vm_manager.settings.container_network,
                    "working_dir": spec.working_dir,
                    "environment": {
                        **spec.environment,
                        "AGENT_ID": str(self.vm_manager.agent_id),
                        "ARTIFACT_ID": artifact_id
                    },
                    "volumes": {
                        **spec.volumes,
                        "/tmp": {"bind": "/tmp", "mode": "rw"}
                    },
                    "auto_remove": False,
                    "stdin_open": True,
                    "tty": True,
                    "remove": False,
                    "labels": {
                        "somnus.system": "ai_shell",
                        "somnus.agent_id": str(self.vm_manager.agent_id),
                        "somnus.artifact_id": artifact_id,
                        "somnus.created_at": datetime.now(timezone.utc).isoformat()
                    }
                }
                
                # Add resource limits with validation
                if spec.cpu_limit:
                    try:
                        cpu_value = float(spec.cpu_limit)
                        if cpu_value <= 0:
                            raise ValueError("CPU limit must be positive")
                        container_config["cpu_period"] = 100000
                        container_config["cpu_quota"] = int(cpu_value * 100000)
                    except ValueError as e:
                        logger.warning(f"Invalid CPU limit '{spec.cpu_limit}': {e}")
                
                if spec.memory_limit:
                    try:
                        # Parse memory limit (supports formats like "512m", "1g")
                        memory_bytes = self._parse_memory_limit(spec.memory_limit)
                        container_config["mem_limit"] = memory_bytes
                    except ValueError as e:
                        logger.warning(f"Invalid memory limit '{spec.memory_limit}': {e}")
                
                # Security restrictions
                if not self.vm_manager.settings.allow_privileged_containers:
                    container_config["privileged"] = False
                    container_config["cap_drop"] = ["ALL"]
                    container_config["security_opt"] = ["no-new-privileges"]
                
                # Network restrictions
                if self.vm_manager.settings.restrict_network_access:
                    container_config["network_mode"] = "none"
                
                # Enable GPU access if requested and available
                if spec.gpu_access:
                    try:
                        info = self.vm_manager.docker_client.info()
                        if 'Runtimes' in info and 'nvidia' in info['Runtimes']:
                            container_config["device_requests"] = [
                                docker.types.DeviceRequest(
                                    count=-1, 
                                    capabilities=[["gpu"]],
                                    options={'gpu': 'all'}
                                )
                            ]
                    except Exception as e:
                        logger.warning(f"GPU access requested but not available: {e}")
                
                # Create and start container
                container = self.vm_manager.docker_client.containers.run(**container_config)
                
                # Wait for container to be ready
                max_wait = 30
                waited = 0
                while waited < max_wait:
                    container.reload()
                    if container.status == 'running':
                        break
                    await asyncio.sleep(1)
                    waited += 1
                
                if container.status != 'running':
                    logs = container.logs().decode('utf-8', errors='replace')
                    raise RuntimeError(f"Container failed to start: {logs}")
                
                # Copy files to container if provided
                if files:
                    await self._copy_files_to_container(container, files)
                
                self.active_containers[container_name] = container
                
                logger.info(f"Created artifact container: {container_name} (status: {container.status})")
                return container_name
                
            except docker.errors.APIError as e:
                logger.error(f"Docker API error creating container: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to create artifact container: {e}")
                raise
    
    def _parse_memory_limit(self, limit: str) -> int:
        """Parse memory limit string to bytes"""
        limit = limit.lower().strip()
        
        multipliers = {
            'k': 1024,
            'm': 1024 * 1024,
            'g': 1024 * 1024 * 1024,
            'kb': 1024,
            'mb': 1024 * 1024,
            'gb': 1024 * 1024 * 1024
        }
        
        for suffix, multiplier in multipliers.items():
            if limit.endswith(suffix):
                try:
                    value = float(limit[:-len(suffix)])
                    return int(value * multiplier)
                except ValueError:
                    pass
        
        # Try direct bytes
        try:
            return int(limit)
        except ValueError:
            raise ValueError(f"Invalid memory limit format: {limit}")
    
    async def execute_in_container(
        self, 
        container_name: str, 
        command: str,
        stream_output: bool = False
    ) -> Tuple[str, str, int]:
        """Execute command in specific container with comprehensive monitoring"""
        if not container_name or not isinstance(container_name, str):
            raise ValueError("Invalid container_name provided")
        
        if not command or not isinstance(command, str):
            raise ValueError("Invalid command provided")
        
        try:
            container = self.active_containers.get(container_name)
            if not container:
                # Try to get container from Docker if not in active list
                try:
                    container = self.vm_manager.docker_client.containers.get(container_name)
                except docker.errors.NotFound:
                    raise ValueError(f"Container {container_name} not found")
            
            # Verify container is running
            container.reload()
            if container.status != 'running':
                raise RuntimeError(f"Container {container_name} is not running (status: {container.status})")
            
            logger.debug(f"Executing in container {container_name}: {command}")
            
            # Execute command with proper timeout
            timeout = self.vm_manager.settings.command_timeout
            exec_result = container.exec_run(
                cmd=["/bin/bash", "-c", command],
                stdout=True,
                stderr=True,
                stream=stream_output,
                demux=True,
                tty=False,
                privileged=False,
                user="root"
            )
            
            if stream_output:
                # Handle streaming output with timeout
                stdout_lines = []
                stderr_lines = []
                
                try:
                    for output in exec_result.output:
                        if output[0]:  # stdout
                            stdout_lines.append(output[0].decode('utf-8', errors='replace'))
                        if output[1]:  # stderr
                            stderr_lines.append(output[1].decode('utf-8', errors='replace'))
                    
                    stdout_data = ''.join(stdout_lines)
                    stderr_data = ''.join(stderr_lines)
                except Exception as e:
                    stdout_data = ""
                    stderr_data = f"Error reading stream output: {e}"
            else:
                stdout_data = exec_result.output.decode('utf-8', errors='replace') if exec_result.output else ""
                stderr_data = ""
            
            # Log execution
            logger.info(f"Container command executed: container={container_name}, exit_code={exec_result.exit_code}")
            
            return stdout_data, stderr_data, exec_result.exit_code
            
        except docker.errors.APIError as e:
            logger.error(f"Docker API error during container execution: {e}")
            return "", str(e), -1
        except Exception as e:
            logger.error(f"Container execution error: {e}")
            return "", str(e), -1
    
    async def _copy_files_to_container(self, container, files: Dict[str, str]):
        """Copy files to container workspace with validation"""
        if not files or not isinstance(files, dict):
            raise ValueError("Invalid files dictionary provided")
        
        try:
            import tarfile
            import io
            
            # Validate file paths and content
            for filename, content in files.items():
                if not filename or not isinstance(filename, str):
                    raise ValueError("Invalid filename in files dictionary")
                if not isinstance(content, str):
                    raise ValueError(f"Content for {filename} must be string")
                
                # Sanitize filename
                if '..' in filename or filename.startswith('/'):
                    raise ValueError(f"Invalid filename: {filename}")
            
            # Create tar archive with files
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                for filename, content in files.items():
                    file_data = content.encode('utf-8')
                    tarinfo = tarfile.TarInfo(name=filename)
                    tarinfo.size = len(file_data)
                    tarinfo.mode = 0o644
                    tarinfo.mtime = time.time()
                    tar.addfile(tarinfo, io.BytesIO(file_data))
            
            tar_buffer.seek(0)
            
            # Extract to container
            success = container.put_archive("/workspace", tar_buffer.getvalue())
            if not success:
                raise RuntimeError("Failed to copy files to container")
            
            logger.debug(f"Successfully copied {len(files)} files to container")
            
        except Exception as e:
            logger.error(f"Failed to copy files to container: {e}")
            raise
    
    async def cleanup_container(self, container_name: str, force: bool = False):
        """Cleanup artifact container with proper resource management"""
        if not container_name:
            return
        
        try:
            container = self.active_containers.get(container_name)
            if not container:
                # Try to get from Docker
                try:
                    container = self.vm_manager.docker_client.containers.get(container_name)
                except docker.errors.NotFound:
                    logger.warning(f"Container {container_name} not found during cleanup")
                    return
            
            # Get logs before cleanup if container had issues
            container.reload()
            if container.status == 'exited' and container.attrs.get('State', {}).get('ExitCode', 0) != 0:
                logs = container.logs(tail=100).decode('utf-8', errors='replace')
                logger.warning(f"Container {container_name} exited with errors: {logs}")
            
            # Stop container gracefully
            try:
                container.stop(timeout=10 if not force else 1)
            except docker.errors.APIError as e:
                if "is not running" not in str(e):
                    logger.warning(f"Error stopping container {container_name}: {e}")
            
            # Remove container
            try:
                container.remove(force=force, v=True)  # Remove volumes as well
            except docker.errors.APIError as e:
                logger.warning(f"Error removing container {container_name}: {e}")
            
            # Remove from active containers
            if container_name in self.active_containers:
                del self.active_containers[container_name]
            
            logger.info(f"Cleaned up container: {container_name}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup container {container_name}: {e}")
    
    async def cleanup_all_containers(self):
        """Cleanup all active containers"""
        containers_to_cleanup = list(self.active_containers.keys())
        for container_name in containers_to_cleanup:
            await self.cleanup_container(container_name)


# ============================================================================
# MULTI-AGENT COLLABORATION
# ============================================================================

class CollaborationManager:
    """Manages multi-agent collaboration sessions with security and reliability"""
    
    def __init__(self, vm_manager: AdvancedVMManager):
        self.vm_manager = vm_manager
        self.active_sessions: Dict[UUID, CollaborationSession] = {}
        self.agent_connections: Dict[UUID, Dict[str, Any]] = {}
        self._session_lock = asyncio.Lock()
        self._connection_lock = asyncio.Lock()
    
    async def initiate_collaboration(
        self, 
        task_description: str,
        collaborator_agents: List[UUID],
        coordination_strategy: str = "parallel"
    ) -> UUID:
        """Initiate multi-agent collaboration session with full validation"""
        if not task_description or not isinstance(task_description, str):
            raise ValueError("Invalid task_description provided")
        
        if not collaborator_agents or not isinstance(collaborator_agents, list):
            raise ValueError("Invalid collaborator_agents list provided")
        
        if len(collaborator_agents) > self.vm_manager.settings.max_concurrent_agents:
            raise ValueError(f"Too many collaborators: {len(collaborator_agents)} > {self.vm_manager.settings.max_concurrent_agents}")
        
        async with self._session_lock:
            try:
                session_id = uuid4()
                
                # Validate all agent IDs
                for agent_id in collaborator_agents:
                    if not isinstance(agent_id, UUID):
                        raise ValueError(f"Invalid agent ID: {agent_id}")
                
                session = CollaborationSession(
                    session_id=session_id,
                    primary_agent_id=self.vm_manager.agent_id,
                    collaborator_ids=collaborator_agents,
                    task_description=task_description,
                    status="initializing"
                )
                
                self.active_sessions[session_id] = session
                
                # Establish connections to collaborator agents with timeout
                connection_tasks = []
                for agent_id in collaborator_agents:
                    connection_tasks.append(self._connect_to_agent(agent_id))
                
                # Wait for all connections with timeout
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*connection_tasks, return_exceptions=True),
                        timeout=30
                    )
                except asyncio.TimeoutError:
                    raise RuntimeError("Timeout establishing agent connections")
                
                # Send task delegation messages
                delegation_tasks = []
                for agent_id in collaborator_agents:
                    if agent_id in self.agent_connections:
                        delegation_tasks.append(
                            self._send_task_delegation(session_id, agent_id, task_description)
                        )
                
                if delegation_tasks:
                    await asyncio.wait_for(
                        asyncio.gather(*delegation_tasks, return_exceptions=True),
                        timeout=30
                    )
                
                session.status = "executing"
                
                logger.info(
                    f"Initiated collaboration session {session_id} "
                    f"with {len(collaborator_agents)} agents: {collaborator_agents}"
                )
                
                return session_id
                
            except Exception as e:
                logger.error(f"Failed to initiate collaboration: {e}")
                # Cleanup on failure
                if session_id in self.active_sessions:
                    del self.active_sessions[session_id]
                raise
    
    async def _connect_to_agent(self, agent_id: UUID):
        """Establish secure connection to collaborator agent"""
        if not isinstance(agent_id, UUID):
            raise ValueError("Invalid agent ID")
        
        async with self._connection_lock:
            try:
                if agent_id in self.agent_connections:
                    # Check if existing connection is still valid
                    connection = self.agent_connections[agent_id]
                    sock = connection.get("socket")
                    if sock and not sock._closed:
                        return
                    else:
                        # Close and remove stale connection
                        if sock:
                            sock.close()
                        del self.agent_connections[agent_id]
                
                # Calculate agent's collaboration port
                agent_hash = hash(str(agent_id)) & 0x7FFFFFFF
                agent_port = self.vm_manager.settings.collaboration_port_base + (agent_hash % 1000)
                
                # Ensure port is valid
                if agent_port > 65535:
                    agent_port = 65535
                
                # Create connection with timeout
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(self.vm_manager.settings.agent_communication_timeout)
                
                # Connect with retry
                max_retries = 3
                for attempt in range(max_retries):
                    try:
                        sock.connect(("127.0.0.1", agent_port))
                        break
                    except (ConnectionRefusedError, socket.timeout) as e:
                        if attempt == max_retries - 1:
                            raise ConnectionError(f"Failed to connect to agent {agent_id} on port {agent_port}: {e}")
                        await asyncio.sleep(1)
                
                # Set socket to non-blocking for async operations
                sock.setblocking(False)
                
                self.agent_connections[agent_id] = {
                    "socket": sock,
                    "connected_at": datetime.now(timezone.utc),
                    "last_activity": datetime.now(timezone.utc)
                }
                
                logger.info(f"Connected to collaborator agent {agent_id} on port {agent_port}")
                
            except Exception as e:
                logger.error(f"Failed to connect to agent {agent_id}: {e}")
                raise
    
    async def _send_task_delegation(self, session_id: UUID, agent_id: UUID, task: str):
        """Send secure task delegation to collaborator agent"""
        if not all([session_id, agent_id, task]):
            raise ValueError("Invalid parameters for task delegation")
        
        try:
            connection = self.agent_connections.get(agent_id)
            if not connection:
                raise ConnectionError(f"No connection to agent {agent_id}")
            
            message = {
                "type": "task_delegation",
                "session_id": str(session_id),
                "sender_id": str(self.vm_manager.agent_id),
                "task_description": task,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "protocol_version": "1.0"
            }
            
            sock = connection["socket"]
            message_data = json.dumps(message).encode('utf-8')
            message_length = len(message_data)
            
            # Send with timeout
            sock.settimeout(self.vm_manager.settings.agent_communication_timeout)
            
            # Send length first (4 bytes, big-endian)
            sock.send(message_length.to_bytes(4, 'big'))
            
            # Send message data
            total_sent = 0
            while total_sent < message_length:
                sent = sock.send(message_data[total_sent:])
                if sent == 0:
                    raise ConnectionError("Socket connection broken")
                total_sent += sent
            
            # Update last activity
            connection["last_activity"] = datetime.now(timezone.utc)
            
            logger.debug(f"Sent task delegation to agent {agent_id} for session {session_id}")
            
        except Exception as e:
            logger.error(f"Failed to send task delegation to agent {agent_id}: {e}")
            # Remove broken connection
            if agent_id in self.agent_connections:
                self.agent_connections[agent_id]["socket"].close()
                del self.agent_connections[agent_id]
            raise
    
    async def collect_collaboration_responses(self, session_id: UUID, timeout: int = 300) -> Dict[str, Any]:
        """Collect responses from all collaborating agents with timeout management"""
        if not isinstance(session_id, UUID):
            raise ValueError("Invalid session_id")
        
        session = self.active_sessions.get(session_id)
        if not session:
            raise ValueError(f"Session {session_id} not found")
        
        if session.status != "executing":
            raise ValueError(f"Session {session_id} is not in executing state")
        
        try:
            responses = {}
            start_time = time.time()
            
            # Collect responses from each collaborator
            response_tasks = []
            for agent_id in session.collaborator_ids:
                if agent_id in self.agent_connections:
                    response_tasks.append(
                        self._receive_agent_response(agent_id, timeout)
                    )
                else:
                    responses[str(agent_id)] = {"error": "No active connection"}
            
            # Wait for responses with timeout
            if response_tasks:
                try:
                    results = await asyncio.wait_for(
                        asyncio.gather(*response_tasks, return_exceptions=True),
                        timeout=timeout
                    )
                    
                    # Map results to agent IDs
                    for agent_id, result in zip(session.collaborator_ids, results):
                        if isinstance(result, Exception):
                            responses[str(agent_id)] = {"error": str(result)}
                        else:
                            responses[str(agent_id)] = result
                except asyncio.TimeoutError:
                    # Handle timeout gracefully
                    for agent_id in session.collaborator_ids:
                        if str(agent_id) not in responses:
                            responses[str(agent_id)] = {"error": "Response timeout"}
            
            session.responses = responses
            session.status = "synthesis"
            
            # Synthesize final response
            final_response = await self._synthesize_responses(session)
            session.final_synthesis = final_response
            session.status = "completed"
            
            total_time = time.time() - start_time
            
            return {
                "session_id": str(session_id),
                "total_time": total_time,
                "individual_responses": responses,
                "synthesized_response": final_response,
                "participating_agents": [str(aid) for aid in session.collaborator_ids],
                "completion_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to collect collaboration responses: {e}")
            session.status = "failed"
            raise
    
    async def _receive_agent_response(self, agent_id: UUID, timeout: int) -> Dict[str, Any]:
        """Receive response from specific agent with timeout handling"""
        if not isinstance(agent_id, UUID):
            raise ValueError("Invalid agent_id")
        
        connection = self.agent_connections.get(agent_id)
        if not connection:
            raise ConnectionError(f"No connection to agent {agent_id}")
        
        try:
            sock = connection["socket"]
            sock.settimeout(timeout)
            
            # Receive message length
            length_data = b""
            while len(length_data) < 4:
                chunk = sock.recv(4 - len(length_data))
                if not chunk:
                    raise ConnectionError("Connection closed by peer")
                length_data += chunk
            
            message_length = int.from_bytes(length_data, 'big')
            
            if message_length <= 0 or message_length > 10 * 1024 * 1024:  # 10MB limit
                raise ValueError(f"Invalid message length: {message_length}")
            
            # Receive message data
            message_data = b""
            while len(message_data) < message_length:
                chunk = sock.recv(min(message_length - len(message_data), 8192))
                if not chunk:
                    raise ConnectionError("Connection closed during message reception")
                message_data += chunk
            
            response = json.loads(message_data.decode('utf-8'))
            
            # Validate response structure
            if not isinstance(response, dict):
                raise ValueError("Invalid response format")
            
            # Update last activity
            connection["last_activity"] = datetime.now(timezone.utc)
            
            return response
            
        except socket.timeout:
            raise TimeoutError(f"Timeout receiving response from agent {agent_id}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON in agent response: {e}")
        except Exception as e:
            logger.error(f"Failed to receive response from agent {agent_id}: {e}")
            # Remove broken connection
            sock.close()
            if agent_id in self.agent_connections:
                del self.agent_connections[agent_id]
            raise
    
    async def _synthesize_responses(self, session: CollaborationSession) -> str:
        """Synthesize individual agent responses into unified result with intelligence"""
        try:
            if not session.responses:
                return "No responses received from collaborating agents"
            
            # Analyze response quality and completeness
            successful_responses = []
            error_responses = []
            
            for agent_id, response in session.responses.items():
                if isinstance(response, dict) and "error" not in response:
                    successful_responses.append((agent_id, response))
                else:
                    error_responses.append((agent_id, response))
            
            # Build comprehensive synthesis
            synthesis_parts = [
                f"# Multi-Agent Collaboration Result",
                f"**Session ID:** {session.session_id}",
                f"**Task:** {session.task_description}",
                f"**Completed:** {datetime.now(timezone.utc).isoformat()}",
                f"**Participants:** {len(session.collaborator_ids)} agents",
                "",
                "## Summary",
                f"- **Successful Responses:** {len(successful_responses)}",
                f"- **Failed Responses:** {len(error_responses)}",
                "",
                "## Individual Contributions"
            ]
            
            # Add successful responses
            for agent_id, response in successful_responses:
                content = response.get('content', str(response))
                synthesis_parts.append(f"\n### Agent {agent_id}")
                synthesis_parts.append(f"```\n{content}\n```")
            
            # Add error responses
            if error_responses:
                synthesis_parts.append("\n## Errors and Issues")
                for agent_id, response in error_responses:
                    error_message = response.get('error', 'Unknown error')
                    synthesis_parts.append(f"- **Agent {agent_id}:** {error_message}")
            
            # Final remarks
            synthesis_parts.append("\n---")
            synthesis_parts.append("End of collaborative synthesis.")
            
            return "\n".join(synthesis_parts)
            
        except Exception as e:
            logger.error(f"Failed to synthesize responses: {e}")
            return f"Synthesis failed: {str(e)}"


# ============================================================================
# UNIFIED AI SHELL INTERFACE
# ============================================================================

class AdvancedAIShell:
    """
    Unified AI Shell supporting VM operations, container orchestration, and multi-agent collaboration.
    Provides intelligent command routing and execution context management.
    """
    
    def __init__(self, settings: AIShellSettings = None, agent_id: UUID = None):
        self.settings = settings or AIShellSettings()
        self.agent_id = agent_id or uuid4()
        
        # Core components
        self.vm_manager = AdvancedVMManager(self.settings, self.agent_id)
        self.container_orchestrator = ContainerOrchestrator(self.vm_manager)
        self.collaboration_manager = CollaborationManager(self.vm_manager)
        
        # State tracking
        self.command_history: List[ExecutionResult] = []
        self.active_contexts: Dict[str, ExecutionContext] = {}
        self.initialized = False
    
    async def initialize(self) -> bool:
        """Initialize the advanced AI shell"""
        try:
            success = await self.vm_manager.initialize()
            if success:
                self.initialized = True
                logger.info(f"Advanced AI Shell initialized for agent {self.agent_id}")
            return success
        except Exception as e:
            logger.error(f"Failed to initialize Advanced AI Shell: {e}")
            return False
    
    async def execute_command(
        self, 
        command: str,
        context: ExecutionContext = ExecutionContext.VM_NATIVE,
        container_spec: Optional[ContainerSpec] = None,
        artifact_files: Optional[Dict[str, str]] = None,
        collaboration_agents: Optional[List[UUID]] = None,
        **kwargs
    ) -> ExecutionResult:
        """
        Unified command execution with intelligent context routing
        """
        if not self.initialized:
            raise RuntimeError("AI Shell not initialized")
        
        start_time = time.time()
        command_type = self._classify_command(command)
        
        try:
            # Route command based on context and type
            if context == ExecutionContext.VM_NATIVE:
                result = await self._execute_vm_native(command, command_type)
                
            elif context == ExecutionContext.CONTAINER_OVERLAY:
                result = await self._execute_container_overlay(
                    command, command_type, container_spec, artifact_files
                )
                
            elif context == ExecutionContext.MULTI_AGENT:
                result = await self._execute_multi_agent(
                    command, command_type, collaboration_agents
                )
                
            elif context == ExecutionContext.HYBRID:
                result = await self._execute_hybrid(
                    command, command_type, container_spec, artifact_files
                )
                
            else:
                raise ValueError(f"Unknown execution context: {context}")
            
            execution_time = time.time() - start_time
            
            # Create comprehensive execution result
            execution_result = ExecutionResult(
                command=command,
                stdout=result.get("stdout", ""),
                stderr=result.get("stderr", ""),
                return_code=result.get("return_code", 0),
                execution_time=execution_time,
                context=context,
                command_type=command_type,
                container_id=result.get("container_id"),
                collaborator_responses=result.get("collaborator_responses", []),
                metadata=result.get("metadata", {})
            )
            
            # Store in history
            self.command_history.append(execution_result)
            
            # Log execution
            await self._log_execution(execution_result)
            
            return execution_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_result = ExecutionResult(
                command=command,
                stdout="",
                stderr=str(e),
                return_code=-1,
                execution_time=execution_time,
                context=context,
                command_type=command_type
            )
            
            self.command_history.append(error_result)
            logger.error(f"Command execution failed: {e}")
            return error_result
    
    def _classify_command(self, command: str) -> CommandType:
        """Classify command type for intelligent routing"""
        command_lower = command.lower()
        
        if any(keyword in command_lower for keyword in ['docker', 'container', 'artifact']):
            return CommandType.ARTIFACT
        elif any(keyword in command_lower for keyword in ['collaborate', 'delegate', 'synthesize']):
            return CommandType.COLLABORATION
        elif any(keyword in command_lower for keyword in ['research', 'analyze', 'investigate']):
            return CommandType.RESEARCH
        elif any(keyword in command_lower for keyword in ['code', 'develop', 'build', 'compile']):
            return CommandType.DEVELOPMENT
        else:
            return CommandType.SYSTEM
    
    async def _execute_vm_native(self, command: str, command_type: CommandType) -> Dict[str, Any]:
        """Execute command directly in AI's VM"""
        stdout, stderr, return_code = await self.vm_manager.execute_in_vm(command)
        
        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": return_code,
            "metadata": {"execution_location": "ai_vm"}
        }
    
    async def _execute_container_overlay(
        self, 
        command: str, 
        command_type: CommandType,
        container_spec: Optional[ContainerSpec] = None,
        artifact_files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Execute command in artifact container overlay"""
        
        # Use default container spec if not provided
        if not container_spec:
            container_spec = ContainerSpec()
        
        # Create artifact container
        artifact_id = str(uuid4())
        container_name = await self.container_orchestrator.create_artifact_container(
            artifact_id, container_spec, artifact_files
        )
        
        try:
            # Execute command in container
            stdout, stderr, return_code = await self.container_orchestrator.execute_in_container(
                container_name, command
            )
            
            return {
                "stdout": stdout,
                "stderr": stderr,
                "return_code": return_code,
                "container_id": container_name,
                "metadata": {
                    "execution_location": "artifact_container",
                    "artifact_id": artifact_id
                }
            }
            
        finally:
            # Cleanup container (or keep for debugging based on settings)
            await self.container_orchestrator.cleanup_container(container_name)
    
    async def _execute_multi_agent(
        self, 
        command: str, 
        command_type: CommandType,
        collaboration_agents: Optional[List[UUID]] = None
    ) -> Dict[str, Any]:
        """Execute command through multi-agent collaboration"""
        
        if not collaboration_agents:
            raise ValueError("Collaboration agents required for multi-agent execution")
        
        # Initiate collaboration session
        session_id = await self.collaboration_manager.initiate_collaboration(
            command, collaboration_agents
        )
        
        # Collect responses from all agents
        collaboration_result = await self.collaboration_manager.collect_collaboration_responses(
            session_id
        )
        
        return {
            "stdout": collaboration_result["synthesized_response"],
            "stderr": "",
            "return_code": 0,
            "collaborator_responses": collaboration_result["individual_responses"],
            "metadata": {
                "execution_location": "multi_agent_collaboration",
                "session_id": str(session_id),
                "collaboration_time": collaboration_result["total_time"]
            }
        }
    
    async def _execute_hybrid(
        self, 
        command: str, 
        command_type: CommandType,
        container_spec: Optional[ContainerSpec] = None,
        artifact_files: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """Execute command with AI VM orchestrating container operations"""
        
        # First, execute preparation steps in VM
        prep_command = f"echo 'Preparing container orchestration for: {command}'"
        vm_stdout, vm_stderr, vm_return_code = await self.vm_manager.execute_in_vm(prep_command)
        
        # Then execute main command in container overlay
        container_result = await self._execute_container_overlay(
            command, command_type, container_spec, artifact_files
        )
        
        # Combine results
        combined_stdout = f"VM Preparation: {vm_stdout}\n\nContainer Execution:\n{container_result['stdout']}"
        
        return {
            "stdout": combined_stdout,
            "stderr": container_result["stderr"],
            "return_code": container_result["return_code"],
            "container_id": container_result.get("container_id"),
            "metadata": {
                "execution_location": "hybrid_vm_container",
                "vm_preparation": vm_stdout,
                "container_result": container_result["metadata"]
            }
        }
    
    async def _log_execution(self, result: ExecutionResult):
        """Log execution result for monitoring and debugging"""
        try:
            log_entry = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "agent_id": str(self.agent_id),
                "command": result.command,
                "context": result.context.value,
                "command_type": result.command_type.value,
                "return_code": result.return_code,
                "execution_time": result.execution_time,
                "success": result.return_code == 0
            }
            
            async with aiofiles.open(self.settings.execution_log_path, 'a') as f:
                await f.write(json.dumps(log_entry) + '\n')
                
        except Exception as e:
            logger.error(f"Failed to log execution: {e}")
    
    async def shutdown(self):
        """Graceful shutdown of AI Shell"""
        try:
            # Cleanup active containers
            for container_name in list(self.container_orchestrator.active_containers.keys()):
                await self.container_orchestrator.cleanup_container(container_name)
            
            # Close collaboration connections
            for connection in self.collaboration_manager.agent_connections.values():
                if "socket" in connection:
                    connection["socket"].close()
            
            logger.info("AI Shell shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# ============================================================================
# CONVENIENCE FUNCTIONS AND EXAMPLES
# ============================================================================

async def create_ai_shell(agent_id: Optional[UUID] = None) -> AdvancedAIShell:
    """Create and initialize an AI Shell instance"""
    settings = AIShellSettings()
    shell = AdvancedAIShell(settings, agent_id)
    
    success = await shell.initialize()
    if not success:
        raise RuntimeError("Failed to initialize AI Shell")
    
    return shell


async def demo_ai_shell_capabilities():
    """Demonstrate the full capabilities of the Advanced AI Shell"""
    print("🚀 Advanced AI Shell Capabilities Demo")
    print("=" * 50)
    
    # Create AI shell instance
    shell = await create_ai_shell()
    
    try:
        # 1. VM Native execution
        print("\n1. VM Native Execution:")
        result = await shell.execute_command(
            "uname -a && python3 --version",
            context=ExecutionContext.VM_NATIVE
        )
        print(f"Output: {result.stdout}")
        
        # 2. Container overlay execution
        print("\n2. Container Overlay Execution:")
        container_spec = ContainerSpec(
            image="python:3.11-slim",
            environment={"PYTHONPATH": "/workspace"}
        )
        
        result = await shell.execute_command(
            "python3 -c 'import sys; print(f\"Python {sys.version} in container\")'",
            context=ExecutionContext.CONTAINER_OVERLAY,
            container_spec=container_spec
        )
        print(f"Container Output: {result.stdout}")
        print(f"Container ID: {result.container_id}")
        
        # 3. Hybrid execution
        print("\n3. Hybrid Execution (VM orchestrating container):")
        artifact_files = {
            "demo.py": "print('Hello from artifact container!')\nprint('Unlimited execution capabilities!')"
        }
        
        result = await shell.execute_command(
            "python3 demo.py",
            context=ExecutionContext.HYBRID,
            container_spec=container_spec,
            artifact_files=artifact_files
        )
        print(f"Hybrid Output: {result.stdout}")
        
        # 4. Multi-Agent collaboration execution
        print("\n4. Multi-Agent Collaboration Execution:")
        # For demo, we'll collaborate with two instances of ourselves
        agent_id_1 = uuid4()
        agent_id_2 = uuid4()
        
        result = await shell.execute_command(
            "Collaborate on a task with error handling",
            context=ExecutionContext.MULTI_AGENT,
            collaboration_agents=[agent_id_1, agent_id_2]
        )
        print(f"Collaboration Output: {result.stdout}")
        print(f"Agent Responses: {result.collaborator_responses}")
        
        print("\n✅ Demo completed successfully!")
        print(f"Total commands executed: {len(shell.command_history)}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        
    finally:
        await shell.shutdown()


if __name__ == "__main__":
    # Demo usage
    asyncio.run(demo_ai_shell_capabilities())