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
import paramiko
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
        self.ssh_client = None
        self.docker_client = None
        self.active_containers: Dict[str, Any] = {}
        self.collaboration_socket: Optional[socket.socket] = None
        
    async def initialize(self) -> bool:
        """Initialize VM and container connections"""
        try:
            # Establish SSH connection to AI's VM
            await self._connect_to_vm()
            
            # Initialize Docker client for container orchestration
            await self._initialize_docker_client()
            
            # Setup collaboration networking if needed
            await self._setup_collaboration_network()
            
            logger.info(f"AI Shell initialized for agent {self.agent_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize AI Shell: {e}")
            return False
    
    async def _connect_to_vm(self):
        """Establish persistent SSH connection to AI's VM (offloaded to thread)"""
        if self.ssh_client and self.ssh_client.get_transport() and self.ssh_client.get_transport().is_active():
            return

        def _connect_sync():
            import paramiko as _paramiko
            client = _paramiko.SSHClient()
            client.set_missing_host_key_policy(_paramiko.AutoAddPolicy())
            client.connect(
                hostname=self.settings.vm_host,
                port=self.settings.vm_port,
                username=self.settings.vm_user,
                password=self.settings.vm_password,
                key_filename=self.settings.vm_ssh_key_path,
                timeout=30
            )
            return client

        try:
            logger.info(f"Connecting to AI VM at {self.settings.vm_host}:{self.settings.vm_port}")
            self.ssh_client = await asyncio.to_thread(_connect_sync)
            logger.info("SSH connection to AI VM established")
        except Exception as e:
            logger.error(f"Failed to connect to AI VM: {e}")
            raise
    
    async def _initialize_docker_client(self):
        """Initialize Docker client for container orchestration (offloaded to thread)"""
        def _init_docker_sync():
            import docker as _docker
            client = _docker.from_env()
            client.ping()
            try:
                client.networks.get(self.settings.container_network)
            except _docker.errors.NotFound:
                client.networks.create(
                    self.settings.container_network,
                    driver="bridge",
                    options={"com.docker.network.bridge.enable_icc": "true"}
                )
            return client

        try:
            self.docker_client = await asyncio.to_thread(_init_docker_sync)
            logger.info("Docker client initialized for container orchestration")
        except Exception as e:
            logger.error(f"Failed to initialize Docker client: {e}")
            raise
    
    async def _setup_collaboration_network(self):
        """Setup networking for multi-agent collaboration"""
        try:
            # Create listening socket for agent communication
            collaboration_port = self.settings.collaboration_port_base + hash(str(self.agent_id)) % 1000
            
            self.collaboration_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.collaboration_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.collaboration_socket.bind(("127.0.0.1", collaboration_port))
            self.collaboration_socket.listen(5)
            
            logger.info(f"Collaboration network setup on port {collaboration_port}")
            
        except Exception as e:
            logger.warning(f"Failed to setup collaboration network: {e}")
    
    async def execute_in_vm(self, command: str, **kwargs) -> Tuple[str, str, int]:
        """Execute command directly in AI's VM (offloaded to thread)"""
        if not self.ssh_client:
            await self._connect_to_vm()

        def _exec_sync():
            stdin, stdout, stderr = self.ssh_client.exec_command(
                command,
                timeout=self.settings.command_timeout
            )
            exit_status = stdout.channel.recv_exit_status()
            stdout_data = stdout.read().decode('utf-8', errors='replace').strip()
            stderr_data = stderr.read().decode('utf-8', errors='replace').strip()
            return stdout_data, stderr_data, exit_status

        try:
            return await asyncio.to_thread(_exec_sync)
        except Exception as e:
            logger.error(f"VM execution error: {e}")
            await self._connect_to_vm()
            return "", str(e), -1


# ============================================================================
# CONTAINER ORCHESTRATION
# ============================================================================

class ContainerOrchestrator:
    """Orchestrates container overlays for artifact execution"""
    
    def __init__(self, vm_manager: AdvancedVMManager):
        self.vm_manager = vm_manager
        self.active_containers: Dict[str, docker.models.containers.Container] = {}
    
    async def create_artifact_container(
        self, 
        artifact_id: str, 
        spec: ContainerSpec,
        files: Dict[str, str] = None
    ) -> str:
        """Create specialized container for artifact execution"""
        try:
            container_name = f"artifact_{artifact_id}_{int(time.time())}"
            container_config = {
                "image": spec.image,
                "name": container_name,
                "detach": True,
                "network": self.vm_manager.settings.container_network,
                "working_dir": spec.working_dir,
                "environment": spec.environment,
                "volumes": spec.volumes,
                "auto_remove": False,
                "stdin_open": True,
                "tty": True
            }
            if spec.cpu_limit or spec.memory_limit:
                container_config["cpu_period"] = 100000
                if spec.cpu_limit:
                    container_config["cpu_quota"] = int(float(spec.cpu_limit) * 100000)
                if spec.memory_limit:
                    container_config["mem_limit"] = spec.memory_limit
            if spec.gpu_access:
                container_config["device_requests"] = [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]

            def _run_container_sync():
                return self.vm_manager.docker_client.containers.run(**container_config)

            container = await asyncio.to_thread(_run_container_sync)

            if files:
                await self._copy_files_to_container(container, files)

            self.active_containers[container_name] = container
            logger.info(f"Created artifact container: {container_name}")
            return container_name

        except Exception as e:
            logger.error(f"Failed to create artifact container: {e}")
            raise

    async def execute_in_container(
        self, 
        container_name: str, 
        command: str,
        stream_output: bool = False
    ) -> Tuple[str, str, int]:
        """Execute command in specific container (offloaded to thread)"""
        try:
            container = self.active_containers.get(container_name)
            if not container:
                raise ValueError(f"Container {container_name} not found")

            def _exec_once():
                res = container.exec_run(
                    cmd=["bash", "-c", command],
                    stdout=True,
                    stderr=True,
                    stream=False,
                    demux=False
                )
                out = res.output.decode('utf-8', errors='replace') if res.output else ""
                return out, "", res.exit_code

            if not stream_output:
                return await asyncio.to_thread(_exec_once)
            else:
                def _exec_stream():
                    res = container.exec_run(
                        cmd=["bash", "-c", command],
                        stdout=True,
                        stderr=True,
                        stream=True,
                        demux=True
                    )
                    stdout_lines, stderr_lines = [], []
                    for out_pair in res.output:
                        if out_pair[0]:
                            stdout_lines.append(out_pair[0].decode('utf-8', errors='replace'))
                        if out_pair[1]:
                            stderr_lines.append(out_pair[1].decode('utf-8', errors='replace'))
                    return ''.join(stdout_lines), ''.join(stderr_lines), res.exit_code

                return await asyncio.to_thread(_exec_stream)

        except Exception as e:
            logger.error(f"Container execution error: {e}")
            return "", str(e), -1

    async def _copy_files_to_container(self, container, files: Dict[str, str]):
        """Copy files to container workspace (offloaded to thread)"""
        try:
            import tarfile
            import io
            tar_buffer = io.BytesIO()
            with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                for filename, content in files.items():
                    file_data = content.encode('utf-8')
                    tarinfo = tarfile.TarInfo(name=filename)
                    tarinfo.size = len(file_data)
                    tar.addfile(tarinfo, io.BytesIO(file_data))
            tar_buffer.seek(0)

            def _put_archive_sync():
                container.put_archive("/workspace", tar_buffer.getvalue())

            await asyncio.to_thread(_put_archive_sync)
        except Exception as e:
            logger.error(f"Failed to copy files to container: {e}")
    
    async def cleanup_container(self, container_name: str):
        """Cleanup artifact container"""
        try:
            container = self.active_containers.get(container_name)
            if container:
                container.stop(timeout=10)
                container.remove()
                del self.active_containers[container_name]
                logger.info(f"Cleaned up container: {container_name}")
        except Exception as e:
            logger.error(f"Failed to cleanup container {container_name}: {e}")


# ============================================================================
# MULTI-AGENT COLLABORATION
# ============================================================================

class CollaborationManager:
    """Manages multi-agent collaboration sessions"""
    
    def __init__(self, vm_manager: AdvancedVMManager):
        self.vm_manager = vm_manager
        self.active_sessions: Dict[UUID, CollaborationSession] = {}
        self.agent_connections: Dict[UUID, Dict[str, Any]] = {}
    
    async def initiate_collaboration(
        self, 
        task_description: str,
        collaborator_agents: List[UUID],
        coordination_strategy: str = "parallel"
    ) -> UUID:
        """Initiate multi-agent collaboration session"""
        try:
            session_id = uuid4()
            
            session = CollaborationSession(
                session_id=session_id,
                primary_agent_id=self.vm_manager.agent_id,
                collaborator_ids=collaborator_agents,
                task_description=task_description,
                status="initializing"
            )
            
            self.active_sessions[session_id] = session
            
            # Establish connections to collaborator agents
            for agent_id in collaborator_agents:
                await self._connect_to_agent(agent_id)
            
            # Send task delegation messages
            for agent_id in collaborator_agents:
                await self._send_task_delegation(session_id, agent_id, task_description)
            
            session.status = "executing"
            
            logger.info(f"Initiated collaboration session {session_id} with {len(collaborator_agents)} agents")
            return session_id
            
        except Exception as e:
            logger.error(f"Failed to initiate collaboration: {e}")
            raise
    
    async def _connect_to_agent(self, agent_id: UUID):
        """Establish connection to collaborator agent"""
        try:
            # Calculate agent's collaboration port
            agent_port = self.vm_manager.settings.collaboration_port_base + hash(str(agent_id)) % 1000
            
            # Create connection
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(self.vm_manager.settings.agent_communication_timeout)
            sock.connect(("127.0.0.1", agent_port))
            
            self.agent_connections[agent_id] = {
                "socket": sock,
                "connected_at": datetime.now(timezone.utc)
            }
            
            logger.info(f"Connected to collaborator agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to connect to agent {agent_id}: {e}")
    
    async def _send_task_delegation(self, session_id: UUID, agent_id: UUID, task: str):
        """Send task delegation to collaborator agent"""
        try:
            message = {
                "type": "task_delegation",
                "session_id": str(session_id),
                "sender_id": str(self.vm_manager.agent_id),
                "task_description": task,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            connection = self.agent_connections.get(agent_id)
            if connection:
                sock = connection["socket"]
                message_data = json.dumps(message).encode('utf-8')
                sock.send(len(message_data).to_bytes(4, 'big'))  # Send length first
                sock.send(message_data)
                
                logger.debug(f"Sent task delegation to agent {agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to send task delegation to agent {agent_id}: {e}")
    
    async def collect_collaboration_responses(self, session_id: UUID, timeout: int = 300) -> Dict[str, Any]:
        """Collect responses from all collaborating agents"""
        try:
            session = self.active_sessions.get(session_id)
            if not session:
                raise ValueError(f"Session {session_id} not found")
            
            responses = {}
            start_time = time.time()
            
            # Collect responses from each collaborator
            for agent_id in session.collaborator_ids:
                try:
                    response = await self._receive_agent_response(agent_id, timeout)
                    responses[str(agent_id)] = response
                except Exception as e:
                    logger.error(f"Failed to receive response from agent {agent_id}: {e}")
                    responses[str(agent_id)] = {"error": str(e)}
            
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
                "participating_agents": [str(aid) for aid in session.collaborator_ids]
            }
            
        except Exception as e:
            logger.error(f"Failed to collect collaboration responses: {e}")
            raise
    
    async def _receive_agent_response(self, agent_id: UUID, timeout: int) -> Dict[str, Any]:
        """Receive response from specific agent"""
        try:
            connection = self.agent_connections.get(agent_id)
            if not connection:
                raise ValueError(f"No connection to agent {agent_id}")
            
            sock = connection["socket"]
            sock.settimeout(timeout)
            
            # Receive message length
            length_data = sock.recv(4)
            if len(length_data) != 4:
                raise ValueError("Invalid message format")
            
            message_length = int.from_bytes(length_data, 'big')
            
            # Receive message data
            message_data = b""
            while len(message_data) < message_length:
                chunk = sock.recv(message_length - len(message_data))
                if not chunk:
                    raise ValueError("Connection closed")
                message_data += chunk
            
            response = json.loads(message_data.decode('utf-8'))
            return response
            
        except Exception as e:
            logger.error(f"Failed to receive response from agent {agent_id}: {e}")
            raise
    
    async def _synthesize_responses(self, session: CollaborationSession) -> str:
        """Synthesize individual agent responses into unified result"""
        try:
            # Simple synthesis - in production, this would use an LLM
            synthesis_parts = [
                f"Collaborative Task: {session.task_description}",
                "",
                "Individual Agent Contributions:"
            ]
            
            for agent_id, response in session.responses.items():
                if "error" not in response:
                    synthesis_parts.append(f"Agent {agent_id}: {response.get('content', 'No content')}")
                else:
                    synthesis_parts.append(f"Agent {agent_id}: Error - {response['error']}")
            
            synthesis_parts.append("")
            synthesis_parts.append("Synthesized Result: Task completed through multi-agent collaboration.")
            
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
            # Ensure directory exists
            log_dir = os.path.dirname(self.settings.execution_log_path)
            if log_dir:
                os.makedirs(log_dir, exist_ok=True)

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
            
            # Close SSH connection
            if self.vm_manager.ssh_client:
                self.vm_manager.ssh_client.close()
            
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
        
        print("\n✅ Demo completed successfully!")
        print(f"Total commands executed: {len(shell.command_history)}")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        
    finally:
        await shell.shutdown()


if __name__ == "__main__":
    # Demo usage
    asyncio.run(demo_ai_shell_capabilities())