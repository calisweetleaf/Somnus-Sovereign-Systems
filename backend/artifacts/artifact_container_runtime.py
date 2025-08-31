"""
SOMNUS SOVEREIGN SYSTEMS - Complete Artifact Container Runtime Integration
One Unlimited Container Per Artifact - Complete Isolation + Unlimited Power

ARCHITECTURE:
- Each artifact gets its own Docker container (disposable supercomputer)
- Container provides security isolation, not artificial limits
- Real-time WebSocket streaming between container and UI
- Complete integration with unlimited artifact system
- Support for all capabilities: ML training, video processing, YouTube downloads
- Resource monitoring without restrictions
- File management and persistence
- VM bridge integration
"""

import asyncio
import base64
import json
import logging
import os
import shutil
import signal
import subprocess
import tempfile
import threading
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple, AsyncGenerator
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from contextlib import asynccontextmanager
import tarfile
import io

# Core dependencies
import docker
import psutil
import aiofiles
from fastapi import (
    FastAPI, HTTPException, WebSocket, WebSocketDisconnect,
    BackgroundTasks, Request, Response, status, Depends, APIRouter, File, UploadFile
)
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from pydantic import BaseModel, Field, validator
from starlette.websockets import WebSocketState

# Import our unlimited artifact system
from complete_artifact_system import (
    UnlimitedArtifact, UnlimitedArtifactManager, ArtifactType, 
    ExecutionEnvironment, ArtifactCapability, UnlimitedExecutionResult,
    UnlimitedExecutionConfig
)

# Somnus system imports (when integrated)
try:
    from schemas.session import SessionID, UserID
    from core.memory_core import MemoryManager, MemoryType, MemoryImportance
    from backend.virtual_machine.vm_supervisor import VMInstanceManager
    SOMNUS_INTEGRATION = True
except ImportError:
    # Standalone mode
    SessionID = str
    UserID = str
    SOMNUS_INTEGRATION = False

logger = logging.getLogger(__name__)


# ============================================================================
# CONTAINER RUNTIME MODELS
# ============================================================================

class ContainerState(str, Enum):
    """Container lifecycle states"""
    CREATING = "creating"
    BUILDING = "building"
    STARTING = "starting"
    RUNNING = "running"
    EXECUTING = "executing"
    PAUSED = "paused"
    STOPPED = "stopped"
    ERROR = "error"
    DESTROYED = "destroyed"


@dataclass
class ContainerConfig:
    """Container configuration - NO LIMITS"""
    # Container specifications (unlimited by default)
    image_name: str = "somnus-artifact:unlimited"
    cpu_limit: Optional[float] = None      # No CPU limit
    memory_limit: Optional[str] = None     # No memory limit
    disk_limit: Optional[str] = None       # No disk limit
    
    # Network and security
    network_mode: str = "bridge"           # Full network access
    privileged: bool = True                # Full privileges for unlimited power
    security_opt: List[str] = field(default_factory=list)
    
    # Resource access
    gpu_enabled: bool = True               # GPU access by default
    internet_enabled: bool = True          # Internet access by default
    host_network: bool = False             # Option for host networking
    
    # Ports and services
    exposed_ports: List[int] = field(default_factory=lambda: [8000, 8080, 3000, 5000, 6006, 8888])
    port_mappings: Dict[int, int] = field(default_factory=dict)
    
    # Volumes and file system
    workspace_path: str = "/workspace"
    shared_volumes: List[str] = field(default_factory=list)
    tmpfs_mounts: Dict[str, str] = field(default_factory=lambda: {"/tmp": "size=1G"})
    
    # Environment
    environment_vars: Dict[str, str] = field(default_factory=lambda: {
        "PYTHONUNBUFFERED": "1",
        "DEBIAN_FRONTEND": "noninteractive",
        "CUDA_VISIBLE_DEVICES": "all",
        "NVIDIA_VISIBLE_DEVICES": "all"
    })
    
    # Execution
    working_dir: str = "/workspace"
    user: str = "root"                     # Root access for unlimited power
    shell: str = "/bin/bash"


@dataclass
class ContainerMetrics:
    """Real-time container metrics"""
    # Basic metrics
    cpu_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_limit_mb: float = 0.0
    memory_percent: float = 0.0
    
    # Storage metrics
    disk_usage_mb: float = 0.0
    disk_limit_mb: float = 0.0
    disk_percent: float = 0.0
    
    # Network metrics
    network_rx_mb: float = 0.0
    network_tx_mb: float = 0.0
    network_rx_packets: int = 0
    network_tx_packets: int = 0
    
    # GPU metrics (if available)
    gpu_utilization: float = 0.0
    gpu_memory_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    
    # Process metrics
    processes: int = 0
    threads: int = 0
    file_descriptors: int = 0
    
    # Container state
    uptime_seconds: int = 0
    restart_count: int = 0
    exit_code: Optional[int] = None
    
    # Timestamps
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ContainerExecutionContext:
    """Context for container execution"""
    artifact_id: UUID
    container_id: str
    container_name: str
    config: ContainerConfig
    state: ContainerState
    created_at: datetime
    
    # Docker objects
    docker_container: Optional[docker.models.containers.Container] = None
    
    # Execution state
    current_executions: List[str] = field(default_factory=list)
    execution_history: List[Dict] = field(default_factory=list)
    
    # WebSocket connections
    websocket_connections: Set[WebSocket] = field(default_factory=set)
    
    # Metrics
    metrics: ContainerMetrics = field(default_factory=ContainerMetrics)
    metrics_history: List[ContainerMetrics] = field(default_factory=list)


# ============================================================================
# UNLIMITED CONTAINER RUNTIME
# ============================================================================

class UnlimitedContainerRuntime:
    """
    Container runtime for unlimited artifact execution
    Each artifact gets its own disposable supercomputer
    """
    
    def __init__(
        self,
        artifact_manager: UnlimitedArtifactManager,
        vm_manager: Optional[Any] = None,
        memory_manager: Optional[Any] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        self.artifact_manager = artifact_manager
        self.vm_manager = vm_manager
        self.memory_manager = memory_manager
        self.config = config or {}
        
        # Docker client
        try:
            self.docker_client = docker.from_env()
            self.docker_available = True
        except Exception as e:
            logger.error(f"Docker not available: {e}")
            self.docker_client = None
            self.docker_available = False
        
        # Container tracking
        self.active_containers: Dict[UUID, ContainerExecutionContext] = {}
        self.container_images: Dict[str, str] = {}  # Image cache
        
        # Background tasks
        self.metrics_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.metrics_running = False
        
        # WebSocket management
        self.global_websockets: Set[WebSocket] = set()
        
        logger.info("Unlimited Container Runtime initialized")
    
    async def initialize(self):
        """Initialize the container runtime"""
        if not self.docker_available:
            raise RuntimeError("Docker is required for container runtime")
        
        try:
            # Build base images
            await self._build_base_images()
            
            # Start background monitoring
            await self._start_background_tasks()
            
            # Clean up any orphaned containers
            await self._cleanup_orphaned_containers()
            
            logger.info("Container runtime fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize container runtime: {e}")
            raise
    
    async def _build_base_images(self):
        """Build base Docker images for artifacts"""
        
        # Base unlimited image with all capabilities
        unlimited_dockerfile = '''
FROM nvidia/cuda:12.2-devel-ubuntu22.04

# Prevent interactive installations
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies - EVERYTHING
RUN apt-get update && apt-get install -y \\
    # Basic tools
    curl wget git vim nano htop tmux screen \\
    build-essential cmake pkg-config \\
    # Programming languages
    python3 python3-pip python3-dev \\
    nodejs npm yarn \\
    openjdk-17-jdk \\
    golang-go \\
    rustc cargo \\
    # Media processing
    ffmpeg imagemagick \\
    # ML/AI tools
    nvidia-container-runtime \\
    # Network tools
    net-tools iputils-ping telnet netcat \\
    # Additional utilities
    tree jq zip unzip rar p7zip-full \\
    # Database clients
    mysql-client postgresql-client redis-tools \\
    # Development tools
    gdb valgrind strace \\
    # Compression
    lz4 zstd \\
    && rm -rf /var/lib/apt/lists/*

# Install Python packages - ALL THE ML/AI TOOLS
RUN pip3 install --no-cache-dir \\
    # Core ML frameworks
    torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 \\
    tensorflow[and-cuda] \\
    jax[cuda] -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html \\
    # Transformers and NLP
    transformers datasets tokenizers accelerate \\
    sentence-transformers \\
    # Data science
    numpy pandas scikit-learn scipy \\
    matplotlib seaborn plotly \\
    # Computer vision
    opencv-python pillow \\
    # Audio processing
    librosa soundfile \\
    # Video processing
    moviepy \\
    # Web and APIs
    requests beautifulsoup4 scrapy \\
    fastapi uvicorn websockets \\
    # Jupyter and notebooks
    jupyter jupyterlab ipywidgets \\
    # Utilities
    tqdm rich typer click \\
    # YouTube and media
    yt-dlp youtube-dl \\
    # Database connectors
    pymongo psycopg2-binary mysql-connector-python \\
    # Cloud and distributed
    ray dask \\
    # Monitoring
    wandb tensorboard \\
    # Development
    black flake8 pytest \\
    # Additional ML
    xgboost lightgbm catboost \\
    optuna hyperopt \\
    # Crypto and blockchain
    web3 ecdsa \\
    # Everything else
    boto3 google-cloud-storage azure-storage-blob

# Install Node.js packages globally
RUN npm install -g \\
    typescript ts-node \\
    @angular/cli \\
    create-react-app \\
    vue-cli \\
    express \\
    webpack webpack-cli \\
    pm2

# Install additional tools
RUN curl -fsSL https://get.docker.com | sh
RUN go install github.com/go-delve/delve/cmd/dlv@latest

# Create workspace
WORKDIR /workspace
RUN chmod 777 /workspace

# Setup entrypoint
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

# Expose common ports
EXPOSE 8000 8080 3000 5000 6006 8888

ENTRYPOINT ["/entrypoint.sh"]
CMD ["/bin/bash"]
'''
        
        # Entrypoint script
        entrypoint_script = '''#!/bin/bash
set -e

# Setup environment
export PATH="/usr/local/go/bin:$PATH"
export PYTHONPATH="/workspace:$PYTHONPATH"

# Start SSH daemon if requested
if [ "$ENABLE_SSH" = "true" ]; then
    service ssh start
fi

# Setup GPU if available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader)"
    export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-all}
fi

# Execute the command
exec "$@"
'''
        
        # Build image
        try:
            build_path = Path(tempfile.mkdtemp(prefix="somnus_build_"))
            
            # Write Dockerfile
            (build_path / "Dockerfile").write_text(unlimited_dockerfile)
            (build_path / "entrypoint.sh").write_text(entrypoint_script)
            
            logger.info("Building unlimited artifact container image...")
            
            image, build_logs = self.docker_client.images.build(
                path=str(build_path),
                tag="somnus-artifact:unlimited",
                rm=True,
                forcerm=True,
                pull=True,
                nocache=False
            )
            
            self.container_images["unlimited"] = image.id
            
            logger.info(f"Built unlimited container image: {image.id[:12]}")
            
            # Log build progress
            for log in build_logs:
                if 'stream' in log:
                    logger.debug(f"Build: {log['stream'].strip()}")
            
        except Exception as e:
            logger.error(f"Failed to build container image: {e}")
            raise
        
        finally:
            # Cleanup build directory
            try:
                shutil.rmtree(build_path)
            except:
                pass
    
    async def create_artifact_container(
        self, 
        artifact: UnlimitedArtifact,
        config: Optional[ContainerConfig] = None
    ) -> ContainerExecutionContext:
        """Create container for artifact execution"""
        
        if not self.docker_available:
            raise RuntimeError("Docker not available")
        
        artifact_id = artifact.metadata.artifact_id
        
        # Use provided config or create default
        if config is None:
            config = ContainerConfig()
        
        # Generate container name
        container_name = f"somnus_artifact_{artifact_id.hex[:12]}"
        
        try:
            logger.info(f"Creating container for artifact {artifact_id}")
            
            # Prepare container volumes
            workspace_volume = f"somnus_workspace_{artifact_id.hex}"
            
            # Create container specification
            container_spec = await self._create_container_spec(
                artifact, config, container_name, workspace_volume
            )
            
            # Create container
            container = self.docker_client.containers.create(**container_spec)
            
            # Create execution context
            context = ContainerExecutionContext(
                artifact_id=artifact_id,
                container_id=container.id,
                container_name=container_name,
                config=config,
                state=ContainerState.CREATING,
                created_at=datetime.now(timezone.utc),
                docker_container=container
            )
            
            # Start container
            container.start()
            context.state = ContainerState.STARTING
            
            # Wait for container to be ready
            await self._wait_for_container_ready(container)
            context.state = ContainerState.RUNNING
            
            # Write artifact files to container
            await self._write_artifact_files_to_container(artifact, container)
            
            # Store context
            self.active_containers[artifact_id] = context
            
            logger.info(f"Container {container.id[:12]} created and running for artifact {artifact_id}")
            
            return context
            
        except Exception as e:
            logger.error(f"Failed to create artifact container: {e}")
            context.state = ContainerState.ERROR
            raise HTTPException(status_code=500, detail=f"Container creation failed: {str(e)}")
    
    async def _create_container_spec(
        self,
        artifact: UnlimitedArtifact,
        config: ContainerConfig,
        container_name: str,
        workspace_volume: str
    ) -> Dict[str, Any]:
        """Create container specification"""
        
        # Base specification
        spec = {
            "image": config.image_name,
            "name": container_name,
            "detach": True,
            "tty": True,
            "stdin_open": True,
            "working_dir": config.working_dir,
            "user": config.user,
            
            # Environment variables
            "environment": {
                **config.environment_vars,
                "ARTIFACT_ID": str(artifact.metadata.artifact_id),
                "ARTIFACT_NAME": artifact.metadata.name,
                "ARTIFACT_TYPE": artifact.metadata.artifact_type.value,
                "SESSION_ID": artifact.metadata.session_id,
                "USER_ID": artifact.metadata.created_by,
                "SOMNUS_UNLIMITED": "true"
            },
            
            # Volumes
            "volumes": {
                workspace_volume: {"bind": config.workspace_path, "mode": "rw"}
            },
            
            # Labels for management
            "labels": {
                "somnus.artifact_id": str(artifact.metadata.artifact_id),
                "somnus.session_id": artifact.metadata.session_id,
                "somnus.user_id": artifact.metadata.created_by,
                "somnus.created_at": datetime.now(timezone.utc).isoformat(),
                "somnus.unlimited": "true"
            }
        }
        
        # Resource limits (or lack thereof for unlimited)
        if config.cpu_limit:
            spec["cpu_quota"] = int(config.cpu_limit * 100000)
            spec["cpu_period"] = 100000
        
        if config.memory_limit:
            spec["mem_limit"] = config.memory_limit
        
        # Network configuration
        if config.internet_enabled:
            if config.host_network:
                spec["network_mode"] = "host"
            else:
                spec["network_mode"] = config.network_mode
                
                # Port mappings
                if config.exposed_ports:
                    spec["ports"] = {}
                    for port in config.exposed_ports:
                        host_port = config.port_mappings.get(port)
                        spec["ports"][f"{port}/tcp"] = host_port
        else:
            spec["network_mode"] = "none"
        
        # GPU support
        if config.gpu_enabled:
            try:
                spec["device_requests"] = [
                    docker.types.DeviceRequest(count=-1, capabilities=[["gpu"]])
                ]
            except Exception as e:
                logger.warning(f"GPU support not available: {e}")
        
        # Security configuration
        if config.privileged:
            spec["privileged"] = True
        
        if config.security_opt:
            spec["security_opt"] = config.security_opt
        
        # Additional volumes
        for volume in config.shared_volumes:
            spec["volumes"][volume] = {"bind": volume, "mode": "rw"}
        
        # Tmpfs mounts
        if config.tmpfs_mounts:
            spec["tmpfs"] = config.tmpfs_mounts
        
        return spec
    
    async def _wait_for_container_ready(self, container: docker.models.containers.Container, timeout: int = 60):
        """Wait for container to be ready"""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                container.reload()
                if container.status == "running":
                    # Test basic connectivity
                    result = container.exec_run(["echo", "container_ready"], timeout=5)
                    if result.exit_code == 0:
                        return
                        
            except Exception as e:
                logger.debug(f"Container readiness check: {e}")
            
            await asyncio.sleep(1)
        
        raise TimeoutError(f"Container failed to become ready within {timeout} seconds")
    
    async def _write_artifact_files_to_container(
        self,
        artifact: UnlimitedArtifact,
        container: docker.models.containers.Container
    ):
        """Write all artifact files to container"""
        try:
            # Create tar archive with all files
            tar_buffer = io.BytesIO()
            
            with tarfile.open(fileobj=tar_buffer, mode='w') as tar:
                for filename, artifact_file in artifact.files.items():
                    if artifact_file.binary_data:
                        # Binary file
                        file_data = artifact_file.binary_data
                    else:
                        # Text file
                        file_data = artifact_file.content.encode(artifact_file.encoding)
                    
                    # Create tar info
                    tarinfo = tarfile.TarInfo(name=filename)
                    tarinfo.size = len(file_data)
                    tarinfo.mode = 0o755 if artifact_file.is_executable else 0o644
                    
                    tar.addfile(tarinfo, io.BytesIO(file_data))
            
            tar_buffer.seek(0)
            
            # Extract to container workspace
            container.put_archive("/workspace", tar_buffer.getvalue())
            
            logger.debug(f"Wrote {len(artifact.files)} files to container {container.id[:12]}")
            
        except Exception as e:
            logger.error(f"Failed to write artifact files to container: {e}")
            raise
    
    async def execute_in_container(
        self,
        artifact_id: UUID,
        command: str,
        stream_output: bool = True,
        timeout: Optional[int] = None,
        environment: Optional[Dict[str, str]] = None
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """Execute command in artifact container with real-time streaming"""
        
        context = self.active_containers.get(artifact_id)
        if not context or not context.docker_container:
            raise HTTPException(status_code=404, detail="Container not found")
        
        container = context.docker_container
        context.state = ContainerState.EXECUTING
        
        try:
            # Prepare environment
            exec_env = {}
            if environment:
                exec_env.update(environment)
            
            logger.info(f"Executing in container {container.id[:12]}: {command}")
            
            # Create execution
            exec_id = container.client.api.exec_create(
                container.id,
                cmd=["bash", "-c", command],
                stdout=True,
                stderr=True,
                stdin=False,
                tty=True,
                environment=exec_env,
                workdir="/workspace"
            )
            
            # Track execution
            execution_uuid = str(uuid4())
            context.current_executions.append(execution_uuid)
            
            # Start execution
            exec_stream = container.client.api.exec_start(
                exec_id["Id"],
                stream=True,
                socket=False
            )
            
            # Yield execution started event
            yield {
                "type": "execution_started",
                "execution_id": execution_uuid,
                "command": command,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            output_buffer = ""
            error_buffer = ""
            
            # Stream output in real-time
            for chunk in exec_stream:
                if chunk:
                    try:
                        decoded_chunk = chunk.decode('utf-8', errors='replace')
                        output_buffer += decoded_chunk
                        
                        if stream_output:
                            yield {
                                "type": "output",
                                "execution_id": execution_uuid,
                                "data": decoded_chunk,
                                "timestamp": datetime.now(timezone.utc).isoformat()
                            }
                    except Exception as e:
                        logger.debug(f"Stream decoding error: {e}")
            
            # Get execution result
            exec_info = container.client.api.exec_inspect(exec_id["Id"])
            exit_code = exec_info["ExitCode"]
            
            # Final result
            yield {
                "type": "execution_completed",
                "execution_id": execution_uuid,
                "exit_code": exit_code,
                "success": exit_code == 0,
                "output": output_buffer,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            # Store execution history
            context.execution_history.append({
                "execution_id": execution_uuid,
                "command": command,
                "exit_code": exit_code,
                "output": output_buffer,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            
        except Exception as e:
            logger.error(f"Container execution failed: {e}")
            yield {
                "type": "execution_error",
                "execution_id": execution_uuid,
                "error": str(e),
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
        
        finally:
            # Clean up execution tracking
            if execution_uuid in context.current_executions:
                context.current_executions.remove(execution_uuid)
            
            context.state = ContainerState.RUNNING
    
    async def get_container_files(self, artifact_id: UUID, path: str = "/workspace") -> List[Dict[str, Any]]:
        """Get file listing from container"""
        context = self.active_containers.get(artifact_id)
        if not context or not context.docker_container:
            raise HTTPException(status_code=404, detail="Container not found")
        
        container = context.docker_container
        
        try:
            # List files using ls command
            result = container.exec_run([
                "find", path, "-type", "f", "-exec", "ls", "-la", "{}", "+"
            ])
            
            if result.exit_code != 0:
                return []
            
            files = []
            output_lines = result.output.decode('utf-8').strip().split('\n')
            
            for line in output_lines:
                if line.strip():
                    parts = line.split()
                    if len(parts) >= 9:
                        file_info = {
                            "name": parts[8],
                            "path": parts[8],
                            "size": int(parts[4]),
                            "modified": " ".join(parts[5:8]),
                            "permissions": parts[0],
                            "type": "file",
                            "is_executable": "x" in parts[0]
                        }
                        files.append(file_info)
            
            return files
            
        except Exception as e:
            logger.error(f"Failed to list container files: {e}")
            return []
    
    async def read_container_file(self, artifact_id: UUID, file_path: str) -> Union[str, bytes]:
        """Read file from container"""
        context = self.active_containers.get(artifact_id)
        if not context or not context.docker_container:
            raise HTTPException(status_code=404, detail="Container not found")
        
        container = context.docker_container
        
        try:
            # Try to read as text first
            result = container.exec_run(["cat", file_path])
            
            if result.exit_code == 0:
                return result.output.decode('utf-8', errors='replace')
            else:
                raise HTTPException(status_code=404, detail=f"File not found: {file_path}")
                
        except Exception as e:
            logger.error(f"Failed to read container file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def write_container_file(
        self, 
        artifact_id: UUID, 
        file_path: str, 
        content: Union[str, bytes],
        is_binary: bool = False
    ) -> bool:
        """Write file to container"""
        context = self.active_containers.get(artifact_id)
        if not context or not context.docker_container:
            return False
        
        container = context.docker_container
        
        try:
            if is_binary and isinstance(content, bytes):
                # Handle binary files
                import tempfile
                with tempfile.NamedTemporaryFile() as tmp_file:
                    tmp_file.write(content)
                    tmp_file.flush()
                    
                    # Copy to container
                    with open(tmp_file.name, 'rb') as f:
                        container.put_archive("/workspace", [(file_path, f.read())])
            else:
                # Handle text files
                if isinstance(content, bytes):
                    content = content.decode('utf-8')
                
                # Write using shell redirection
                escaped_content = content.replace("'", "'\"'\"'")
                command = f"cat > '{file_path}' << 'EOF'\n{escaped_content}\nEOF"
                result = container.exec_run(["bash", "-c", command])
                
                return result.exit_code == 0
                
        except Exception as e:
            logger.error(f"Failed to write container file: {e}")
            return False
    
    async def get_container_metrics(self, artifact_id: UUID) -> Optional[ContainerMetrics]:
        """Get real-time container metrics"""
        context = self.active_containers.get(artifact_id)
        if not context or not context.docker_container:
            return None
        
        container = context.docker_container
        
        try:
            # Get container stats
            stats = container.stats(stream=False)
            
            # Calculate metrics
            metrics = ContainerMetrics()
            
            # CPU metrics
            cpu_stats = stats.get("cpu_stats", {})
            precpu_stats = stats.get("precpu_stats", {})
            
            cpu_delta = cpu_stats.get("cpu_usage", {}).get("total_usage", 0) - \
                       precpu_stats.get("cpu_usage", {}).get("total_usage", 0)
            system_delta = cpu_stats.get("system_cpu_usage", 0) - \
                          precpu_stats.get("system_cpu_usage", 0)
            
            if system_delta > 0 and cpu_delta > 0:
                num_cpus = len(cpu_stats.get("cpu_usage", {}).get("percpu_usage", [1]))
                metrics.cpu_percent = (cpu_delta / system_delta) * num_cpus * 100
            
            # Memory metrics
            memory_stats = stats.get("memory_stats", {})
            memory_usage = memory_stats.get("usage", 0)
            memory_limit = memory_stats.get("limit", 1)
            
            metrics.memory_usage_mb = memory_usage / (1024 * 1024)
            metrics.memory_limit_mb = memory_limit / (1024 * 1024)
            metrics.memory_percent = (memory_usage / memory_limit) * 100
            
            # Network metrics
            networks = stats.get("networks", {})
            rx_bytes = sum(net.get("rx_bytes", 0) for net in networks.values())
            tx_bytes = sum(net.get("tx_bytes", 0) for net in networks.values())
            
            metrics.network_rx_mb = rx_bytes / (1024 * 1024)
            metrics.network_tx_mb = tx_bytes / (1024 * 1024)
            
            # Process metrics
            pids_stats = stats.get("pids_stats", {})
            metrics.processes = pids_stats.get("current", 0)
            
            # Container state
            container.reload()
            created_time = datetime.fromisoformat(container.attrs["Created"].replace("Z", "+00:00"))
            metrics.uptime_seconds = int((datetime.now(timezone.utc) - created_time).total_seconds())
            
            # GPU metrics (if available)
            try:
                gpu_result = container.exec_run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
                if gpu_result.exit_code == 0:
                    gpu_lines = gpu_result.output.decode().strip().split('\n')
                    if gpu_lines and gpu_lines[0]:
                        gpu_parts = gpu_lines[0].split(',')
                        if len(gpu_parts) >= 3:
                            metrics.gpu_utilization = float(gpu_parts[0].strip())
                            metrics.gpu_memory_mb = float(gpu_parts[1].strip())
                            metrics.gpu_memory_total_mb = float(gpu_parts[2].strip())
            except:
                pass  # GPU metrics not available
            
            # Update context
            context.metrics = metrics
            context.metrics_history.append(metrics)
            
            # Keep only recent metrics history
            if len(context.metrics_history) > 100:
                context.metrics_history = context.metrics_history[-100:]
            
            return metrics
            
        except Exception as e:
            logger.error(f"Failed to get container metrics: {e}")
            return None
    
    async def stop_container(self, artifact_id: UUID, save_state: bool = True) -> bool:
        """Stop and clean up container"""
        context = self.active_containers.get(artifact_id)
        if not context or not context.docker_container:
            return False
        
        container = context.docker_container
        
        try:
            logger.info(f"Stopping container {container.id[:12]} for artifact {artifact_id}")
            
            # Save state if requested
            if save_state:
                await self._save_container_state(context)
            
            # Stop container gracefully
            container.stop(timeout=30)
            context.state = ContainerState.STOPPED
            
            # Remove container
            container.remove(force=True, v=False)  # Keep volumes for potential restore
            context.state = ContainerState.DESTROYED
            
            # Close WebSocket connections
            for ws in context.websocket_connections.copy():
                try:
                    await ws.close()
                except:
                    pass
            
            # Remove from active containers
            self.active_containers.pop(artifact_id, None)
            
            logger.info(f"Container {container.id[:12]} stopped and cleaned up")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            context.state = ContainerState.ERROR
            return False
    
    async def _save_container_state(self, context: ContainerExecutionContext):
        """Save container state for potential restore"""
        try:
            container = context.docker_container
            
            # Create snapshot
            snapshot_name = f"somnus_snapshot_{context.artifact_id.hex[:12]}_{int(time.time())}"
            container.commit(
                repository="somnus-snapshots",
                tag=snapshot_name,
                message=f"Snapshot of artifact {context.artifact_id}"
            )
            
            # Save state metadata
            state_data = {
                "snapshot_tag": f"somnus-snapshots:{snapshot_name}",
                "artifact_id": str(context.artifact_id),
                "created_at": datetime.now(timezone.utc).isoformat(),
                "execution_history": context.execution_history,
                "metrics": context.metrics.__dict__ if context.metrics else {}
            }
            
            # Store in memory manager if available
            if self.memory_manager and SOMNUS_INTEGRATION:
                await self.memory_manager.store_memory(
                    user_id=context.config.environment_vars.get("USER_ID", "unknown"),
                    content=f"Container state snapshot for artifact",
                    memory_type=MemoryType.ARTIFACT_STATE,
                    importance=MemoryImportance.MEDIUM,
                    metadata=state_data
                )
            
            logger.info(f"Saved container state snapshot: {snapshot_name}")
            
        except Exception as e:
            logger.error(f"Failed to save container state: {e}")
    
    async def _start_background_tasks(self):
        """Start background monitoring tasks"""
        self.metrics_running = True
        self.metrics_task = asyncio.create_task(self._metrics_monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_monitoring_loop())
        
        logger.info("Background monitoring tasks started")
    
    async def _metrics_monitoring_loop(self):
        """Background metrics monitoring"""
        while self.metrics_running:
            try:
                for artifact_id in list(self.active_containers.keys()):
                    await self.get_container_metrics(artifact_id)
                
                await asyncio.sleep(5)  # Update every 5 seconds
                
            except Exception as e:
                logger.error(f"Metrics monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _cleanup_monitoring_loop(self):
        """Background cleanup monitoring"""
        while self.metrics_running:
            try:
                await asyncio.sleep(300)  # Check every 5 minutes
                await self._cleanup_orphaned_containers()
                
            except Exception as e:
                logger.error(f"Cleanup monitoring error: {e}")
    
    async def _cleanup_orphaned_containers(self):
        """Clean up orphaned containers"""
        try:
            containers = self.docker_client.containers.list(
                all=True,
                filters={"label": "somnus.unlimited=true"}
            )
            
            cleaned_count = 0
            
            for container in containers:
                try:
                    # Check container age
                    created_at_str = container.labels.get("somnus.created_at")
                    if created_at_str:
                        created_at = datetime.fromisoformat(created_at_str.replace("Z", "+00:00"))
                        age = datetime.now(timezone.utc) - created_at
                        
                        # Remove containers older than 24 hours
                        if age > timedelta(hours=24):
                            container.remove(force=True)
                            cleaned_count += 1
                            logger.info(f"Cleaned up orphaned container: {container.name}")
                
                except Exception as e:
                    logger.error(f"Error cleaning up container {container.name}: {e}")
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} orphaned containers")
                
        except Exception as e:
            logger.error(f"Orphaned container cleanup failed: {e}")
    
    async def add_websocket(self, artifact_id: UUID, websocket: WebSocket):
        """Add WebSocket for real-time updates"""
        context = self.active_containers.get(artifact_id)
        if context:
            context.websocket_connections.add(websocket)
        
        self.global_websockets.add(websocket)
    
    async def remove_websocket(self, artifact_id: UUID, websocket: WebSocket):
        """Remove WebSocket"""
        context = self.active_containers.get(artifact_id)
        if context:
            context.websocket_connections.discard(websocket)
        
        self.global_websockets.discard(websocket)
    
    async def broadcast_to_artifact(self, artifact_id: UUID, message: Dict[str, Any]):
        """Broadcast message to all WebSockets for artifact"""
        context = self.active_containers.get(artifact_id)
        if not context:
            return
        
        connections = context.websocket_connections.copy()
        
        for websocket in connections:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
                else:
                    context.websocket_connections.discard(websocket)
            except Exception as e:
                logger.debug(f"Failed to send WebSocket message: {e}")
                context.websocket_connections.discard(websocket)
    
    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down Unlimited Container Runtime...")
        
        # Stop background tasks
        self.metrics_running = False
        
        if self.metrics_task:
            self.metrics_task.cancel()
        if self.cleanup_task:
            self.cleanup_task.cancel()
        
        # Stop all containers
        shutdown_tasks = []
        for artifact_id in list(self.active_containers.keys()):
            task = asyncio.create_task(self.stop_container(artifact_id, save_state=True))
            shutdown_tasks.append(task)
        
        if shutdown_tasks:
            await asyncio.gather(*shutdown_tasks, return_exceptions=True)
        
        # Close WebSocket connections
        for ws in self.global_websockets.copy():
            try:
                await ws.close()
            except:
                pass
        
        logger.info("Container runtime shutdown complete")


# ============================================================================
# FASTAPI INTEGRATION
# ============================================================================

def create_container_runtime_router(
    container_runtime: UnlimitedContainerRuntime
) -> APIRouter:
    """Create FastAPI router for container runtime"""
    
    router = APIRouter(prefix="/api/container", tags=["container-runtime"])
    
    # Request models
    class CreateContainerRequest(BaseModel):
        artifact_id: str
        config: Optional[Dict[str, Any]] = None
    
    class ExecuteRequest(BaseModel):
        command: str
        environment: Optional[Dict[str, str]] = None
        timeout: Optional[int] = None
    
    class WriteFileRequest(BaseModel):
        file_path: str
        content: str
        is_binary: bool = False
    
    @router.post("/{artifact_id}/create")
    async def create_container(artifact_id: str, request: CreateContainerRequest):
        """Create container for artifact"""
        try:
            artifact = await container_runtime.artifact_manager.get_artifact(UUID(artifact_id))
            if not artifact:
                raise HTTPException(status_code=404, detail="Artifact not found")
            
            # Create container config
            config = ContainerConfig()
            if request.config:
                # Update config with provided values
                for key, value in request.config.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            context = await container_runtime.create_artifact_container(artifact, config)
            
            return {
                "container_id": context.container_id,
                "container_name": context.container_name,
                "state": context.state.value,
                "created_at": context.created_at.isoformat(),
                "unlimited_power": True
            }
            
        except Exception as e:
            logger.error(f"Failed to create container: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/{artifact_id}/execute")
    async def execute_command(artifact_id: str, request: ExecuteRequest):
        """Execute command in container with streaming"""
        try:
            async def stream_execution():
                async for result in container_runtime.execute_in_container(
                    UUID(artifact_id),
                    request.command,
                    stream_output=True,
                    timeout=request.timeout,
                    environment=request.environment
                ):
                    yield f"data: {json.dumps(result)}\\n\\n"
            
            return StreamingResponse(
                stream_execution(),
                media_type="text/plain",
                headers={"Cache-Control": "no-cache"}
            )
            
        except Exception as e:
            logger.error(f"Failed to execute command: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{artifact_id}/files")
    async def list_files(artifact_id: str, path: str = "/workspace"):
        """List files in container"""
        try:
            files = await container_runtime.get_container_files(UUID(artifact_id), path)
            return {"files": files}
            
        except Exception as e:
            logger.error(f"Failed to list files: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{artifact_id}/files/read")
    async def read_file(artifact_id: str, file_path: str):
        """Read file from container"""
        try:
            content = await container_runtime.read_container_file(UUID(artifact_id), file_path)
            return {"content": content}
            
        except Exception as e:
            logger.error(f"Failed to read file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.post("/{artifact_id}/files/write")
    async def write_file(artifact_id: str, request: WriteFileRequest):
        """Write file to container"""
        try:
            success = await container_runtime.write_container_file(
                UUID(artifact_id),
                request.file_path,
                request.content,
                request.is_binary
            )
            
            return {"success": success}
            
        except Exception as e:
            logger.error(f"Failed to write file: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.get("/{artifact_id}/metrics")
    async def get_metrics(artifact_id: str):
        """Get container metrics"""
        try:
            metrics = await container_runtime.get_container_metrics(UUID(artifact_id))
            
            if metrics:
                return {
                    "metrics": {
                        "cpu_percent": metrics.cpu_percent,
                        "memory_usage_mb": metrics.memory_usage_mb,
                        "memory_percent": metrics.memory_percent,
                        "network_rx_mb": metrics.network_rx_mb,
                        "network_tx_mb": metrics.network_tx_mb,
                        "gpu_utilization": metrics.gpu_utilization,
                        "processes": metrics.processes,
                        "uptime_seconds": metrics.uptime_seconds,
                        "last_updated": metrics.last_updated.isoformat()
                    }
                }
            else:
                raise HTTPException(status_code=404, detail="Metrics not available")
                
        except Exception as e:
            logger.error(f"Failed to get metrics: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.delete("/{artifact_id}")
    async def stop_container(artifact_id: str, save_state: bool = True):
        """Stop container"""
        try:
            success = await container_runtime.stop_container(UUID(artifact_id), save_state)
            return {"success": success}
            
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @router.websocket("/{artifact_id}/stream")
    async def websocket_stream(websocket: WebSocket, artifact_id: str):
        """WebSocket for real-time container interaction"""
        await websocket.accept()
        
        try:
            await container_runtime.add_websocket(UUID(artifact_id), websocket)
            
            # Send initial state
            context = container_runtime.active_containers.get(UUID(artifact_id))
            if context:
                await websocket.send_json({
                    "type": "container_state",
                    "container_id": context.container_id,
                    "state": context.state.value,
                    "unlimited_power": True
                })
            
            # Handle incoming messages
            while True:
                try:
                    data = await websocket.receive_json()
                    message_type = data.get("type")
                    
                    if message_type == "execute":
                        # Stream command execution
                        command = data.get("command", "")
                        async for result in container_runtime.execute_in_container(
                            UUID(artifact_id),
                            command,
                            stream_output=True
                        ):
                            await websocket.send_json(result)
                    
                    elif message_type == "get_metrics":
                        # Send current metrics
                        metrics = await container_runtime.get_container_metrics(UUID(artifact_id))
                        if metrics:
                            await websocket.send_json({
                                "type": "metrics",
                                "data": metrics.__dict__
                            })
                
                except WebSocketDisconnect:
                    break
                except Exception as e:
                    logger.error(f"WebSocket error: {e}")
                    await websocket.send_json({
                        "type": "error",
                        "message": str(e)
                    })
        
        finally:
            await container_runtime.remove_websocket(UUID(artifact_id), websocket)
    
    return router


# ============================================================================
# INTEGRATION FACTORY
# ============================================================================

async def create_integrated_artifact_system(
    vm_manager: Optional[Any] = None,
    memory_manager: Optional[Any] = None,
    config: Optional[Dict[str, Any]] = None
) -> Tuple[UnlimitedArtifactManager, UnlimitedContainerRuntime, APIRouter]:
    """
    Create complete integrated artifact system with container runtime
    """
    
    # Create artifact manager
    artifact_manager = UnlimitedArtifactManager(
        storage_dir="data/unlimited_artifacts",
        enable_persistence=True,
        enable_collaboration=True,
        max_artifacts_per_session=None  # UNLIMITED
    )
    
    await artifact_manager.initialize()
    
    # Create container runtime
    container_runtime = UnlimitedContainerRuntime(
        artifact_manager=artifact_manager,
        vm_manager=vm_manager,
        memory_manager=memory_manager,
        config=config or {}
    )
    
    await container_runtime.initialize()
    
    # Create integrated router
    router = create_container_runtime_router(container_runtime)
    
    logger.info("Integrated artifact system with container runtime created successfully")
    
    return artifact_manager, container_runtime, router


# ============================================================================
# DEMO AND TESTING
# ============================================================================

async def demo_container_integration():
    """Demonstrate container integration"""
    
    print("🎨 Somnus Container Runtime Integration Demo")
    print("=" * 50)
    
    # Create integrated system
    artifact_manager, container_runtime, router = await create_integrated_artifact_system()
    
    # Create unlimited artifact
    artifact = await artifact_manager.create_artifact(
        name="Container Integration Demo",
        artifact_type=ArtifactType.PYTHON,
        created_by="demo_user",
        session_id="demo_session",
        description="Demonstrate unlimited container execution",
        initial_content='''#!/usr/bin/env python3
"""
Unlimited Container Execution Demo
"""
import os
import sys
import subprocess
import torch
import time

print("🐳 CONTAINER EXECUTION STARTED")
print("=" * 40)

print(f"Python version: {sys.version}")
print(f"Working directory: {os.getcwd()}")
print(f"Container hostname: {os.uname().nodename}")
print(f"Process ID: {os.getpid()}")

# Check GPU availability
print(f"\\nGPU available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU device count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

# Check unlimited resources
print(f"\\nCPU count: {os.cpu_count()}")

# Check internet connectivity
try:
    result = subprocess.run(["ping", "-c", "1", "google.com"], 
                          capture_output=True, timeout=5)
    internet_status = "Connected" if result.returncode == 0 else "Disconnected"
except:
    internet_status = "Unknown"

print(f"Internet status: {internet_status}")

# Demonstrate unlimited execution
print("\\n🚀 UNLIMITED CAPABILITIES DEMO:")

# No timeout demo
print("Running unlimited loop (no timeout)...")
for i in range(10):
    print(f"  Iteration {i+1}: No artificial limits!")
    time.sleep(1)

# File system access
print("\\nFile system access:")
os.makedirs("/workspace/demo", exist_ok=True)
with open("/workspace/demo/test.txt", "w") as f:
    f.write("Unlimited file access from container!")

print("  ✅ Created files in container")

# Package installation demo
print("\\nPackage installation:")
try:
    subprocess.run([sys.executable, "-m", "pip", "install", "requests"], 
                  check=True, capture_output=True)
    print("  ✅ Installed Python package")
except:
    print("  ❌ Package installation failed")

print("\\n🎉 Container demo complete!")
print("🔓 All unlimited capabilities verified!")
''',
        execution_environment=ExecutionEnvironment.UNLIMITED,
        enabled_capabilities={
            ArtifactCapability.UNLIMITED_POWER,
            ArtifactCapability.INTERNET_ACCESS,
            ArtifactCapability.GPU_COMPUTE
        }
    )
    
    print(f"Created artifact: {artifact.metadata.artifact_id}")
    
    # Create container
    context = await container_runtime.create_artifact_container(artifact)
    print(f"Created container: {context.container_id[:12]}")
    
    # Execute in container
    print("\\nExecuting in container...")
    async for result in container_runtime.execute_in_container(
        artifact.metadata.artifact_id,
        "python3 main.py",
        stream_output=True
    ):
        if result["type"] == "output":
            print(result["data"], end="")
        elif result["type"] == "execution_completed":
            print(f"\\nExecution completed with exit code: {result['exit_code']}")
            break
    
    # Get metrics
    metrics = await container_runtime.get_container_metrics(artifact.metadata.artifact_id)
    if metrics:
        print(f"\\nContainer metrics:")
        print(f"  CPU: {metrics.cpu_percent:.1f}%")
        print(f"  Memory: {metrics.memory_usage_mb:.1f} MB ({metrics.memory_percent:.1f}%)")
        print(f"  Uptime: {metrics.uptime_seconds} seconds")
    
    # List files created
    files = await container_runtime.get_container_files(artifact.metadata.artifact_id)
    print(f"\\nFiles in container: {len(files)}")
    
    # Cleanup
    await container_runtime.stop_container(artifact.metadata.artifact_id)
    await container_runtime.shutdown()
    await artifact_manager.shutdown()
    
    print("\\n🎉 Container integration demo complete!")


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run demo
    asyncio.run(demo_container_integration())