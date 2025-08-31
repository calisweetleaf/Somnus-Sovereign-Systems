"""
SOMNUS SOVEREIGN SYSTEMS - Complete Artifact Base Layer
Production Implementation with All File Types and Unlimited Execution

Integrates all artifact types from artifact_config.py with unlimited execution capabilities.
No restrictions, no timeouts, complete sovereignty.
"""

import asyncio
import hashlib
import html
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from contextlib import asynccontextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Set, Union, Tuple
from uuid import UUID, uuid4
import signal
import threading

from pydantic import BaseModel, Field

try:
    import docker
    from docker.errors import DockerException, BuildError, ImageNotFound, ContainerError, NotFound
    DOCKER_AVAILABLE = True
except ImportError:
    DOCKER_AVAILABLE = False

try:
    from starlette.websockets import WebSocket, WebSocketState
    WEBSOCKET_AVAILABLE = True
    WebSocketType = WebSocket
except ImportError:
    WEBSOCKET_AVAILABLE = False
    WebSocketType = Any
    class WebSocketState:
        CONNECTED = "connected"

try:
    import psutil
    SYSTEM_MONITORING = True
except ImportError:
    SYSTEM_MONITORING = False

logger = logging.getLogger(__name__)


# ============================================================================
# COMPREHENSIVE ARTIFACT TYPES FROM ARTIFACT_CONFIG.PY
# ============================================================================

class ArtifactType(str, Enum):
    """ALL POSSIBLE ARTIFACT TYPES - NO RESTRICTIONS"""
    # Text and markup
    TEXT = "text/plain"
    MARKDOWN = "text/markdown"
    HTML = "text/html"
    CSS = "text/css"
    JSON = "application/json"
    YAML = "text/yaml"
    XML = "text/xml"
    
    # Programming languages - COMPREHENSIVE SUPPORT
    PYTHON = "text/python"
    JAVASCRIPT = "application/javascript"
    TYPESCRIPT = "text/typescript"
    BASH = "text/bash"
    POWERSHELL = "text/powershell"
    GO = "text/go"
    RUST = "text/rust"
    C = "text/c"
    CPP = "text/cpp"
    JAVA = "text/java"
    CSHARP = "text/csharp"
    PHP = "text/php"
    RUBY = "text/ruby"
    SWIFT = "text/swift"
    KOTLIN = "text/kotlin"
    SQL = "text/sql"
    R = "text/r"
    MATLAB = "text/matlab"
    SCALA = "text/scala"
    PERL = "text/perl"
    LUA = "text/lua"
    HASKELL = "text/haskell"
    ELIXIR = "text/elixir"
    CLOJURE = "text/clojure"
    
    # Web technologies
    REACT = "application/vnd.react"
    VUE = "application/vnd.vue"
    ANGULAR = "application/vnd.angular"
    SVELTE = "application/vnd.svelte"
    
    # Data and AI
    JUPYTER = "application/jupyter"
    NOTEBOOK = "application/notebook"
    CSV = "text/csv"
    PARQUET = "application/parquet"
    
    # Infrastructure
    DOCKERFILE = "text/dockerfile"
    DOCKER_COMPOSE = "application/docker-compose"
    KUBERNETES = "application/kubernetes"
    TERRAFORM = "text/terraform"
    ANSIBLE = "text/ansible"
    
    # Media and visualization
    SVG = "image/svg+xml"
    MERMAID = "text/mermaid"
    GRAPHVIZ = "text/graphviz"
    
    # AI/ML specific
    MODEL_TRAINING = "application/model-training"
    FINE_TUNING = "application/fine-tuning"
    INFERENCE = "application/inference"
    DATA_PROCESSING = "application/data-processing"
    VIDEO_PROCESSING = "application/video-processing"
    AUDIO_PROCESSING = "application/audio-processing"
    IMAGE_PROCESSING = "application/image-processing"
    
    # Unlimited custom types
    CUSTOM = "application/custom"
    EXPERIMENTAL = "application/experimental"


class ExecutionEnvironment(str, Enum):
    """Execution environments - NO SECURITY RESTRICTIONS"""
    UNLIMITED = "unlimited"        # No restrictions whatsoever
    CONTAINER = "container"        # Docker container (main security layer)
    NATIVE = "native"             # Direct system execution
    GPU_ACCELERATED = "gpu"       # GPU-enabled execution
    DISTRIBUTED = "distributed"   # Multi-node execution
    QUANTUM = "quantum"           # Future quantum execution


class ArtifactStatus(str, Enum):
    """Artifact lifecycle status"""
    CREATED = "created"
    VALIDATING = "validating"
    VALIDATED = "validated"
    EXECUTING = "executing"
    COMPLETED = "completed"
    ERROR = "error"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"


class SecurityLevel(str, Enum):
    """Security levels - user choice, not artificial restriction"""
    SANDBOXED = "sandboxed"
    TRUSTED = "trusted"
    UNRESTRICTED = "unrestricted"


class CollaborationRole(str, Enum):
    """Collaboration roles"""
    OWNER = "owner"
    EDITOR = "editor"
    VIEWER = "viewer"
    COLLABORATOR = "collaborator"


# ============================================================================
# CORE DATA MODELS
# ============================================================================

@dataclass
class ExecutionResult:
    """Comprehensive execution result"""
    success: bool
    output: str = ""
    error: str = ""
    exit_code: int = 0
    execution_time: float = 0.0
    memory_usage: float = 0.0
    cpu_usage: float = 0.0
    gpu_usage: float = 0.0
    network_io: Dict[str, int] = field(default_factory=dict)
    file_operations: List[str] = field(default_factory=list)
    process_id: Optional[int] = None
    container_id: Optional[str] = None
    artifacts_created: List[str] = field(default_factory=list)
    performance_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ArtifactFile:
    """Individual file within artifact"""
    name: str
    content: str = ""
    binary_data: Optional[bytes] = None
    file_type: ArtifactType = ArtifactType.TEXT
    is_executable: bool = False
    encoding: str = "utf-8"
    size_bytes: int = 0
    checksum: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    permissions: str = "644"
    
    def __post_init__(self):
        if self.content and not self.binary_data:
            self.size_bytes = len(self.content.encode(self.encoding))
            self.checksum = hashlib.sha256(self.content.encode(self.encoding)).hexdigest()
        elif self.binary_data:
            self.size_bytes = len(self.binary_data)
            self.checksum = hashlib.sha256(self.binary_data).hexdigest()


@dataclass
class CollaborationCursor:
    """Real-time collaboration cursor"""
    user_id: str
    line: int
    column: int
    file_name: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ArtifactMetadata:
    """Complete artifact metadata"""
    name: str
    artifact_type: ArtifactType
    created_by: str
    session_id: str
    artifact_id: UUID = field(default_factory=uuid4)
    description: str = ""
    status: ArtifactStatus = ArtifactStatus.CREATED
    security_level: SecurityLevel = SecurityLevel.SANDBOXED
    execution_environment: ExecutionEnvironment = ExecutionEnvironment.UNLIMITED
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    execution_count: int = 0
    total_runtime: float = 0.0
    file_count: int = 0
    total_size_bytes: int = 0
    max_memory_used: float = 0.0
    total_cpu_time: float = 0.0
    collaborators: Dict[str, CollaborationRole] = field(default_factory=dict)
    tags: Set[str] = field(default_factory=set)
    dependencies: List[str] = field(default_factory=list)
    outputs: List[str] = field(default_factory=list)


# ============================================================================
# CORE ARTIFACT CLASS
# ============================================================================

class Artifact:
    """Complete artifact implementation with unlimited capabilities"""

    def __init__(
        self,
        name: str,
        artifact_type: ArtifactType,
        created_by: str,
        session_id: str,
        description: str = "",
        security_level: SecurityLevel = SecurityLevel.SANDBOXED,
        execution_environment: ExecutionEnvironment = ExecutionEnvironment.UNLIMITED
    ):
        self.metadata = ArtifactMetadata(
            name=name,
            artifact_type=artifact_type,
            created_by=created_by,
            session_id=session_id,
            description=description,
            security_level=security_level,
            execution_environment=execution_environment,
            collaborators={created_by: CollaborationRole.OWNER}
        )
        
        self.files: Dict[str, ArtifactFile] = {}
        self.execution_history: List[ExecutionResult] = []
        self.collaboration_cursors: Dict[str, CollaborationCursor] = {}
        self.websocket_connections: Set[WebSocketType] = set()
        
        # Execution state
        self._execution_lock = asyncio.Lock()
        self._current_process: Optional[subprocess.Popen] = None
        self._current_container: Optional[Any] = None
        self._is_executing = False
        self._resource_monitor_task: Optional[asyncio.Task] = None
        
        # Modification tracking
        self.is_modified = False
        self._auto_save_task: Optional[asyncio.Task] = None

    # ========================================================================
    # FILE MANAGEMENT
    # ========================================================================

    def add_file(self, name: str, content: str = "", binary_data: Optional[bytes] = None,
                 file_type: Optional[ArtifactType] = None, is_executable: bool = False) -> ArtifactFile:
        """Add file with comprehensive type detection"""
        if file_type is None:
            file_type = self._guess_file_type(name)

        artifact_file = ArtifactFile(
            name=name,
            content=content,
            binary_data=binary_data,
            file_type=file_type,
            is_executable=is_executable
        )

        self.files[name] = artifact_file
        self._update_metadata()
        self.is_modified = True
        return artifact_file

    def remove_file(self, name: str) -> bool:
        """Remove file from artifact"""
        if name in self.files:
            del self.files[name]
            self._update_metadata()
            self.is_modified = True
            return True
        return False

    def get_file(self, name: str) -> Optional[ArtifactFile]:
        """Get file by name"""
        return self.files.get(name)

    def get_main_file(self) -> Optional[ArtifactFile]:
        """Get main executable file"""
        main_candidates = [
            "main.py", "app.py", "__main__.py", "run.py",
            "index.html", "index.htm", "main.html",
            "app.js", "main.js", "index.js", "server.js",
            "main.go", "main.rs", "main.c", "main.cpp",
            "Main.java", "main.java", "app.java",
            "main.rb", "app.rb", "main.php", "index.php",
            "main.cs", "Program.cs", "main.swift",
            "main.kt", "main.scala", "main.clj",
            "main.hs", "main.ex", "main.lua",
            "Dockerfile", "docker-compose.yml", "docker-compose.yaml",
            "main.ipynb", "notebook.ipynb",
            "main.r", "main.R", "main.m",
            "main.sql", "script.sql"
        ]

        for candidate in main_candidates:
            if candidate in self.files:
                return self.files[candidate]

        # Find first executable file
        for file in self.files.values():
            if file.is_executable:
                return file

        # Return first file of matching type
        if self.files:
            return list(self.files.values())[0]

        return None

    def _guess_file_type(self, filename: str) -> ArtifactType:
        """Comprehensive file type detection"""
        extension_map = {
            # Programming languages
            '.py': ArtifactType.PYTHON,
            '.js': ArtifactType.JAVASCRIPT,
            '.ts': ArtifactType.TYPESCRIPT,
            '.html': ArtifactType.HTML,
            '.htm': ArtifactType.HTML,
            '.css': ArtifactType.CSS,
            '.md': ArtifactType.MARKDOWN,
            '.json': ArtifactType.JSON,
            '.yaml': ArtifactType.YAML,
            '.yml': ArtifactType.YAML,
            '.xml': ArtifactType.XML,
            '.sh': ArtifactType.BASH,
            '.bash': ArtifactType.BASH,
            '.ps1': ArtifactType.POWERSHELL,
            '.go': ArtifactType.GO,
            '.rs': ArtifactType.RUST,
            '.c': ArtifactType.C,
            '.cpp': ArtifactType.CPP,
            '.cc': ArtifactType.CPP,
            '.cxx': ArtifactType.CPP,
            '.java': ArtifactType.JAVA,
            '.cs': ArtifactType.CSHARP,
            '.php': ArtifactType.PHP,
            '.rb': ArtifactType.RUBY,
            '.swift': ArtifactType.SWIFT,
            '.kt': ArtifactType.KOTLIN,
            '.sql': ArtifactType.SQL,
            '.r': ArtifactType.R,
            '.R': ArtifactType.R,
            '.m': ArtifactType.MATLAB,
            '.scala': ArtifactType.SCALA,
            '.pl': ArtifactType.PERL,
            '.lua': ArtifactType.LUA,
            '.hs': ArtifactType.HASKELL,
            '.ex': ArtifactType.ELIXIR,
            '.clj': ArtifactType.CLOJURE,
            
            # Web frameworks
            '.jsx': ArtifactType.REACT,
            '.tsx': ArtifactType.REACT,
            '.vue': ArtifactType.VUE,
            
            # Infrastructure
            '.tf': ArtifactType.TERRAFORM,
            '.yml': ArtifactType.ANSIBLE,  # Context dependent
            
            # Data formats
            '.csv': ArtifactType.CSV,
            '.parquet': ArtifactType.PARQUET,
            
            # Notebooks
            '.ipynb': ArtifactType.JUPYTER,
            
            # Graphics
            '.svg': ArtifactType.SVG,
            '.mmd': ArtifactType.MERMAID,
            '.dot': ArtifactType.GRAPHVIZ,
        }
        
        # Special filename handling
        special_files = {
            'dockerfile': ArtifactType.DOCKERFILE,
            'docker-compose.yml': ArtifactType.DOCKER_COMPOSE,
            'docker-compose.yaml': ArtifactType.DOCKER_COMPOSE,
            'makefile': ArtifactType.BASH,
            'rakefile': ArtifactType.RUBY,
            'gemfile': ArtifactType.RUBY,
            'package.json': ArtifactType.JSON,
            'requirements.txt': ArtifactType.TEXT,
            'cargo.toml': ArtifactType.RUST,
        }
        
        filename_lower = filename.lower()
        if filename_lower in special_files:
            return special_files[filename_lower]
        
        ext = Path(filename).suffix.lower()
        return extension_map.get(ext, ArtifactType.TEXT)

    def _update_metadata(self):
        """Update artifact metadata"""
        self.metadata.updated_at = datetime.now(timezone.utc)
        self.metadata.file_count = len(self.files)
        self.metadata.total_size_bytes = sum(f.size_bytes for f in self.files.values())

    # ========================================================================
    # UNLIMITED EXECUTION ENGINE
    # ========================================================================

    async def execute(self, timeout: Optional[int] = None, 
                     environment_vars: Optional[Dict[str, str]] = None,
                     gpu_enabled: bool = False,
                     memory_limit: Optional[int] = None,
                     cpu_limit: Optional[float] = None) -> ExecutionResult:
        """Execute artifact with unlimited capabilities"""
        async with self._execution_lock:
            if self._is_executing:
                return ExecutionResult(success=False, error="Artifact is already executing")

            self._is_executing = True
            self.metadata.status = ArtifactStatus.EXECUTING
            start_time = time.time()

            try:
                await self._broadcast_status_update("execution_started")
                
                # Start resource monitoring
                if SYSTEM_MONITORING:
                    self._resource_monitor_task = asyncio.create_task(self._monitor_resources())

                # Execute based on environment
                if self.metadata.execution_environment == ExecutionEnvironment.UNLIMITED:
                    result = await self._execute_unlimited(timeout, environment_vars, gpu_enabled)
                elif self.metadata.execution_environment == ExecutionEnvironment.CONTAINER:
                    result = await self._execute_container(timeout, environment_vars, gpu_enabled, memory_limit, cpu_limit)
                elif self.metadata.execution_environment == ExecutionEnvironment.NATIVE:
                    result = await self._execute_native(timeout, environment_vars)
                elif self.metadata.execution_environment == ExecutionEnvironment.GPU_ACCELERATED:
                    result = await self._execute_gpu(timeout, environment_vars)
                elif self.metadata.execution_environment == ExecutionEnvironment.DISTRIBUTED:
                    result = await self._execute_distributed(timeout, environment_vars)
                else:
                    result = await self._execute_unlimited(timeout, environment_vars, gpu_enabled)

                # Update statistics
                result.execution_time = time.time() - start_time
                self.metadata.execution_count += 1
                self.metadata.total_runtime += result.execution_time
                self.metadata.max_memory_used = max(self.metadata.max_memory_used, result.memory_usage)
                self.metadata.total_cpu_time += result.cpu_usage

                self.execution_history.append(result)
                self.metadata.status = ArtifactStatus.COMPLETED if result.success else ArtifactStatus.ERROR
                self._update_metadata()
                self.is_modified = True

                await self._broadcast_execution_result(result)
                return result

            except Exception as e:
                error_result = ExecutionResult(
                    success=False,
                    error=f"Execution failed: {str(e)}",
                    execution_time=time.time() - start_time
                )
                self.metadata.status = ArtifactStatus.ERROR
                await self._broadcast_execution_result(error_result)
                return error_result

            finally:
                self._is_executing = False
                if self._resource_monitor_task:
                    self._resource_monitor_task.cancel()

    async def _execute_unlimited(self, timeout: Optional[int], environment_vars: Optional[Dict[str, str]], 
                                gpu_enabled: bool) -> ExecutionResult:
        """Execute with no restrictions"""
        main_file = self.get_main_file()
        if not main_file:
            return ExecutionResult(success=False, error="No main file found")

        # Create temporary workspace
        with tempfile.TemporaryDirectory(prefix="somnus_artifact_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write all files
            for filename, artifact_file in self.files.items():
                file_path = temp_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if artifact_file.binary_data:
                    file_path.write_bytes(artifact_file.binary_data)
                else:
                    file_path.write_text(artifact_file.content, encoding=artifact_file.encoding)
                
                if artifact_file.is_executable:
                    os.chmod(file_path, 0o755)

            return await self._execute_file_type(main_file, temp_path, timeout, environment_vars, gpu_enabled)

    async def _execute_container(self, timeout: Optional[int], environment_vars: Optional[Dict[str, str]],
                                gpu_enabled: bool, memory_limit: Optional[int], cpu_limit: Optional[float]) -> ExecutionResult:
        """Execute in Docker container with full isolation"""
        if not DOCKER_AVAILABLE:
            return ExecutionResult(success=False, error="Docker not available")

        main_file = self.get_main_file()
        if not main_file:
            return ExecutionResult(success=False, error="No main file found")

        try:
            client = docker.from_env()
            
            # Create temporary workspace
            with tempfile.TemporaryDirectory(prefix="somnus_container_") as temp_dir:
                temp_path = Path(temp_dir)
                
                # Write all files
                for filename, artifact_file in self.files.items():
                    file_path = temp_path / filename
                    file_path.parent.mkdir(parents=True, exist_ok=True)
                    
                    if artifact_file.binary_data:
                        file_path.write_bytes(artifact_file.binary_data)
                    else:
                        file_path.write_text(artifact_file.content, encoding=artifact_file.encoding)
                    
                    if artifact_file.is_executable:
                        os.chmod(file_path, 0o755)

                # Determine container image and command
                image, command = self._get_container_config(main_file)
                
                # Container configuration
                container_config = {
                    'image': image,
                    'command': command,
                    'volumes': {str(temp_path): {'bind': '/workspace', 'mode': 'rw'}},
                    'working_dir': '/workspace',
                    'detach': True,
                    'auto_remove': True,
                    'environment': environment_vars or {},
                    'network_mode': 'bridge'
                }
                
                # Resource limits
                if memory_limit:
                    container_config['mem_limit'] = f"{memory_limit}m"
                if cpu_limit:
                    container_config['cpu_quota'] = int(cpu_limit * 100000)
                    container_config['cpu_period'] = 100000
                
                # GPU support
                if gpu_enabled:
                    container_config['device_requests'] = [
                        docker.types.DeviceRequest(count=-1, capabilities=[['gpu']])
                    ]

                # Run container
                container = client.containers.run(**container_config)
                self._current_container = container

                # Wait for completion
                result = container.wait(timeout=timeout)
                stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
                stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
                
                return ExecutionResult(
                    success=result['StatusCode'] == 0,
                    output=stdout,
                    error=stderr,
                    exit_code=result['StatusCode'],
                    container_id=container.id
                )

        except Exception as e:
            return ExecutionResult(success=False, error=f"Container execution failed: {str(e)}")

    async def _execute_native(self, timeout: Optional[int], environment_vars: Optional[Dict[str, str]]) -> ExecutionResult:
        """Execute directly on host system"""
        main_file = self.get_main_file()
        if not main_file:
            return ExecutionResult(success=False, error="No main file found")

        # Security check for native execution
        if self.metadata.security_level == SecurityLevel.SANDBOXED:
            return ExecutionResult(success=False, error="Native execution requires trusted or unrestricted security level")

        with tempfile.TemporaryDirectory(prefix="somnus_native_") as temp_dir:
            temp_path = Path(temp_dir)
            
            # Write all files
            for filename, artifact_file in self.files.items():
                file_path = temp_path / filename
                file_path.parent.mkdir(parents=True, exist_ok=True)
                
                if artifact_file.binary_data:
                    file_path.write_bytes(artifact_file.binary_data)
                else:
                    file_path.write_text(artifact_file.content, encoding=artifact_file.encoding)
                
                if artifact_file.is_executable:
                    os.chmod(file_path, 0o755)

            return await self._execute_file_type(main_file, temp_path, timeout, environment_vars, False)

    async def _execute_gpu(self, timeout: Optional[int], environment_vars: Optional[Dict[str, str]]) -> ExecutionResult:
        """Execute with GPU acceleration"""
        # Set GPU environment variables
        gpu_env = {
            'CUDA_VISIBLE_DEVICES': '0,1,2,3,4,5,6,7',
            'NVIDIA_VISIBLE_DEVICES': 'all',
            'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:512',
        }
        if environment_vars:
            gpu_env.update(environment_vars)
        
        return await self._execute_unlimited(timeout, gpu_env, True)

    async def _execute_distributed(self, timeout: Optional[int], environment_vars: Optional[Dict[str, str]]) -> ExecutionResult:
        """Execute across multiple nodes (future implementation)"""
        # For now, fall back to unlimited execution
        return await self._execute_unlimited(timeout, environment_vars, False)

    async def _execute_file_type(self, main_file: ArtifactFile, work_dir: Path, 
                                timeout: Optional[int], environment_vars: Optional[Dict[str, str]],
                                gpu_enabled: bool) -> ExecutionResult:
        """Execute based on file type"""
        file_path = work_dir / main_file.name
        env = os.environ.copy()
        if environment_vars:
            env.update(environment_vars)
        
        # Add work directory to PATH and PYTHONPATH
        env['PATH'] = f"{work_dir}:{env.get('PATH', '')}"
        env['PYTHONPATH'] = f"{work_dir}:{env.get('PYTHONPATH', '')}"

        # Determine execution command
        if main_file.file_type == ArtifactType.PYTHON:
            cmd = [sys.executable, str(file_path)]
        elif main_file.file_type == ArtifactType.JAVASCRIPT:
            cmd = ['node', str(file_path)]
        elif main_file.file_type == ArtifactType.TYPESCRIPT:
            cmd = ['npx', 'ts-node', str(file_path)]
        elif main_file.file_type == ArtifactType.BASH:
            cmd = ['bash', str(file_path)]
        elif main_file.file_type == ArtifactType.POWERSHELL:
            cmd = ['pwsh', '-File', str(file_path)]
        elif main_file.file_type == ArtifactType.GO:
            cmd = ['go', 'run', str(file_path)]
        elif main_file.file_type == ArtifactType.RUST:
            return await self._execute_rust_project(work_dir, timeout, env)
        elif main_file.file_type == ArtifactType.JAVA:
            return await self._execute_java_file(file_path, work_dir, timeout, env)
        elif main_file.file_type == ArtifactType.C:
            return await self._execute_c_file(file_path, work_dir, timeout, env)
        elif main_file.file_type == ArtifactType.CPP:
            return await self._execute_cpp_file(file_path, work_dir, timeout, env)
        elif main_file.file_type == ArtifactType.CSHARP:
            cmd = ['dotnet', 'run', '--project', str(work_dir)]
        elif main_file.file_type == ArtifactType.RUBY:
            cmd = ['ruby', str(file_path)]
        elif main_file.file_type == ArtifactType.PHP:
            cmd = ['php', str(file_path)]
        elif main_file.file_type == ArtifactType.R:
            cmd = ['Rscript', str(file_path)]
        elif main_file.file_type == ArtifactType.JUPYTER:
            return await self._execute_jupyter_notebook(file_path, timeout, env)
        elif main_file.file_type == ArtifactType.DOCKERFILE:
            return await self._execute_dockerfile(work_dir, timeout, env)
        elif main_file.file_type == ArtifactType.DOCKER_COMPOSE:
            return await self._execute_docker_compose(work_dir, timeout, env)
        elif main_file.file_type in [ArtifactType.HTML, ArtifactType.CSS, ArtifactType.SVG, ArtifactType.JSON]:
            # Return content for client-side rendering
            return ExecutionResult(success=True, output=main_file.content)
        elif main_file.file_type in [ArtifactType.MODEL_TRAINING, ArtifactType.FINE_TUNING]:
            return await self._execute_ml_training(file_path, work_dir, timeout, env, gpu_enabled)
        else:
            # Generic execution attempt
            if main_file.is_executable:
                cmd = [str(file_path)]
            else:
                return ExecutionResult(success=False, error=f"Unsupported file type: {main_file.file_type}")

        return await self._run_subprocess(cmd, work_dir, timeout, env)

    async def _run_subprocess(self, cmd: List[str], work_dir: Path, timeout: Optional[int], 
                             env: Dict[str, str]) -> ExecutionResult:
        """Run subprocess with comprehensive monitoring"""
        try:
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=work_dir,
                env=env
            )
            
            self._current_process = process
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
                
                return ExecutionResult(
                    success=process.returncode == 0,
                    output=stdout.decode('utf-8') if stdout else "",
                    error=stderr.decode('utf-8') if stderr else "",
                    exit_code=process.returncode or 0,
                    process_id=process.pid
                )
                
            except asyncio.TimeoutError:
                process.terminate()
                await asyncio.sleep(1)
                if process.returncode is None:
                    process.kill()
                
                return ExecutionResult(
                    success=False,
                    error=f"Execution timed out after {timeout} seconds",
                    exit_code=-1
                )
                
        except Exception as e:
            return ExecutionResult(success=False, error=f"Subprocess execution failed: {str(e)}")

    # ========================================================================
    # SPECIALIZED EXECUTION METHODS
    # ========================================================================

    async def _execute_rust_project(self, work_dir: Path, timeout: Optional[int], env: Dict[str, str]) -> ExecutionResult:
        """Execute Rust project with Cargo"""
        cargo_toml = work_dir / "Cargo.toml"
        if not cargo_toml.exists():
            # Create minimal Cargo.toml
            cargo_content = """[package]
name = "somnus-artifact"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "main"
path = "main.rs"
"""
            cargo_toml.write_text(cargo_content)
        
        return await self._run_subprocess(['cargo', 'run'], work_dir, timeout, env)

    async def _execute_java_file(self, file_path: Path, work_dir: Path, timeout: Optional[int], 
                                env: Dict[str, str]) -> ExecutionResult:
        """Compile and execute Java file"""
        # Compile
        compile_result = await self._run_subprocess(['javac', str(file_path)], work_dir, timeout, env)
        if not compile_result.success:
            return compile_result
        
        # Run
        class_name = file_path.stem
        return await self._run_subprocess(['java', class_name], work_dir, timeout, env)

    async def _execute_c_file(self, file_path: Path, work_dir: Path, timeout: Optional[int], 
                             env: Dict[str, str]) -> ExecutionResult:
        """Compile and execute C file"""
        binary_path = work_dir / "main"
        compile_result = await self._run_subprocess(['gcc', str(file_path), '-o', str(binary_path)], work_dir, timeout, env)
        if not compile_result.success:
            return compile_result
        
        os.chmod(binary_path, 0o755)
        return await self._run_subprocess([str(binary_path)], work_dir, timeout, env)

    async def _execute_cpp_file(self, file_path: Path, work_dir: Path, timeout: Optional[int], 
                               env: Dict[str, str]) -> ExecutionResult:
        """Compile and execute C++ file"""
        binary_path = work_dir / "main"
        compile_result = await self._run_subprocess(['g++', str(file_path), '-o', str(binary_path)], work_dir, timeout, env)
        if not compile_result.success:
            return compile_result
        
        os.chmod(binary_path, 0o755)
        return await self._run_subprocess([str(binary_path)], work_dir, timeout, env)

    async def _execute_jupyter_notebook(self, file_path: Path, timeout: Optional[int], 
                                       env: Dict[str, str]) -> ExecutionResult:
        """Execute Jupyter notebook"""
        output_path = file_path.with_suffix('.executed.ipynb')
        cmd = ['jupyter', 'nbconvert', '--to', 'notebook', '--execute', 
               '--output', str(output_path), str(file_path)]
        
        result = await self._run_subprocess(cmd, file_path.parent, timeout, env)
        if result.success and output_path.exists():
            result.artifacts_created.append(str(output_path))
        
        return result

    async def _execute_dockerfile(self, work_dir: Path, timeout: Optional[int], 
                                 env: Dict[str, str]) -> ExecutionResult:
        """Build and run Docker image"""
        if not DOCKER_AVAILABLE:
            return ExecutionResult(success=False, error="Docker not available")
        
        try:
            client = docker.from_env()
            image_tag = f"somnus-artifact-{uuid4().hex[:8]}"
            
            # Build image
            image, build_logs = client.images.build(path=str(work_dir), tag=image_tag, rm=True)
            
            # Run container
            container = client.containers.run(
                image_tag,
                detach=True,
                auto_remove=True
            )
            
            result = container.wait(timeout=timeout)
            stdout = container.logs(stdout=True, stderr=False).decode('utf-8')
            stderr = container.logs(stdout=False, stderr=True).decode('utf-8')
            
            return ExecutionResult(
                success=result['StatusCode'] == 0,
                output=stdout,
                error=stderr,
                exit_code=result['StatusCode'],
                container_id=container.id
            )
            
        except Exception as e:
            return ExecutionResult(success=False, error=f"Docker execution failed: {str(e)}")

    async def _execute_docker_compose(self, work_dir: Path, timeout: Optional[int], 
                                     env: Dict[str, str]) -> ExecutionResult:
        """Execute Docker Compose"""
        return await self._run_subprocess(['docker-compose', 'up', '--build'], work_dir, timeout, env)

    async def _execute_ml_training(self, file_path: Path, work_dir: Path, timeout: Optional[int], 
                                  env: Dict[str, str], gpu_enabled: bool) -> ExecutionResult:
        """Execute ML training with GPU support"""
        if gpu_enabled:
            env['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
            env['NVIDIA_VISIBLE_DEVICES'] = 'all'
        
        return await self._run_subprocess([sys.executable, str(file_path)], work_dir, timeout, env)

    def _get_container_config(self, main_file: ArtifactFile) -> Tuple[str, List[str]]:
        """Get container image and command for file type"""
        config_map = {
            ArtifactType.PYTHON: ("python:3.11-slim", ["python", f"/workspace/{main_file.name}"]),
            ArtifactType.JAVASCRIPT: ("node:18-slim", ["node", f"/workspace/{main_file.name}"]),
            ArtifactType.TYPESCRIPT: ("node:18-slim", ["npx", "ts-node", f"/workspace/{main_file.name}"]),
            ArtifactType.GO: ("golang:1.20-alpine", ["go", "run", f"/workspace/{main_file.name}"]),
            ArtifactType.RUST: ("rust:1.70-slim", ["cargo", "run"]),
            ArtifactType.JAVA: ("openjdk:17-slim", ["sh", "-c", f"javac /workspace/{main_file.name} && java -cp /workspace {main_file.name.replace('.java', '')}"]),
            ArtifactType.RUBY: ("ruby:3.1-slim", ["ruby", f"/workspace/{main_file.name}"]),
            ArtifactType.PHP: ("php:8.1-cli", ["php", f"/workspace/{main_file.name}"]),
            ArtifactType.R: ("r-base:4.2.0", ["Rscript", f"/workspace/{main_file.name}"]),
            ArtifactType.BASH: ("ubuntu:22.04", ["bash", f"/workspace/{main_file.name}"]),
        }
        
        return config_map.get(main_file.file_type, ("ubuntu:22.04", [f"/workspace/{main_file.name}"]))

    # ========================================================================
    # RESOURCE MONITORING
    # ========================================================================

    async def _monitor_resources(self):
        """Monitor resource usage during execution"""
        if not SYSTEM_MONITORING:
            return
        
        try:
            while self._is_executing:
                if self._current_process:
                    try:
                        process = psutil.Process(self._current_process.pid)
                        memory_mb = process.memory_info().rss / 1024 / 1024
                        cpu_percent = process.cpu_percent()
                        
                        # Update metrics in latest execution result
                        if self.execution_history:
                            latest = self.execution_history[-1]
                            latest.memory_usage = max(latest.memory_usage, memory_mb)
                            latest.cpu_usage = max(latest.cpu_usage, cpu_percent)
                            
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        pass
                
                await asyncio.sleep(1)
                
        except asyncio.CancelledError:
            pass

    # ========================================================================
    # COLLABORATION AND WEBSOCKETS
    # ========================================================================

    async def add_websocket(self, websocket: WebSocketType):
        """Add WebSocket connection for real-time updates"""
        if WEBSOCKET_AVAILABLE:
            self.websocket_connections.add(websocket)

    async def remove_websocket(self, websocket: WebSocketType):
        """Remove WebSocket connection"""
        if WEBSOCKET_AVAILABLE:
            self.websocket_connections.discard(websocket)

    async def _broadcast_status_update(self, event_type: str, data: Optional[Dict] = None):
        """Broadcast status update to all connected WebSockets"""
        if not WEBSOCKET_AVAILABLE or not self.websocket_connections:
            return

        message = {
            "type": "status_update",
            "event": event_type,
            "artifact_id": str(self.metadata.artifact_id),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "data": data or {}
        }

        disconnected = set()
        for websocket in self.websocket_connections:
            try:
                if websocket.client_state == WebSocketState.CONNECTED:
                    await websocket.send_json(message)
                else:
                    disconnected.add(websocket)
            except Exception:
                disconnected.add(websocket)

        # Clean up disconnected WebSockets
        for websocket in disconnected:
            self.websocket_connections.discard(websocket)

    async def _broadcast_execution_result(self, result: ExecutionResult):
        """Broadcast execution result to all connected WebSockets"""
        await self._broadcast_status_update("execution_completed", {
            "success": result.success,
            "output": result.output[:1000] if result.output else "",  # Truncate for WebSocket
            "error": result.error[:1000] if result.error else "",
            "execution_time": result.execution_time,
            "memory_usage": result.memory_usage,
            "cpu_usage": result.cpu_usage
        })

    async def stop_execution(self) -> bool:
        """Stop current execution"""
        if not self._is_executing:
            return False

        try:
            if self._current_process:
                self._current_process.terminate()
                await asyncio.sleep(2)
                if self._current_process.poll() is None:
                    self._current_process.kill()

            if self._current_container and DOCKER_AVAILABLE:
                self._current_container.stop()

            self._is_executing = False
            self.metadata.status = ArtifactStatus.SUSPENDED
            await self._broadcast_status_update("execution_stopped")
            return True

        except Exception as e:
            logger.error(f"Failed to stop execution: {e}")
            return False

    # ========================================================================
    # EXPORT AND SERIALIZATION
    # ========================================================================

    def to_dict(self) -> Dict[str, Any]:
        """Convert artifact to dictionary"""
        return {
            "metadata": {
                "name": self.metadata.name,
                "artifact_type": self.metadata.artifact_type.value,
                "created_by": self.metadata.created_by,
                "session_id": self.metadata.session_id,
                "artifact_id": str(self.metadata.artifact_id),
                "description": self.metadata.description,
                "status": self.metadata.status.value,
                "security_level": self.metadata.security_level.value,
                "execution_environment": self.metadata.execution_environment.value,
                "created_at": self.metadata.created_at.isoformat(),
                "updated_at": self.metadata.updated_at.isoformat(),
                "execution_count": self.metadata.execution_count,
                "total_runtime": self.metadata.total_runtime,
                "file_count": self.metadata.file_count,
                "total_size_bytes": self.metadata.total_size_bytes,
                "max_memory_used": self.metadata.max_memory_used,
                "total_cpu_time": self.metadata.total_cpu_time,
                "collaborators": {k: v.value for k, v in self.metadata.collaborators.items()},
                "tags": list(self.metadata.tags),
                "dependencies": self.metadata.dependencies,
                "outputs": self.metadata.outputs
            },
            "files": {
                name: {
                    "name": file.name,
                    "content": file.content,
                    "file_type": file.file_type.value,
                    "is_executable": file.is_executable,
                    "size_bytes": file.size_bytes,
                    "checksum": file.checksum,
                    "created_at": file.created_at.isoformat(),
                    "last_modified": file.last_modified.isoformat(),
                    "permissions": file.permissions
                }
                for name, file in self.files.items()
            },
            "execution_history": [
                {
                    "success": result.success,
                    "output": result.output,
                    "error": result.error,
                    "exit_code": result.exit_code,
                    "execution_time": result.execution_time,
                    "memory_usage": result.memory_usage,
                    "cpu_usage": result.cpu_usage,
                    "gpu_usage": result.gpu_usage,
                    "process_id": result.process_id,
                    "container_id": result.container_id,
                    "artifacts_created": result.artifacts_created,
                    "performance_metrics": result.performance_metrics
                }
                for result in self.execution_history
            ]
        }

    async def export_zip(self) -> bytes:
        """Export artifact as ZIP file"""
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            # Add metadata
            zip_file.writestr("metadata.json", json.dumps(self.to_dict()["metadata"], indent=2))
            
            # Add all files
            for filename, artifact_file in self.files.items():
                if artifact_file.binary_data:
                    zip_file.writestr(filename, artifact_file.binary_data)
                else:
                    zip_file.writestr(filename, artifact_file.content.encode(artifact_file.encoding))
        
        zip_buffer.seek(0)
        return zip_buffer.read()


# ============================================================================
# ARTIFACT MANAGER
# ============================================================================

class ArtifactManager:
    """Comprehensive artifact management system"""
    
    def __init__(self, storage_path: Path, enable_persistence: bool = True):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.enable_persistence = enable_persistence
        
        self.artifacts: Dict[UUID, Artifact] = {}
        self.sessions: Dict[str, Set[UUID]] = defaultdict(set)
        
        # Auto-save configuration
        self._auto_save_interval = 30  # seconds
        self._auto_save_task: Optional[asyncio.Task] = None

    async def create_artifact(
        self,
        name: str,
        artifact_type: ArtifactType,
        created_by: str,
        session_id: str,
        description: str = "",
        security_level: SecurityLevel = SecurityLevel.SANDBOXED,
        execution_environment: ExecutionEnvironment = ExecutionEnvironment.UNLIMITED
    ) -> Artifact:
        """Create new artifact with comprehensive configuration"""
        
        artifact = Artifact(
            name=name,
            artifact_type=artifact_type,
            created_by=created_by,
            session_id=session_id,
            description=description,
            security_level=security_level,
            execution_environment=execution_environment
        )
        
        self.artifacts[artifact.metadata.artifact_id] = artifact
        self.sessions[session_id].add(artifact.metadata.artifact_id)
        
        if self.enable_persistence:
            await self._save_artifact(artifact)
        
        logger.info(f"Created artifact {artifact.metadata.name} ({artifact.metadata.artifact_id})")
        return artifact

    async def get_artifact(self, artifact_id: UUID) -> Optional[Artifact]:
        """Get artifact by ID"""
        return self.artifacts.get(artifact_id)

    async def delete_artifact(self, artifact_id: UUID, user_id: str) -> bool:
        """Delete artifact with permission check"""
        artifact = self.artifacts.get(artifact_id)
        if not artifact:
            return False
        
        # Permission check
        if (artifact.metadata.created_by != user_id and 
            artifact.metadata.collaborators.get(user_id) != CollaborationRole.OWNER):
            return False
        
        # Stop execution if running
        if artifact._is_executing:
            await artifact.stop_execution()
        
        # Remove from session tracking
        if artifact.metadata.session_id in self.sessions:
            self.sessions[artifact.metadata.session_id].discard(artifact_id)
        
        # Delete from storage
        if self.enable_persistence:
            storage_file = self.storage_path / f"{artifact_id}.json"
            if storage_file.exists():
                storage_file.unlink()
        
        # Remove from memory
        del self.artifacts[artifact_id]
        
        logger.info(f"Deleted artifact {artifact_id}")
        return True

    async def _save_artifact(self, artifact: Artifact):
        """Save artifact to persistent storage"""
        if not self.enable_persistence:
            return
        
        storage_file = self.storage_path / f"{artifact.metadata.artifact_id}.json"
        artifact_data = artifact.to_dict()
        
        async with aiofiles.open(storage_file, 'w') as f:
            await f.write(json.dumps(artifact_data, indent=2, default=str))

    async def start_auto_save(self):
        """Start automatic saving of modified artifacts"""
        if self._auto_save_task:
            return
        
        self._auto_save_task = asyncio.create_task(self._auto_save_loop())
        logger.info("Started auto-save task")

    async def stop_auto_save(self):
        """Stop automatic saving"""
        if self._auto_save_task:
            self._auto_save_task.cancel()
            self._auto_save_task = None
        logger.info("Stopped auto-save task")

    async def _auto_save_loop(self):
        """Auto-save loop for modified artifacts"""
        try:
            while True:
                await asyncio.sleep(self._auto_save_interval)
                
                for artifact in self.artifacts.values():
                    if artifact.is_modified:
                        await self._save_artifact(artifact)
                        artifact.is_modified = False
                        
        except asyncio.CancelledError:
            pass

    async def get_session_artifacts(self, session_id: str) -> List[Artifact]:
        """Get all artifacts for a session"""
        artifact_ids = self.sessions.get(session_id, set())
        return [self.artifacts[aid] for aid in artifact_ids if aid in self.artifacts]

    async def shutdown(self):
        """Graceful shutdown"""
        logger.info("Shutting down artifact manager...")
        
        # Stop auto-save
        await self.stop_auto_save()
        
        # Stop all executing artifacts
        for artifact in self.artifacts.values():
            if artifact._is_executing:
                await artifact.stop_execution()
        
        # Save all modified artifacts
        if self.enable_persistence:
            for artifact in self.artifacts.values():
                if artifact.is_modified:
                    await self._save_artifact(artifact)
        
        logger.info("Artifact manager shutdown complete")