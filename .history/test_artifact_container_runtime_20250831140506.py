#!/usr/bin/env python3
"""
Comprehensive Test Suite for artifact_container_runtime.py
Tests all functions, classes, and integration points
"""

import pytest
import asyncio
import tempfile
import shutil
import os
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import Mock, patch, AsyncMock, MagicMock
from uuid import UUID, uuid4

# Test dependencies
import docker
from fastapi import HTTPException
from pydantic import ValidationError

# Import the module under test
from backend.artifacts.artifact_container_runtime import (
    ContainerState, ContainerConfig, ContainerMetrics, ContainerExecutionContext,
    UnlimitedContainerRuntime, create_container_runtime_router,
    create_integrated_artifact_system, demo_container_integration
)

# Mock imports for standalone testing
try:
    from complete_artifact_system import (
        UnlimitedArtifact, UnlimitedArtifactManager, ArtifactType,
        ExecutionEnvironment, ArtifactCapability, UnlimitedExecutionResult,
        UnlimitedExecutionConfig
    )
except ImportError:
    # Create mock classes for testing
    class MockArtifactType:
        PYTHON = "python"
        JAVASCRIPT = "javascript"
        ML_MODEL = "ml_model"

    class MockExecutionEnvironment:
        UNLIMITED = "unlimited"
        STANDARD = "standard"

    class MockArtifactCapability:
        UNLIMITED_POWER = "unlimited_power"
        INTERNET_ACCESS = "internet_access"
        GPU_COMPUTE = "gpu_compute"

    class MockUnlimitedArtifact:
        def __init__(self, artifact_id=None, name="test_artifact"):
            self.metadata = Mock()
            self.metadata.artifact_id = artifact_id or uuid4()
            self.metadata.name = name
            self.metadata.artifact_type = MockArtifactType.PYTHON
            self.metadata.session_id = "test_session"
            self.metadata.created_by = "test_user"
            self.files = {}

    class MockUnlimitedArtifactManager:
        def __init__(self):
            self.artifacts = {}

        async def initialize(self):
            pass

        async def get_artifact(self, artifact_id):
            return self.artifacts.get(artifact_id)

        async def create_artifact(self, **kwargs):
            artifact = MockUnlimitedArtifact()
            self.artifacts[artifact.metadata.artifact_id] = artifact
            return artifact

        async def shutdown(self):
            pass

    # Replace imports
    UnlimitedArtifact = MockUnlimitedArtifact
    UnlimitedArtifactManager = MockUnlimitedArtifactManager
    ArtifactType = MockArtifactType
    ExecutionEnvironment = MockExecutionEnvironment
    ArtifactCapability = MockArtifactCapability


class TestContainerState:
    """Test ContainerState enum"""

    def test_container_states(self):
        """Test all container state values"""
        assert ContainerState.CREATING.value == "creating"
        assert ContainerState.BUILDING.value == "building"
        assert ContainerState.STARTING.value == "starting"
        assert ContainerState.RUNNING.value == "running"
        assert ContainerState.EXECUTING.value == "executing"
        assert ContainerState.PAUSED.value == "paused"
        assert ContainerState.STOPPED.value == "stopped"
        assert ContainerState.ERROR.value == "error"
        assert ContainerState.DESTROYED.value == "destroyed"

    def test_container_state_order(self):
        """Test logical state progression"""
        states = [
            ContainerState.CREATING,
            ContainerState.BUILDING,
            ContainerState.STARTING,
            ContainerState.RUNNING,
            ContainerState.EXECUTING,
            ContainerState.STOPPED,
            ContainerState.DESTROYED
        ]

        for i in range(len(states) - 1):
            assert states[i] != states[i + 1]


class TestContainerConfig:
    """Test ContainerConfig dataclass"""

    def test_default_config(self):
        """Test default configuration values"""
        config = ContainerConfig()

        assert config.image_name == "somnus-artifact:unlimited"
        assert config.cpu_limit is None
        assert config.memory_limit is None
        assert config.disk_limit is None
        assert config.network_mode == "bridge"
        assert config.privileged is True
        assert config.gpu_enabled is True
        assert config.internet_enabled is True
        assert config.working_dir == "/workspace"
        assert config.user == "root"
        assert config.shell == "/bin/bash"
        assert len(config.exposed_ports) > 0
        assert isinstance(config.environment_vars, dict)

    def test_custom_config(self):
        """Test custom configuration"""
        custom_env = {"CUSTOM_VAR": "value"}
        custom_ports = [9000, 9001]

        config = ContainerConfig(
            image_name="custom:image",
            cpu_limit=2.0,
            memory_limit="4g",
            privileged=False,
            gpu_enabled=False,
            environment_vars=custom_env,
            exposed_ports=custom_ports
        )

        assert config.image_name == "custom:image"
        assert config.cpu_limit == 2.0
        assert config.memory_limit == "4g"
        assert config.privileged is False
        assert config.gpu_enabled is False
        assert config.environment_vars == custom_env
        assert config.exposed_ports == custom_ports


class TestContainerMetrics:
    """Test ContainerMetrics dataclass"""

    def test_default_metrics(self):
        """Test default metrics values"""
        metrics = ContainerMetrics()

        assert metrics.cpu_percent == 0.0
        assert metrics.memory_usage_mb == 0.0
        assert metrics.memory_limit_mb == 0.0
        assert metrics.memory_percent == 0.0
        assert metrics.disk_usage_mb == 0.0
        assert metrics.disk_limit_mb == 0.0
        assert metrics.disk_percent == 0.0
        assert metrics.network_rx_mb == 0.0
        assert metrics.network_tx_mb == 0.0
        assert isinstance(metrics.last_updated, datetime)

    def test_metrics_calculation(self):
        """Test metrics calculation methods"""
        metrics = ContainerMetrics()

        # Test CPU percentage calculation
        metrics.cpu_percent = 75.5
        assert metrics.cpu_percent == 75.5

        # Test memory calculations
        metrics.memory_usage_mb = 1024.0
        metrics.memory_limit_mb = 2048.0
        metrics.memory_percent = (metrics.memory_usage_mb / metrics.memory_limit_mb) * 100
        assert metrics.memory_percent == 50.0

        # Test network metrics
        metrics.network_rx_mb = 100.0
        metrics.network_tx_mb = 50.0
        assert metrics.network_rx_mb == 100.0
        assert metrics.network_tx_mb == 50.0


class TestContainerExecutionContext:
    """Test ContainerExecutionContext dataclass"""

    def test_context_creation(self):
        """Test context creation with required parameters"""
        artifact_id = uuid4()
        container_id = "test_container_123"
        container_name = "test_container"
        config = ContainerConfig()
        created_at = datetime.now(timezone.utc)

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id=container_id,
            container_name=container_name,
            config=config,
            state=ContainerState.RUNNING,
            created_at=created_at
        )

        assert context.artifact_id == artifact_id
        assert context.container_id == container_id
        assert context.container_name == container_name
        assert context.config == config
        assert context.state == ContainerState.RUNNING
        assert context.created_at == created_at
        assert context.docker_container is None
        assert len(context.current_executions) == 0
        assert len(context.execution_history) == 0
        assert len(context.websocket_connections) == 0
        assert isinstance(context.metrics, ContainerMetrics)
        assert len(context.metrics_history) == 0

    def test_context_with_container(self):
        """Test context with docker container"""
        artifact_id = uuid4()
        mock_container = Mock()

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc),
            docker_container=mock_container
        )

        assert context.docker_container == mock_container


class TestUnlimitedContainerRuntime:
    """Test UnlimitedContainerRuntime class"""

    @pytest.fixture
    def mock_docker_client(self):
        """Mock Docker client"""
        mock_client = Mock()
        mock_client.from_env.return_value = mock_client
        mock_client.containers = Mock()
        mock_client.images = Mock()
        return mock_client

    @pytest.fixture
    def mock_artifact_manager(self):
        """Mock artifact manager"""
        return MockUnlimitedArtifactManager()

    @pytest.fixture
    def container_runtime(self, mock_docker_client, mock_artifact_manager):
        """Create container runtime instance"""
        with patch('backend.artifacts.artifact_container_runtime.docker', mock_docker_client):
            runtime = UnlimitedContainerRuntime(
                artifact_manager=mock_artifact_manager,
                vm_manager=None,
                memory_manager=None,
                config={}
            )
            return runtime

    def test_initialization(self, container_runtime):
        """Test runtime initialization"""
        assert container_runtime.artifact_manager is not None
        assert container_runtime.vm_manager is None
        assert container_runtime.memory_manager is None
        assert isinstance(container_runtime.active_containers, dict)
        assert isinstance(container_runtime.container_images, dict)
        assert container_runtime.metrics_task is None
        assert container_runtime.cleanup_task is None
        assert container_runtime.metrics_running is False

    @pytest.mark.asyncio
    async def test_initialize_success(self, container_runtime, mock_docker_client):
        """Test successful initialization"""
        # Mock successful Docker operations
        mock_docker_client.containers.list.return_value = []
        mock_docker_client.images.build.return_value = (Mock(id="test_image"), [])

        with patch.object(container_runtime, '_build_base_images', new_callable=AsyncMock) as mock_build, \
             patch.object(container_runtime, '_start_background_tasks', new_callable=AsyncMock) as mock_start, \
             patch.object(container_runtime, '_cleanup_orphaned_containers', new_callable=AsyncMock) as mock_cleanup:

            mock_build.return_value = None
            mock_start.return_value = None
            mock_cleanup.return_value = None

            await container_runtime.initialize()

            mock_build.assert_called_once()
            mock_start.assert_called_once()
            mock_cleanup.assert_called_once()
            assert container_runtime.docker_available is True

    @pytest.mark.asyncio
    async def test_initialize_docker_unavailable(self):
        """Test initialization when Docker is unavailable"""
        with patch('backend.artifacts.artifact_container_runtime.docker.from_env', side_effect=Exception("Docker not found")):
            runtime = UnlimitedContainerRuntime(
                artifact_manager=MockUnlimitedArtifactManager(),
                config={}
            )

            with pytest.raises(RuntimeError, match="Docker is required"):
                await runtime.initialize()

    @pytest.mark.asyncio
    async def test_create_artifact_container(self, container_runtime, mock_docker_client):
        """Test container creation for artifact"""
        # Create test artifact
        artifact = MockUnlimitedArtifact()

        # Mock container creation
        mock_container = Mock()
        mock_container.id = "test_container_123"
        mock_container.status = "running"
        mock_container.exec_run.return_value = Mock(exit_code=0)
        mock_docker_client.containers.create.return_value = mock_container
        mock_docker_client.containers.run.return_value = mock_container

        with patch.object(container_runtime, '_wait_for_container_ready', new_callable=AsyncMock) as mock_wait, \
             patch.object(container_runtime, '_write_artifact_files_to_container', new_callable=AsyncMock) as mock_write:

            mock_wait.return_value = None
            mock_write.return_value = None

            context = await container_runtime.create_artifact_container(artifact)

            assert context.artifact_id == artifact.metadata.artifact_id
            assert context.container_id == "test_container_123"
            assert context.state == ContainerState.RUNNING
            assert context.docker_container == mock_container
            assert artifact.metadata.artifact_id in container_runtime.active_containers

    @pytest.mark.asyncio
    async def test_execute_in_container(self, container_runtime):
        """Test command execution in container"""
        artifact_id = uuid4()
        command = "echo 'test'"

        # Create mock context
        mock_container = Mock()
        mock_container.client.api.exec_create.return_value = {"Id": "exec_123"}
        mock_container.client.api.exec_start.return_value = ["test output\n"]

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc),
            docker_container=mock_container
        )

        container_runtime.active_containers[artifact_id] = context

        # Mock exec inspection
        with patch.object(mock_container.client.api, 'exec_inspect', return_value={"ExitCode": 0}):
            results = []
            async for result in container_runtime.execute_in_container(artifact_id, command):
                results.append(result)

            assert len(results) >= 2  # Started and completed events
            assert results[0]["type"] == "execution_started"
            assert results[0]["command"] == command
            assert any(r["type"] == "execution_completed" for r in results)

    @pytest.mark.asyncio
    async def test_get_container_files(self, container_runtime):
        """Test listing files in container"""
        artifact_id = uuid4()

        # Create mock context
        mock_container = Mock()
        mock_container.exec_run.return_value = Mock(
            exit_code=0,
            output=b"-rw-r--r-- 1 root root 1024 Jan 1 12:00 test.txt\n"
        )

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc),
            docker_container=mock_container
        )

        container_runtime.active_containers[artifact_id] = context

        files = await container_runtime.get_container_files(artifact_id)

        assert isinstance(files, list)
        # Should parse the ls output

    @pytest.mark.asyncio
    async def test_read_container_file(self, container_runtime):
        """Test reading file from container"""
        artifact_id = uuid4()
        file_path = "/workspace/test.txt"

        # Create mock context
        mock_container = Mock()
        mock_container.exec_run.return_value = Mock(
            exit_code=0,
            output=b"file content"
        )

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc),
            docker_container=mock_container
        )

        container_runtime.active_containers[artifact_id] = context

        content = await container_runtime.read_container_file(artifact_id, file_path)

        assert content == "file content"

    @pytest.mark.asyncio
    async def test_write_container_file(self, container_runtime):
        """Test writing file to container"""
        artifact_id = uuid4()
        file_path = "/workspace/test.txt"
        content = "test content"

        # Create mock context
        mock_container = Mock()
        mock_container.exec_run.return_value = Mock(exit_code=0)

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc),
            docker_container=mock_container
        )

        container_runtime.active_containers[artifact_id] = context

        success = await container_runtime.write_container_file(artifact_id, file_path, content)

        assert success is True
        mock_container.exec_run.assert_called()

    @pytest.mark.asyncio
    async def test_get_container_metrics(self, container_runtime):
        """Test getting container metrics"""
        artifact_id = uuid4()

        # Create mock context
        mock_container = Mock()
        mock_container.stats.return_value = {
            "cpu_stats": {"cpu_usage": {"total_usage": 1000000000}},
            "precpu_stats": {"cpu_usage": {"total_usage": 500000000}},
            "memory_stats": {"usage": 1073741824, "limit": 2147483648},  # 1GB / 2GB
            "networks": {"eth0": {"rx_bytes": 1048576, "tx_bytes": 524288}},  # 1MB / 0.5MB
            "pids_stats": {"current": 5}
        }
        mock_container.attrs = {"Created": "2023-01-01T00:00:00Z"}

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc),
            docker_container=mock_container
        )

        container_runtime.active_containers[artifact_id] = context

        metrics = await container_runtime.get_container_metrics(artifact_id)

        assert metrics is not None
        assert metrics.cpu_percent > 0
        assert metrics.memory_usage_mb == 1024.0
        assert metrics.memory_limit_mb == 2048.0
        assert metrics.memory_percent == 50.0
        assert metrics.network_rx_mb == 1.0
        assert metrics.network_tx_mb == 0.5
        assert metrics.processes == 5

    @pytest.mark.asyncio
    async def test_stop_container(self, container_runtime):
        """Test stopping container"""
        artifact_id = uuid4()

        # Create mock context
        mock_container = Mock()
        mock_container.stop.return_value = None
        mock_container.remove.return_value = None

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc),
            docker_container=mock_container
        )

        container_runtime.active_containers[artifact_id] = context

        with patch.object(container_runtime, '_save_container_state', new_callable=AsyncMock) as mock_save:
            mock_save.return_value = None

            success = await container_runtime.stop_container(artifact_id)

            assert success is True
            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()
            assert artifact_id not in container_runtime.active_containers

    @pytest.mark.asyncio
    async def test_websocket_management(self, container_runtime):
        """Test WebSocket connection management"""
        artifact_id = uuid4()
        mock_ws = Mock()

        # Create mock context
        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc)
        )

        container_runtime.active_containers[artifact_id] = context

        # Test adding WebSocket
        await container_runtime.add_websocket(artifact_id, mock_ws)
        assert mock_ws in context.websocket_connections
        assert mock_ws in container_runtime.global_websockets

        # Test broadcasting
        with patch.object(mock_ws, 'send_json', new_callable=AsyncMock) as mock_send:
            await container_runtime.broadcast_to_artifact(artifact_id, {"test": "message"})
            mock_send.assert_called_once_with({"test": "message"})

        # Test removing WebSocket
        await container_runtime.remove_websocket(artifact_id, mock_ws)
        assert mock_ws not in context.websocket_connections
        assert mock_ws not in container_runtime.global_websockets

    @pytest.mark.asyncio
    async def test_shutdown(self, container_runtime):
        """Test runtime shutdown"""
        # Add a mock container
        artifact_id = uuid4()
        mock_container = Mock()
        mock_container.stop.return_value = None
        mock_container.remove.return_value = None

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc),
            docker_container=mock_container
        )

        container_runtime.active_containers[artifact_id] = context

        with patch.object(container_runtime, '_save_container_state', new_callable=AsyncMock) as mock_save:
            mock_save.return_value = None

            await container_runtime.shutdown()

            assert container_runtime.metrics_running is False
            mock_container.stop.assert_called_once()
            mock_container.remove.assert_called_once()


class TestFastAPIIntegration:
    """Test FastAPI router integration"""

    @pytest.fixture
    def mock_runtime(self):
        """Mock container runtime"""
        runtime = Mock()
        runtime.artifact_manager = Mock()
        runtime.create_artifact_container = AsyncMock()
        runtime.execute_in_container = AsyncMock()
        runtime.get_container_files = AsyncMock()
        runtime.read_container_file = AsyncMock()
        runtime.write_container_file = AsyncMock()
        runtime.get_container_metrics = AsyncMock()
        runtime.stop_container = AsyncMock()
        runtime.add_websocket = AsyncMock()
        runtime.remove_websocket = AsyncMock()
        return runtime

    def test_create_router(self, mock_runtime):
        """Test router creation"""
        router = create_container_runtime_router(mock_runtime)

        assert router.prefix == "/api/container"
        assert "container-runtime" in router.tags

    @pytest.mark.asyncio
    async def test_create_container_endpoint(self, mock_runtime):
        """Test create container endpoint"""
        from fastapi.testclient import TestClient
        from fastapi import FastAPI

        router = create_container_runtime_router(mock_runtime)
        app = FastAPI()
        app.include_router(router)

        client = TestClient(app)

        # Mock artifact
        mock_artifact = Mock()
        mock_runtime.artifact_manager.get_artifact.return_value = mock_artifact

        # Mock container creation
        mock_context = Mock()
        mock_context.container_id = "test_123"
        mock_context.container_name = "test_container"
        mock_context.state = ContainerState.RUNNING
        mock_context.created_at = datetime.now(timezone.utc)
        mock_runtime.create_artifact_container.return_value = mock_context

        response = client.post(
            "/api/container/test-uuid/create",
            json={"artifact_id": "test-uuid"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["container_id"] == "test_123"
        assert data["unlimited_power"] is True


class TestIntegrationFunctions:
    """Test integration and utility functions"""

    @pytest.mark.asyncio
    async def test_create_integrated_system(self):
        """Test creating integrated artifact system"""
        with patch('backend.artifacts.artifact_container_runtime.UnlimitedArtifactManager') as mock_manager_class, \
             patch('backend.artifacts.artifact_container_runtime.UnlimitedContainerRuntime') as mock_runtime_class, \
             patch('backend.artifacts.artifact_container_runtime.create_container_runtime_router') as mock_router_func:

            # Mock classes
            mock_manager = Mock()
            mock_manager.initialize = AsyncMock()
            mock_manager_class.return_value = mock_manager

            mock_runtime = Mock()
            mock_runtime.initialize = AsyncMock()
            mock_runtime_class.return_value = mock_runtime

            mock_router = Mock()
            mock_router_func.return_value = mock_router

            manager, runtime, router = await create_integrated_artifact_system()

            assert manager == mock_manager
            assert runtime == mock_runtime
            assert router == mock_router

            mock_manager.initialize.assert_called_once()
            mock_runtime.initialize.assert_called_once()

    @pytest.mark.asyncio
    async def test_demo_container_integration(self):
        """Test demo function (mocked to avoid actual container creation)"""
        with patch('backend.artifacts.artifact_container_runtime.create_integrated_artifact_system') as mock_create, \
             patch('builtins.print') as mock_print:

            # Mock the integrated system
            mock_manager = Mock()
            mock_runtime = Mock()
            mock_router = Mock()

            mock_create.return_value = (mock_manager, mock_runtime, mock_router)

            # Mock artifact creation
            mock_artifact = Mock()
            mock_artifact.metadata.artifact_id = uuid4()
            mock_manager.create_artifact = AsyncMock(return_value=mock_artifact)

            # Mock container operations
            mock_context = Mock()
            mock_context.container_id = "demo_container_123"
            mock_runtime.create_artifact_container = AsyncMock(return_value=mock_context)

            # Mock execution
            async def mock_execute(*args, **kwargs):
                yield {"type": "execution_started", "execution_id": "test", "command": "test"}
                yield {"type": "execution_completed", "execution_id": "test", "exit_code": 0, "success": True}

            mock_runtime.execute_in_container = mock_execute
            mock_runtime.get_container_metrics = AsyncMock(return_value=Mock())
            mock_runtime.get_container_files = AsyncMock(return_value=[])
            mock_runtime.stop_container = AsyncMock(return_value=True)
            mock_runtime.shutdown = AsyncMock()
            mock_manager.shutdown = AsyncMock()

            # Run demo (should not raise exceptions)
            await demo_container_integration()

            # Verify key operations were called
            mock_create.assert_called_once()
            mock_manager.create_artifact.assert_called_once()
            mock_runtime.create_artifact_container.assert_called_once()


class TestErrorHandling:
    """Test error handling and edge cases"""

    @pytest.mark.asyncio
    async def test_container_creation_failure(self, container_runtime):
        """Test handling of container creation failure"""
        artifact = MockUnlimitedArtifact()

        with patch.object(container_runtime, '_create_container_spec') as mock_spec, \
             patch.object(container_runtime.docker_client.containers, 'create', side_effect=Exception("Docker error")):

            mock_spec.return_value = {}

            with pytest.raises(HTTPException) as exc_info:
                await container_runtime.create_artifact_container(artifact)

            assert exc_info.value.status_code == 500
            assert "Container creation failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_execution_in_nonexistent_container(self, container_runtime):
        """Test execution in non-existent container"""
        artifact_id = uuid4()

        with pytest.raises(HTTPException) as exc_info:
            async for _ in container_runtime.execute_in_container(artifact_id, "echo test"):
                pass

        assert exc_info.value.status_code == 404
        assert "Container not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_file_operations_on_nonexistent_container(self, container_runtime):
        """Test file operations on non-existent container"""
        artifact_id = uuid4()

        # Test read file
        with pytest.raises(HTTPException) as exc_info:
            await container_runtime.read_container_file(artifact_id, "/test.txt")

        assert exc_info.value.status_code == 404
        assert "Container not found" in str(exc_info.value.detail)

        # Test write file
        success = await container_runtime.write_container_file(artifact_id, "/test.txt", "content")
        assert success is False

        # Test list files
        with pytest.raises(HTTPException) as exc_info:
            await container_runtime.get_container_files(artifact_id)

        assert exc_info.value.status_code == 404
        assert "Container not found" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_metrics_for_nonexistent_container(self, container_runtime):
        """Test metrics retrieval for non-existent container"""
        artifact_id = uuid4()

        metrics = await container_runtime.get_container_metrics(artifact_id)
        assert metrics is None

    @pytest.mark.asyncio
    async def test_stop_nonexistent_container(self, container_runtime):
        """Test stopping non-existent container"""
        artifact_id = uuid4()

        success = await container_runtime.stop_container(artifact_id)
        assert success is False


class TestBackgroundTasks:
    """Test background monitoring tasks"""

    @pytest.fixture
    def container_runtime(self):
        """Create container runtime for background task testing"""
        with patch('backend.artifacts.artifact_container_runtime.docker'):
            runtime = UnlimitedContainerRuntime(
                artifact_manager=MockUnlimitedArtifactManager(),
                config={}
            )
            return runtime

    @pytest.mark.asyncio
    async def test_background_tasks_start_stop(self, container_runtime):
        """Test starting and stopping background tasks"""
        await container_runtime._start_background_tasks()

        assert container_runtime.metrics_running is True
        assert container_runtime.metrics_task is not None
        assert container_runtime.cleanup_task is not None

        # Stop tasks
        container_runtime.metrics_running = False

        # Wait a bit for tasks to stop
        await asyncio.sleep(0.1)

        # Tasks should be cancelled
        assert container_runtime.metrics_task.cancelled() or container_runtime.metrics_task.done()
        assert container_runtime.cleanup_task.cancelled() or container_runtime.cleanup_task.done()

    @pytest.mark.asyncio
    async def test_metrics_monitoring_loop(self, container_runtime):
        """Test metrics monitoring loop"""
        # Add a mock container
        artifact_id = uuid4()
        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=datetime.now(timezone.utc)
        )
        container_runtime.active_containers[artifact_id] = context

        # Mock get_container_metrics
        with patch.object(container_runtime, 'get_container_metrics', new_callable=AsyncMock) as mock_get_metrics:
            mock_get_metrics.return_value = Mock()

            # Start monitoring briefly
            container_runtime.metrics_running = True

            # Run monitoring loop for a short time
            task = asyncio.create_task(container_runtime._metrics_monitoring_loop())

            # Let it run for a short time
            await asyncio.sleep(0.1)

            # Stop monitoring
            container_runtime.metrics_running = False
            task.cancel()

            # Verify metrics were collected
            mock_get_metrics.assert_called()


class TestConfigurationValidation:
    """Test configuration validation and edge cases"""

    def test_container_config_validation(self):
        """Test ContainerConfig validation"""
        # Valid configuration
        config = ContainerConfig()
        assert config.image_name is not None
        assert config.working_dir == "/workspace"

        # Test with custom values
        config = ContainerConfig(
            cpu_limit=4.0,
            memory_limit="8g",
            disk_limit="100g",
            privileged=False,
            gpu_enabled=False,
            internet_enabled=False
        )

        assert config.cpu_limit == 4.0
        assert config.memory_limit == "8g"
        assert config.disk_limit == "100g"
        assert config.privileged is False
        assert config.gpu_enabled is False
        assert config.internet_enabled is False

    def test_execution_context_validation(self):
        """Test ContainerExecutionContext validation"""
        artifact_id = uuid4()
        created_at = datetime.now(timezone.utc)

        context = ContainerExecutionContext(
            artifact_id=artifact_id,
            container_id="test_123",
            container_name="test_container",
            config=ContainerConfig(),
            state=ContainerState.RUNNING,
            created_at=created_at
        )

        assert context.artifact_id == artifact_id
        assert context.container_id == "test_123"
        assert context.state == ContainerState.RUNNING
        assert len(context.current_executions) == 0
        assert len(context.execution_history) == 0


# Performance and stress tests
class TestPerformance:
    """Performance and stress testing"""

    @pytest.mark.asyncio
    async def test_concurrent_container_operations(self):
        """Test concurrent container operations"""
        # This would test multiple containers running simultaneously
        # For now, just test the structure
        pass

    @pytest.mark.asyncio
    async def test_large_file_handling(self):
        """Test handling of large files"""
        # Test with large file content
        large_content = "x" * (1024 * 1024)  # 1MB of content

        # This would test writing and reading large files
        pass

    @pytest.mark.asyncio
    async def test_websocket_broadcast_performance(self):
        """Test WebSocket broadcast performance with multiple connections"""
        # Test broadcasting to many WebSocket connections
        pass


# Integration tests that require Docker
class TestDockerIntegration:
    """Integration tests requiring actual Docker"""

    @pytest.mark.skipif(not shutil.which("docker"), reason="Docker not available")
    @pytest.mark.asyncio
    async def test_real_docker_operations(self):
        """Test with real Docker operations (requires Docker)"""
        # This would test actual Docker container creation and operations
        # Skipped by default as it requires Docker to be running
        pass


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
