"""
Integration tests for the VM Orchestrator with real session management components.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from uuid import UUID, uuid4

from backend.virtual_machine.ai_orchestrator import SovereignAIOrchestrator, CAPABILITY_PACKS
from schemas.dev_session_schemas import DevSession, DevSessionStatus
from core.session_manager import DevSessionManager
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope


class TestSovereignAIOrchestratorIntegration:
    """Integration tests for the SovereignAIOrchestrator with real components."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary storage path for VM instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def mock_vm_supervisor(self):
        """Create a mock VMSupervisor."""
        with patch('backend.virtual_machine.ai_orchestrator.VMSupervisor') as mock_supervisor:
            # Mock the VMSupervisor methods
            mock_instance = Mock()
            mock_instance.create_ai_computer = AsyncMock()
            mock_instance.create_snapshot = Mock()
            mock_instance.shutdown_vm = Mock(return_value=True)
            mock_instance.active_vms = {}
            
            # Mock VM instance
            mock_vm_instance = Mock()
            mock_vm_instance.vm_id = uuid4()
            mock_vm_instance.internal_ip = "192.168.122.100"
            mock_vm_instance.ssh_port = 2222
            mock_vm_instance.vnc_port = 5900
            mock_vm_instance.installed_tools = []
            
            mock_instance.create_ai_computer.return_value = mock_vm_instance
            mock_supervisor.return_value = mock_instance
            
            yield mock_instance

    @pytest.fixture
    def memory_manager(self):
        """Create a real MemoryManager instance."""
        return MemoryManager(base_path=Path("./test_memory"), chroma_db_path=Path("./test_chroma"))

    @pytest.fixture
    def dev_session_manager(self, memory_manager):
        """Create a real DevSessionManager instance."""
        return DevSessionManager(memory_manager)

    @pytest.fixture
    def mock_security_enforcer(self):
        """Create a mock SecurityEnforcer."""
        with patch('backend.virtual_machine.ai_orchestrator.SecurityEnforcer') as mock_enforcer:
            mock_instance = Mock()
            mock_enforcer.return_value = mock_instance
            yield mock_instance

    @pytest.fixture
    def orchestrator(self, mock_vm_supervisor, dev_session_manager, memory_manager, mock_security_enforcer):
        """Create a SovereignAIOrchestrator instance."""
        return SovereignAIOrchestrator(
            vm_supervisor=mock_vm_supervisor,
            dev_session_manager=dev_session_manager,
            memory_manager=memory_manager,
            security_enforcer=mock_security_enforcer
        )

    @pytest.mark.asyncio
    async def test_provision_sovereign_environment(self, orchestrator, mock_vm_supervisor):
        """Test provisioning a new sovereign environment."""
        user_id = "test_user"
        session_title = "Test AI Environment"
        personality_config = {"personality": "helpful"}
        requested_capabilities = ["web_development"]

        # Test the provisioning workflow
        result = await orchestrator.provision_sovereign_environment(
            user_id=user_id,
            session_title=session_title,
            personality_config=personality_config,
            requested_capabilities=requested_capabilities
        )

        # Verify the result
        assert result["status"] == "success"
        assert "dev_session_id" in result
        assert "vm_id" in result
        assert "installed_capabilities" in result
        
        # Verify VM supervisor was called correctly
        mock_vm_supervisor.create_ai_computer.assert_called_once()
        mock_vm_supervisor.create_snapshot.assert_called()
        
        # Verify the correct capabilities were installed
        installed_caps = result["installed_capabilities"]
        assert "base_tools" in installed_caps
        assert "web_development" in installed_caps

    @pytest.mark.asyncio
    async def test_get_environment_status(self, orchestrator, mock_vm_supervisor, dev_session_manager):
        """Test getting environment status."""
        user_id = "test_user"
        session_title = "Test AI Environment"
        personality_config = {"personality": "helpful"}
        requested_capabilities = ["base_tools"]

        # First provision an environment
        result = await orchestrator.provision_sovereign_environment(
            user_id=user_id,
            session_title=session_title,
            personality_config=personality_config,
            requested_capabilities=requested_capabilities
        )
        
        dev_session_id = result["dev_session_id"]
        
        # Mock the VM instance for status check
        mock_vm_instance = Mock()
        mock_vm_instance.vm_state = "running"
        mock_vm_instance.current_profile = "idle"
        mock_vm_instance.internal_ip = "192.168.122.100"
        mock_vm_instance.snapshots = []
        mock_vm_supervisor.active_vms = {result["vm_id"]: mock_vm_instance}
        
        # Test getting environment status
        status = await orchestrator.get_environment_status(UUID(dev_session_id))
        
        # Verify the status
        assert status is not None
        assert "dev_session" in status
        assert "vm_status" in status
        assert status["vm_status"]["state"] == "running"
        assert status["vm_status"]["profile"] == "idle"

    @pytest.mark.asyncio
    async def test_shutdown_environment(self, orchestrator, mock_vm_supervisor, dev_session_manager):
        """Test shutting down an environment."""
        user_id = "test_user"
        session_title = "Test AI Environment"
        personality_config = {"personality": "helpful"}
        requested_capabilities = ["base_tools"]

        # First provision an environment
        result = await orchestrator.provision_sovereign_environment(
            user_id=user_id,
            session_title=session_title,
            personality_config=personality_config,
            requested_capabilities=requested_capabilities
        )
        
        dev_session_id = result["dev_session_id"]
        
        # Test shutting down the environment
        success = await orchestrator.shutdown_environment(UUID(dev_session_id))
        
        # Verify shutdown was successful
        assert success is True
        mock_vm_supervisor.shutdown_vm.assert_called_once()


# Helper class for async mock
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs)


if __name__ == "__main__":
    pytest.main([__file__])