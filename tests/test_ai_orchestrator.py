"""
Test suite for the AI Orchestrator component of the Somnus Sovereign Systems.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, MagicMock
from uuid import UUID

from backend.virtual_machine.ai_orchestrator import (
    SovereignAIOrchestrator, CAPABILITY_PACKS
)


class TestSovereignAIOrchestrator:
    """Test suite for the SovereignAIOrchestrator class."""

    @pytest.fixture
    def mock_managers(self):
        """Create mock manager instances."""
        return {
            'vm_supervisor': Mock(),
            'dev_session_manager': Mock(),
            'memory_manager': Mock(),
            'security_enforcer': Mock()
        }

    @pytest.fixture
    def orchestrator(self, mock_managers):
        """Create a SovereignAIOrchestrator instance with mocked dependencies."""
        return SovereignAIOrchestrator(**mock_managers)

    def test_init(self, orchestrator, mock_managers):
        """Test orchestrator initialization."""
        assert orchestrator.vm_supervisor == mock_managers['vm_supervisor']
        assert orchestrator.dev_session_manager == mock_managers['dev_session_manager']
        assert orchestrator.memory_manager == mock_managers['memory_manager']
        assert orchestrator.security_enforcer == mock_managers['security_enforcer']

    @pytest.mark.asyncio
    async def test_provision_sovereign_environment_success(self, orchestrator):
        """Test successful provisioning of a sovereign environment."""
        # Mock the VM supervisor's create_ai_computer method
        mock_vm_instance = Mock()
        mock_vm_instance.vm_id = UUID('12345678-1234-5678-1234-567812345678')
        mock_vm_instance.internal_ip = "192.168.122.10"
        mock_vm_instance.ssh_port = 2222
        mock_vm_instance.vnc_port = 5900
        orchestrator.vm_supervisor.create_ai_computer.return_value = mock_vm_instance

        # Mock the dev session manager's create_session method
        mock_dev_session = Mock()
        mock_dev_session.dev_session_id = UUID('87654321-4321-8765-4321-876543218765')
        orchestrator.dev_session_manager.create_session.return_value = mock_dev_session

        # Test provisioning
        result = await orchestrator.provision_sovereign_environment(
            user_id="test_user",
            session_title="Test Session",
            personality_config={"personality": "test"},
            requested_capabilities=["web_development"]
        )

        # Verify the result
        assert result["status"] == "success"
        assert result["dev_session_id"] == mock_dev_session.dev_session_id
        assert result["vm_id"] == mock_vm_instance.vm_id
        assert result["vm_ip"] == mock_vm_instance.internal_ip
        assert "web_development" in result["installed_capabilities"]
        assert "base_tools" in result["installed_capabilities"]

        # Verify method calls
        orchestrator.vm_supervisor.create_ai_computer.assert_called_once()
        orchestrator.dev_session_manager.create_session.assert_called_once()
        mock_dev_session.add_event.assert_called()
        orchestrator.dev_session_manager._save_session.assert_called_once()
        orchestrator.vm_supervisor._save_vm_config.assert_called_once()

    @pytest.mark.asyncio
    async def test_provision_sovereign_environment_vm_failure(self, orchestrator):
        """Test provisioning when VM creation fails."""
        # Mock the VM supervisor to raise an exception
        orchestrator.vm_supervisor.create_ai_computer.side_effect = Exception("VM creation failed")

        # Test provisioning
        result = await orchestrator.provision_sovereign_environment(
            user_id="test_user",
            session_title="Test Session",
            personality_config={"personality": "test"},
            requested_capabilities=["web_development"]
        )

        # Verify the result
        assert result["status"] == "error"
        assert "Failed to provision VM" in result["message"]

    def test_capability_packs_structure(self):
        """Test that capability packs have the correct structure."""
        for pack_name, pack_data in CAPABILITY_PACKS.items():
            assert "description" in pack_data
            assert "commands" in pack_data
            assert isinstance(pack_data["description"], str)
            assert isinstance(pack_data["commands"], list)
            assert len(pack_data["commands"]) > 0

    @pytest.mark.asyncio
    async def test_get_environment_status_success(self, orchestrator):
        """Test getting environment status when everything exists."""
        # Mock the dev session manager
        mock_dev_session = Mock()
        mock_dev_session.vm_instance_id = UUID('12345678-1234-5678-1234-567812345678')
        orchestrator.dev_session_manager.get_session.return_value = mock_dev_session

        # Mock the VM supervisor
        mock_vm_instance = Mock()
        mock_vm_instance.vm_state = "running"
        mock_vm_instance.current_profile = "coding"
        mock_vm_instance.internal_ip = "192.168.122.10"
        mock_vm_instance.snapshots = []
        mock_vm_instance.runtime_stats_history = []
        orchestrator.vm_supervisor.active_vms = {
            mock_dev_session.vm_instance_id: mock_vm_instance
        }

        # Test getting status
        result = await orchestrator.get_environment_status(mock_dev_session.dev_session_id)

        # Verify the result
        assert result is not None
        assert "dev_session" in result
        assert "vm_status" in result
        assert result["vm_status"]["state"] == "running"
        assert result["vm_status"]["profile"] == "coding"
        assert result["vm_status"]["ip"] == "192.168.122.10"

    @pytest.mark.asyncio
    async def test_get_environment_status_no_session(self, orchestrator):
        """Test getting environment status when session doesn't exist."""
        # Mock the dev session manager to return None
        orchestrator.dev_session_manager.get_session.return_value = None

        # Test getting status
        result = await orchestrator.get_environment_status(UUID('12345678-1234-5678-1234-567812345678'))

        # Verify the result
        assert result is None

    @pytest.mark.asyncio
    async def test_get_environment_status_no_vm(self, orchestrator):
        """Test getting environment status when VM doesn't exist."""
        # Mock the dev session manager
        mock_dev_session = Mock()
        mock_dev_session.vm_instance_id = UUID('12345678-1234-5678-1234-567812345678')
        orchestrator.dev_session_manager.get_session.return_value = mock_dev_session

        # Mock the VM supervisor with no active VMs
        orchestrator.vm_supervisor.active_vms = {}

        # Test getting status
        result = await orchestrator.get_environment_status(mock_dev_session.dev_session_id)

        # Verify the result
        assert result is None

    @pytest.mark.asyncio
    async def test_shutdown_environment_success(self, orchestrator):
        """Test successful shutdown of an environment."""
        # Mock the dev session manager
        mock_dev_session = Mock()
        mock_dev_session.vm_instance_id = UUID('12345678-1234-5678-1234-567812345678')
        orchestrator.dev_session_manager.get_session.return_value = mock_dev_session

        # Mock the VM supervisor
        orchestrator.vm_supervisor.shutdown_vm.return_value = True

        # Test shutdown
        result = await orchestrator.shutdown_environment(mock_dev_session.dev_session_id)

        # Verify the result
        assert result is True
        orchestrator.vm_supervisor.shutdown_vm.assert_called_once_with(mock_dev_session.vm_instance_id)
        orchestrator.dev_session_manager.set_session_status.assert_called_once()

    @pytest.mark.asyncio
    async def test_shutdown_environment_no_session(self, orchestrator):
        """Test shutdown when session doesn't exist."""
        # Mock the dev session manager to return None
        orchestrator.dev_session_manager.get_session.return_value = None

        # Test shutdown
        result = await orchestrator.shutdown_environment(UUID('12345678-1234-5678-1234-567812345678'))

        # Verify the result
        assert result is False


if __name__ == "__main__":
    pytest.main([__file__])