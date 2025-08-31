"""
Test suite for the VM Supervisor component of the Somnus Sovereign Systems.
"""

import pytest
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from uuid import UUID

from backend.virtual_machine.vm_supervisor import (
    VMSupervisor, AIVMInstance, VMState, ResourceProfile, VMSnapshot
)


class TestVMSupervisor:
    """Test suite for the VMSupervisor class."""

    @pytest.fixture
    def temp_storage_path(self):
        """Create a temporary storage path for VM instances."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def vm_supervisor(self, temp_storage_path):
        """Create a VMSupervisor instance with mocked libvirt connection."""
        with patch('backend.virtual_machine.vm_supervisor.libvirt') as mock_libvirt:
            # Mock the libvirt connection
            mock_conn = Mock()
            mock_libvirt.open.return_value = mock_conn
            
            # Create supervisor instance
            supervisor = VMSupervisor(temp_storage_path, {})
            
            # Verify the libvirt connection was established
            mock_libvirt.open.assert_called_once_with('qemu:///system')
            
            yield supervisor

    def test_init(self, vm_supervisor, temp_storage_path):
        """Test VMSupervisor initialization."""
        assert vm_supervisor.vm_storage_path == temp_storage_path
        assert vm_supervisor.vm_instances_path == temp_storage_path / "instances"
        assert vm_supervisor.vm_snapshots_path == temp_storage_path / "snapshots"
        assert isinstance(vm_supervisor.active_vms, dict)
        assert len(vm_supervisor.resource_profiles) > 0

    def test_save_and_load_vm_config(self, vm_supervisor, temp_storage_path):
        """Test saving and loading VM configuration."""
        # Create a test VM instance
        vm_instance = AIVMInstance(
            instance_name="test-ai-instance",
            libvirt_domain_name="test-domain",
            vm_disk_path="/tmp/test-disk.qcow2"
        )
        
        # Save the VM configuration
        vm_supervisor._save_vm_config(vm_instance)
        
        # Check that the file was created
        config_path = vm_supervisor.vm_instances_path / f"{vm_instance.vm_id}.json"
        assert config_path.exists()
        
        # Load the VM configuration
        with patch('backend.virtual_machine.vm_supervisor.libvirt'):
            vm_supervisor._load_vms_from_disk()
        
        # Verify the VM was loaded
        assert vm_instance.vm_id in vm_supervisor.active_vms
        loaded_vm = vm_supervisor.active_vms[vm_instance.vm_id]
        assert loaded_vm.instance_name == vm_instance.instance_name
        assert loaded_vm.libvirt_domain_name == vm_instance.libvirt_domain_name

    def test_create_snapshot(self, vm_supervisor):
        """Test creating a snapshot of a VM."""
        # Mock libvirt domain
        mock_domain = Mock()
        mock_snapshot = Mock()
        mock_domain.snapshotCreateXML.return_value = mock_snapshot
        
        # Create a test VM instance
        vm_instance = AIVMInstance(
            instance_name="test-ai-instance",
            libvirt_domain_name="test-domain",
            vm_disk_path="/tmp/test-disk.qcow2"
        )
        vm_supervisor.active_vms[vm_instance.vm_id] = vm_instance
        
        # Mock the _get_libvirt_domain method
        with patch.object(vm_supervisor, '_get_libvirt_domain', return_value=mock_domain):
            # Create a snapshot
            snapshot = vm_supervisor.create_snapshot(
                vm_instance.vm_id, 
                "Test snapshot description"
            )
            
            # Verify the snapshot was created
            assert isinstance(snapshot, VMSnapshot)
            assert len(vm_instance.snapshots) == 1
            assert vm_instance.snapshots[0] == snapshot
            assert snapshot.description == "Test snapshot description"
            
            # Verify libvirt was called correctly
            mock_domain.snapshotCreateXML.assert_called_once()

    def test_rollback_to_snapshot(self, vm_supervisor):
        """Test rolling back a VM to a snapshot."""
        # Mock libvirt domain and snapshot
        mock_domain = Mock()
        mock_snapshot = Mock()
        mock_domain.snapshotLookupByName.return_value = mock_snapshot
        
        # Create a test VM instance with a snapshot
        vm_instance = AIVMInstance(
            instance_name="test-ai-instance",
            libvirt_domain_name="test-domain",
            vm_disk_path="/tmp/test-disk.qcow2"
        )
        snapshot = VMSnapshot(
            snapshot_name="test-snapshot",
            description="Test snapshot"
        )
        vm_instance.snapshots.append(snapshot)
        vm_supervisor.active_vms[vm_instance.vm_id] = vm_instance
        
        # Mock the _get_libvirt_domain method
        with patch.object(vm_supervisor, '_get_libvirt_domain', return_value=mock_domain):
            # Rollback to the snapshot
            result = vm_supervisor.rollback_to_snapshot(
                vm_instance.vm_id, 
                "test-snapshot"
            )
            
            # Verify the rollback was successful
            assert result is True
            assert vm_instance.vm_state == VMState.RUNNING
            mock_domain.revertToSnapshot.assert_called_once_with(mock_snapshot)

    def test_apply_resource_profile(self, vm_supervisor):
        """Test applying a resource profile to a VM."""
        # Mock libvirt domain
        mock_domain = Mock()
        mock_domain.isActive.return_value = True
        
        # Create a test VM instance
        vm_instance = AIVMInstance(
            instance_name="test-ai-instance",
            libvirt_domain_name="test-domain",
            vm_disk_path="/tmp/test-disk.qcow2"
        )
        vm_supervisor.active_vms[vm_instance.vm_id] = vm_instance
        
        # Mock the _get_libvirt_domain method
        with patch.object(vm_supervisor, '_get_libvirt_domain', return_value=mock_domain):
            # Apply a resource profile
            result = vm_supervisor.apply_resource_profile(
                vm_instance.vm_id, 
                "coding"
            )
            
            # Verify the profile was applied
            assert result is True
            assert vm_instance.current_profile == "coding"
            assert vm_instance.vm_state == VMState.RUNNING
            
            # Verify libvirt was called correctly
            mock_domain.setMemory.assert_called_once()
            mock_domain.setVcpus.assert_called_once()

    def test_shutdown_vm(self, vm_supervisor):
        """Test shutting down a VM."""
        # Mock libvirt domain
        mock_domain = Mock()
        mock_domain.isActive.return_value = True
        
        # Create a test VM instance
        vm_instance = AIVMInstance(
            instance_name="test-ai-instance",
            libvirt_domain_name="test-domain",
            vm_disk_path="/tmp/test-disk.qcow2"
        )
        vm_supervisor.active_vms[vm_instance.vm_id] = vm_instance
        
        # Mock the _get_libvirt_domain method
        with patch.object(vm_supervisor, '_get_libvirt_domain', return_value=mock_domain):
            # Shutdown the VM
            result = vm_supervisor.shutdown_vm(vm_instance.vm_id)
            
            # Verify the shutdown was successful
            assert result is True
            assert vm_instance.vm_state == VMState.SHUTDOWN
            mock_domain.shutdown.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__])