"""
MORPHEUS CHAT - Project Virtual Machine Manager
Specialized VM management for project environments

Each project gets its own persistent VM with:
- Loaded AI model
- Project workspace
- Autonomous intelligence systems
- Unlimited storage and compute
- Custom development environment
"""

import asyncio
import json
import logging
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass
from enum import Enum

import psutil

from .project_core import ProjectSpecs

logger = logging.getLogger(__name__)


class ProjectVMState(str, Enum):
    """Project VM states"""
    CREATING = "creating"
    INITIALIZING = "initializing"
    RUNNING = "running"
    SUSPENDED = "suspended"
    MAINTENANCE = "maintenance"
    ERROR = "error"
    DESTROYED = "destroyed"


@dataclass
class ProjectVMInstance:
    """Project VM instance with enhanced capabilities"""
    vm_id: UUID
    project_id: UUID
    project_name: str
    
    # VM configuration
    vcpus: int
    memory_gb: int
    storage_gb: int
    
    # Network and access
    ssh_port: int
    vnc_port: int
    web_port: int
    internal_ip: Optional[str] = None
    
    # State tracking
    state: ProjectVMState = ProjectVMState.CREATING
    created_at: datetime = None
    last_active: datetime = None
    
    # Project-specific features
    model_loaded: Optional[str] = None
    intelligence_running: bool = False
    collaboration_enabled: bool = False
    
    # Resource usage
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now(timezone.utc)
        if self.last_active is None:
            self.last_active = datetime.now(timezone.utc)


class ProjectVMManager:
    """
    Specialized VM manager for project environments
    
    Creates and manages persistent VMs that are optimized for:
    - AI model execution
    - File processing and organization
    - Development environments
    - Collaborative workspaces
    """
    
    def __init__(self, vm_storage_path: Path, base_vm_manager, config: Dict[str, Any]):
        self.vm_storage_path = vm_storage_path
        self.vm_storage_path.mkdir(parents=True, exist_ok=True)
        
        self.base_vm_manager = base_vm_manager  # Reference to main VM manager
        self.config = config
        
        # Project VM tracking
        self.project_vms: Dict[UUID, ProjectVMInstance] = {}
        self.port_manager = ProjectPortManager(config.get('port_range', (8000, 9000)))
        
        # VM templates and images
        self.project_vm_template = vm_storage_path / "project_vm_template.qcow2"
        
        logger.info("Project VM Manager initialized")
    
    async def create_project_vm(
        self,
        project_id: UUID,
        vm_specs: ProjectSpecs,
        project_name: str
    ) -> ProjectVMInstance:
        """
        Create dedicated VM for project with optimized configuration
        """
        
        vm_id = uuid4()
        
        # Allocate ports for VM services
        ssh_port = self.port_manager.allocate_port()
        vnc_port = self.port_manager.allocate_port()
        web_port = self.port_manager.allocate_port()
        
        vm_instance = ProjectVMInstance(
            vm_id=vm_id,
            project_id=project_id,
            project_name=project_name,
            vcpus=vm_specs.vcpus,
            memory_gb=vm_specs.memory_gb,
            storage_gb=vm_specs.storage_gb,
            ssh_port=ssh_port,
            vnc_port=vnc_port,
            web_port=web_port,
            state=ProjectVMState.CREATING
        )
        
        try:
            # Create VM disk image
            vm_disk_path = await self._create_project_vm_disk(vm_instance)
            
            # Generate VM configuration
            vm_config = await self._generate_project_vm_config(vm_instance, vm_disk_path)
            
            # Create and start VM
            await self._create_and_start_vm(vm_instance, vm_config)
            
            # Initialize project environment
            await self._initialize_project_environment(vm_instance, vm_specs)
            
            # Setup project services
            await self._setup_project_services(vm_instance)
            
            # Track VM
            self.project_vms[vm_id] = vm_instance
            vm_instance.state = ProjectVMState.RUNNING
            
            logger.info(f"Created project VM {vm_id} for project {project_name}")
            return vm_instance
            
        except Exception as e:
            vm_instance.state = ProjectVMState.ERROR
            logger.error(f"Failed to create project VM: {e}")
            # Cleanup on failure
            await self._cleanup_failed_vm(vm_instance)
            raise
    
    async def _create_project_vm_disk(self, vm_instance: ProjectVMInstance) -> Path:
        """Create VM disk image from template"""
        
        # Ensure project VM template exists
        if not self.project_vm_template.exists():
            await self._create_project_vm_template()
        
        # Create project-specific disk
        vm_disk_path = self.vm_storage_path / f"project_{vm_instance.vm_id}.qcow2"
        
        # Clone template and resize
        clone_cmd = [
            "qemu-img", "create", "-f", "qcow2",
            "-o", f"backing_file={self.project_vm_template}",
            str(vm_disk_path)
        ]
        
        subprocess.run(clone_cmd, check=True)
        
        # Resize to requested storage
        resize_cmd = [
            "qemu-img", "resize", str(vm_disk_path), f"{vm_instance.storage_gb}G"
        ]
        
        subprocess.run(resize_cmd, check=True)
        
        return vm_disk_path
    
    async def _create_project_vm_template(self):
        """Create optimized VM template for projects"""
        
        logger.info("Creating project VM template...")
        
        # Download Ubuntu Server ISO if needed
        ubuntu_iso = self.vm_storage_path / "ubuntu-22.04-server-amd64.iso"
        
        if not ubuntu_iso.exists():
            logger.info("Downloading Ubuntu Server ISO...")
            # In production, implement proper ISO download
            # For now, assume ISO exists or create minimal template
        
        # Create base template disk
        template_cmd = [
            "qemu-img", "create", "-f", "qcow2", 
            str(self.project_vm_template), "20G"
        ]
        
        subprocess.run(template_cmd, check=True)
        
        # Install optimized system (in production, automate installation)
        logger.info("Project VM template created")
    
    async def _generate_project_vm_config(self, vm_instance: ProjectVMInstance, disk_path: Path) -> str:
        """Generate libvirt XML configuration for project VM"""
        
        return f"""
        <domain type='kvm'>
          <name>project-{vm_instance.vm_id}</name>
          <memory unit='GiB'>{vm_instance.memory_gb}</memory>
          <vcpu placement='static'>{vm_instance.vcpus}</vcpu>
          
          <os>
            <type arch='x86_64' machine='pc-q35-6.2'>hvm</type>
            <boot dev='hd'/>
          </os>
          
          <features>
            <acpi/>
            <apic/>
            <vmport state='off'/>
          </features>
          
          <cpu mode='host-passthrough' check='none'/>
          
          <clock offset='utc'>
            <timer name='rtc' tickpolicy='catchup'/>
            <timer name='pit' tickpolicy='delay'/>
            <timer name='hpet' present='no'/>
          </clock>
          
          <devices>
            <!-- Primary disk -->
            <disk type='file' device='disk'>
              <driver name='qemu' type='qcow2' cache='writeback'/>
              <source file='{disk_path}'/>
              <target dev='vda' bus='virtio'/>
            </disk>
            
            <!-- Network interface -->
            <interface type='network'>
              <source network='default'/>
              <model type='virtio'/>
            </interface>
            
            <!-- Graphics and remote access -->
            <graphics type='vnc' port='{vm_instance.vnc_port}' autoport='no' listen='0.0.0.0'/>
            
            <!-- Console access -->
            <serial type='pty'>
              <target type='isa-serial' port='0'/>
            </serial>
            <console type='pty'>
              <target type='serial' port='0'/>
            </console>
            
            <!-- USB redirection -->
            <redirdev bus='usb' type='spicevmc'/>
            
            <!-- Shared memory for performance -->
            <memballoon model='virtio'/>
            
            <!-- RNG for better entropy -->
            <rng model='virtio'>
              <backend model='random'>/dev/urandom</backend>
            </rng>
          </devices>
        </domain>
        """
    
    async def _create_and_start_vm(self, vm_instance: ProjectVMInstance, vm_config: str):
        """Create and start VM using libvirt"""
        
        # Save VM config to file
        config_file = self.vm_storage_path / f"project_{vm_instance.vm_id}.xml"
        with open(config_file, 'w') as f:
            f.write(vm_config)
        
        # Define VM
        define_cmd = ["virsh", "define", str(config_file)]
        subprocess.run(define_cmd, check=True)
        
        # Start VM
        start_cmd = ["virsh", "start", f"project-{vm_instance.vm_id}"]
        subprocess.run(start_cmd, check=True)
        
        # Wait for VM to boot
        await self._wait_for_vm_boot(vm_instance)
    
    async def _wait_for_vm_boot(self, vm_instance: ProjectVMInstance, timeout: int = 300):
        """Wait for VM to boot and become accessible"""
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Try to connect via SSH
                test_cmd = [
                    "ssh", "-o", "ConnectTimeout=5", "-o", "StrictHostKeyChecking=no",
                    "-p", str(vm_instance.ssh_port), "root@localhost", "echo ready"
                ]
                
                result = subprocess.run(test_cmd, capture_output=True, timeout=10)
                
                if result.returncode == 0:
                    logger.info(f"VM {vm_instance.vm_id} is ready")
                    return
                    
            except subprocess.TimeoutExpired:
                pass
            
            await asyncio.sleep(5)
        
        raise TimeoutError(f"VM {vm_instance.vm_id} failed to boot within {timeout} seconds")
    
    async def _initialize_project_environment(self, vm_instance: ProjectVMInstance, vm_specs: ProjectSpecs):
        """Initialize project-specific environment in VM"""
        
        init_script = f'''#!/bin/bash
# Project Environment Initialization
set -e

# Update system
apt-get update -y
apt-get upgrade -y

# Install essential packages
apt-get install -y python3 python3-pip nodejs npm git curl wget
apt-get install -y build-essential software-properties-common
apt-get install -y htop tree jq unzip

# Install Python AI/ML packages
pip3 install --upgrade pip
pip3 install numpy pandas matplotlib seaborn plotly
pip3 install torch transformers sentence-transformers
pip3 install fastapi uvicorn websockets aiofiles
pip3 install requests beautifulsoup4 selenium

# Install development tools
apt-get install -y vim nano emacs-nox
npm install -g typescript eslint prettier

# Create project structure
mkdir -p /project/{{files,workspace,knowledge,memory,automation,intelligence,logs,artifacts,collaboration}}
mkdir -p /home/morpheus/{{workspace,tools,scripts}}

# Setup project user
useradd -m -s /bin/bash morpheus
echo "morpheus:morpheus" | chpasswd
usermod -aG sudo morpheus

# Create project configuration
cat > /project/project_config.json << 'EOF'
{{
    "project_id": "{vm_instance.project_id}",
    "project_name": "{vm_instance.project_name}",
    "vm_id": "{vm_instance.vm_id}",
    "created_at": "{vm_instance.created_at.isoformat()}",
    "specs": {{
        "vcpus": {vm_instance.vcpus},
        "memory_gb": {vm_instance.memory_gb},
        "storage_gb": {vm_instance.storage_gb}
    }},
    "ports": {{
        "ssh": {vm_instance.ssh_port},
        "vnc": {vm_instance.vnc_port},
        "web": {vm_instance.web_port}
    }}
}}
EOF

# Set permissions
chown -R morpheus:morpheus /project
chown -R morpheus:morpheus /home/morpheus

# Setup SSH keys for easy access
mkdir -p /home/morpheus/.ssh
echo "StrictHostKeyChecking no" > /home/morpheus/.ssh/config
chown -R morpheus:morpheus /home/morpheus/.ssh

echo "Project environment initialized successfully"
        '''
        
        # Execute initialization script
        await self.execute_command_in_vm(vm_instance, init_script)
    
    async def _setup_project_services(self, vm_instance: ProjectVMInstance):
        """Setup project-specific services"""
        
        # Create project web interface service
        web_service_script = f'''#!/usr/bin/env python3
"""
Project Web Interface - Running on port {vm_instance.web_port}
"""
import asyncio
import json
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
import uvicorn

app = FastAPI(title="Project {vm_instance.project_name}")

@app.get("/")
async def project_dashboard():
    return HTMLResponse("""
    <html>
        <head><title>Project {vm_instance.project_name}</title></head>
        <body>
            <h1>Project: {vm_instance.project_name}</h1>
            <h2>Project ID: {vm_instance.project_id}</h2>
            <p>VM Status: Running</p>
            <p>Intelligence: Active</p>
            <p>Files: <span id="file-count">Loading...</span></p>
            <script>
                fetch('/api/status').then(r=>r.json()).then(data=>{{
                    document.getElementById('file-count').textContent = data.files || 0;
                }});
            </script>
        </body>
    </html>
    """)

@app.get("/api/status")
async def get_status():
    return {{
        "project_id": "{vm_instance.project_id}",
        "vm_id": "{vm_instance.vm_id}",
        "status": "running",
        "files": 0  # Will be updated by intelligence system
    }}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port={vm_instance.web_port})
        '''
        
        await self.write_file_to_vm(
            vm_instance, 
            "/home/morpheus/project_web_service.py", 
            web_service_script
        )
        
        # Create systemd service for web interface
        systemd_service = f'''[Unit]
Description=Project {vm_instance.project_name} Web Interface
After=network.target

[Service]
Type=simple
User=morpheus
WorkingDirectory=/home/morpheus
ExecStart=/usr/bin/python3 /home/morpheus/project_web_service.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
        '''
        
        await self.write_file_to_vm(
            vm_instance,
            f"/etc/systemd/system/project-web-{vm_instance.vm_id}.service",
            systemd_service
        )
        
        # Enable and start service
        await self.execute_command_in_vm(vm_instance, f'''
systemctl daemon-reload
systemctl enable project-web-{vm_instance.vm_id}
systemctl start project-web-{vm_instance.vm_id}
        ''')
    
    async def execute_command_in_vm(self, vm_instance: ProjectVMInstance, command: str) -> str:
        """Execute command in project VM"""
        
        ssh_cmd = [
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30",
            "-p", str(vm_instance.ssh_port), "root@localhost",
            f"bash -c '{command}'"
        ]
        
        try:
            result = subprocess.run(ssh_cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode != 0:
                logger.error(f"Command failed in VM {vm_instance.vm_id}: {result.stderr}")
                raise Exception(f"Command failed: {result.stderr}")
            
            return result.stdout
            
        except subprocess.TimeoutExpired:
            raise TimeoutError(f"Command timed out in VM {vm_instance.vm_id}")
    
    async def write_file_to_vm(self, vm_instance: ProjectVMInstance, file_path: str, content: str):
        """Write file to project VM"""
        
        # Create temporary file locally
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        try:
            # Copy file to VM via SCP
            scp_cmd = [
                "scp", "-o", "StrictHostKeyChecking=no",
                "-P", str(vm_instance.ssh_port),
                tmp_file_path, f"root@localhost:{file_path}"
            ]
            
            subprocess.run(scp_cmd, check=True)
            
        finally:
            # Cleanup temporary file
            Path(tmp_file_path).unlink()
    
    async def read_file_from_vm(self, vm_instance: ProjectVMInstance, file_path: str) -> str:
        """Read file from project VM"""
        
        return await self.execute_command_in_vm(vm_instance, f"cat '{file_path}'")
    
    async def get_vm_status(self, vm_id: UUID) -> Dict[str, Any]:
        """Get comprehensive VM status"""
        
        if vm_id not in self.project_vms:
            return {"error": "VM not found"}
        
        vm_instance = self.project_vms[vm_id]
        
        try:
            # Get VM state from libvirt
            virsh_cmd = ["virsh", "dominfo", f"project-{vm_id}"]
            result = subprocess.run(virsh_cmd, capture_output=True, text=True)
            
            # Parse VM info
            vm_info = {}
            for line in result.stdout.split('\n'):
                if ':' in line:
                    key, value = line.split(':', 1)
                    vm_info[key.strip().lower().replace(' ', '_')] = value.strip()
            
            # Get resource usage
            resource_usage = await self._get_vm_resource_usage(vm_instance)
            
            return {
                "vm_id": str(vm_id),
                "project_id": str(vm_instance.project_id),
                "project_name": vm_instance.project_name,
                "state": vm_instance.state,
                "vm_info": vm_info,
                "resource_usage": resource_usage,
                "ports": {
                    "ssh": vm_instance.ssh_port,
                    "vnc": vm_instance.vnc_port,
                    "web": vm_instance.web_port
                },
                "uptime": (datetime.now(timezone.utc) - vm_instance.created_at).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Failed to get VM status: {e}")
            return {"error": str(e)}
    
    async def _get_vm_resource_usage(self, vm_instance: ProjectVMInstance) -> Dict[str, float]:
        """Get VM resource usage statistics"""
        
        try:
            # Get VM stats from libvirt
            stats_cmd = ["virsh", "domstats", f"project-{vm_instance.vm_id}"]
            result = subprocess.run(stats_cmd, capture_output=True, text=True)
            
            # Parse stats (simplified)
            cpu_usage = 0.0
            memory_usage = 0.0
            
            # In production, properly parse virsh domstats output
            # For now, return mock data
            return {
                "cpu_percent": cpu_usage,
                "memory_percent": memory_usage,
                "disk_usage_gb": 0.0,
                "network_rx_mb": 0.0,
                "network_tx_mb": 0.0
            }
            
        except Exception as e:
            logger.error(f"Failed to get resource usage: {e}")
            return {}
    
    async def suspend_vm(self, vm_id: UUID) -> bool:
        """Suspend project VM"""
        
        if vm_id not in self.project_vms:
            return False
        
        try:
            suspend_cmd = ["virsh", "suspend", f"project-{vm_id}"]
            subprocess.run(suspend_cmd, check=True)
            
            self.project_vms[vm_id].state = ProjectVMState.SUSPENDED
            logger.info(f"Suspended VM {vm_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to suspend VM {vm_id}: {e}")
            return False
    
    async def resume_vm(self, vm_id: UUID) -> bool:
        """Resume project VM"""
        
        if vm_id not in self.project_vms:
            return False
        
        try:
            resume_cmd = ["virsh", "resume", f"project-{vm_id}"]
            subprocess.run(resume_cmd, check=True)
            
            self.project_vms[vm_id].state = ProjectVMState.RUNNING
            logger.info(f"Resumed VM {vm_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to resume VM {vm_id}: {e}")
            return False
    
    async def destroy_vm(self, vm_id: UUID) -> bool:
        """Destroy project VM and cleanup resources"""
        
        if vm_id not in self.project_vms:
            return False
        
        vm_instance = self.project_vms[vm_id]
        
        try:
            # Stop VM
            stop_cmd = ["virsh", "destroy", f"project-{vm_id}"]
            subprocess.run(stop_cmd, check=False)  # Don't fail if already stopped
            
            # Undefine VM
            undefine_cmd = ["virsh", "undefine", f"project-{vm_id}"]
            subprocess.run(undefine_cmd, check=False)
            
            # Cleanup disk
            vm_disk_path = self.vm_storage_path / f"project_{vm_id}.qcow2"
            if vm_disk_path.exists():
                vm_disk_path.unlink()
            
            # Cleanup config
            config_file = self.vm_storage_path / f"project_{vm_id}.xml"
            if config_file.exists():
                config_file.unlink()
            
            # Release ports
            self.port_manager.release_port(vm_instance.ssh_port)
            self.port_manager.release_port(vm_instance.vnc_port)
            self.port_manager.release_port(vm_instance.web_port)
            
            # Remove from tracking
            del self.project_vms[vm_id]
            
            logger.info(f"Destroyed VM {vm_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to destroy VM {vm_id}: {e}")
            return False
    
    async def _cleanup_failed_vm(self, vm_instance: ProjectVMInstance):
        """Cleanup failed VM creation"""
        try:
            await self.destroy_vm(vm_instance.vm_id)
        except:
            pass
    
    async def cleanup_all(self):
        """Cleanup all project VMs"""
        
        vm_ids = list(self.project_vms.keys())
        for vm_id in vm_ids:
            await self.destroy_vm(vm_id)


class ProjectPortManager:
    """Manages port allocation for project VMs"""
    
    def __init__(self, port_range: Tuple[int, int]):
        self.port_start, self.port_end = port_range
        self.allocated_ports: Set[int] = set()
        self.next_port = self.port_start
    
    def allocate_port(self) -> int:
        """Allocate next available port"""
        
        for _ in range(self.port_end - self.port_start):
            port = self.next_port
            self.next_port += 1
            
            if self.next_port > self.port_end:
                self.next_port = self.port_start
            
            if port not in self.allocated_ports:
                self.allocated_ports.add(port)
                return port
        
        raise Exception("No available ports")
    
    def release_port(self, port: int):
        """Release allocated port"""
        self.allocated_ports.discard(port)