"""
SOMNUS SYSTEMS - VM Settings Manager
Persistent AI Computing Environment Configuration

ARCHITECTURE PHILOSOPHY:
- Each AI agent gets persistent VM that never resets
- Progressive capability building and tool accumulation
- Resource efficiency through intelligent allocation
- Complete user sovereignty over VM configurations
- On-demand VM creation and lifecycle management

This module manages configuration for the revolutionary persistent VM system
where AI agents maintain state, learn tools, and evolve capabilities over time.
"""

import asyncio
import json
import logging
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

import aiofiles
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(__name__)


# ============================================================================
# VM CONFIGURATION MODELS
# ============================================================================

class VMState(str, Enum):
    """Virtual machine lifecycle states"""
    CREATING = "creating"
    RUNNING = "running"
    SUSPENDED = "suspended"
    SAVING = "saving"
    RESTORING = "restoring"
    BACKING_UP = "backing_up"
    ERROR = "error"
    ARCHIVED = "archived"


class BackupSchedule(str, Enum):
    """VM backup frequency options"""
    DISABLED = "disabled"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_CAPABILITY_CHANGE = "on_capability_change"
    CUSTOM = "custom"


class ResourcePolicy(str, Enum):
    """VM resource allocation policies"""
    CONSERVATIVE = "conservative"    # Minimal resources, slow scale-up
    BALANCED = "balanced"           # Moderate resources, adaptive scaling
    PERFORMANCE = "performance"     # High resources, fast response
    UNLIMITED = "unlimited"         # No artificial limits
    CUSTOM = "custom"              # User-defined allocation


@dataclass
class VMHardwareSpec:
    """VM hardware specifications"""
    vcpus: int = 4
    memory_gb: int = 8
    storage_gb: int = 100
    gpu_enabled: bool = False
    gpu_memory_gb: Optional[int] = None
    
    # Network configuration
    network_enabled: bool = True
    ssh_port: int = 2222
    vnc_port: int = 5900
    
    # Performance tuning
    cpu_priority: str = "normal"  # low, normal, high
    memory_ballooning: bool = True
    disk_cache_mode: str = "writethrough"  # none, writethrough, writeback


@dataclass
class VMPersonality:
    """AI personality configuration for VM behavior"""
    agent_name: str = "AI_Assistant"
    specialization: str = "general"  # general, research, coding, analysis
    creativity_level: float = 0.7
    research_methodology: str = "systematic"  # systematic, exploratory, hybrid
    
    # Workspace preferences
    preferred_ide: str = "vscode"
    preferred_shell: str = "bash"
    preferred_browser: str = "firefox"
    
    # Capability preferences
    auto_install_tools: bool = True
    capability_learning_enabled: bool = True
    cross_session_memory: bool = True


class VMInstanceSettings(BaseModel):
    """Configuration for individual VM instances"""
    # Instance identification
    instance_id: UUID = Field(default_factory=uuid4)
    instance_name: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., description="Owner user ID")
    
    # Hardware configuration
    hardware_spec: VMHardwareSpec = Field(default_factory=VMHardwareSpec)
    resource_policy: ResourcePolicy = ResourcePolicy.BALANCED
    
    # AI personality and behavior
    personality: VMPersonality = Field(default_factory=VMPersonality)
    
    # Lifecycle management
    auto_suspend_minutes: int = Field(default=30, ge=5, le=1440)
    max_idle_hours: int = Field(default=24, ge=1, le=168)
    backup_schedule: BackupSchedule = BackupSchedule.DAILY
    retention_days: int = Field(default=90, ge=7, le=365)
    
    # Storage and persistence
    vm_disk_path: Optional[str] = None
    snapshot_path: Optional[str] = None
    backup_path: Optional[str] = None
    
    # Capability tracking
    installed_tools: List[str] = Field(default_factory=list)
    learned_capabilities: List[str] = Field(default_factory=list)
    research_bookmarks: List[str] = Field(default_factory=list)
    custom_workflows: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    total_uptime_hours: float = 0.0
    efficiency_rating: float = 1.0
    capability_acquisition_rate: float = 0.0
    
    # Creation and modification tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_backup: Optional[datetime] = None


class VMPoolSettings(BaseModel):
    """Global VM pool management configuration"""
    # System identification
    system_id: UUID = Field(default_factory=uuid4)
    version: str = "1.0.0"
    
    # Pool management
    max_concurrent_vms: int = Field(default=10, ge=1, le=100)
    max_vms_per_user: int = Field(default=5, ge=1, le=20)
    auto_scaling_enabled: bool = True
    vm_warmup_enabled: bool = False  # Keep dormant for efficiency
    
    # Resource management
    global_resource_limits: Dict[str, Any] = Field(default_factory=lambda: {
        "total_cpu_cores": psutil.cpu_count(),
        "total_memory_gb": int(psutil.virtual_memory().total / (1024**3)),
        "total_storage_gb": 1000,  # Default 1TB allocation
        "reserved_cpu_cores": 2,    # Reserve for host system
        "reserved_memory_gb": 4     # Reserve for host system
    })
    
    # VM templates and base images
    base_image_path: str = "/data/vm_templates/base_ai_computer.qcow2"
    template_directory: str = "/data/vm_templates"
    instance_directory: str = "/data/vm_instances"
    backup_directory: str = "/data/vm_backups"
    
    # Default specifications for new VMs
    default_hardware_spec: VMHardwareSpec = Field(default_factory=VMHardwareSpec)
    default_personality: VMPersonality = Field(default_factory=VMPersonality)
    
    # Automation settings
    auto_backup_enabled: bool = True
    auto_cleanup_enabled: bool = True
    capability_sync_enabled: bool = True
    cross_vm_learning: bool = False  # Future feature
    
    # Performance optimization
    vm_balancing_enabled: bool = True
    resource_monitoring_interval: int = 30
    health_check_interval: int = 60
    
    # Security settings
    ssh_key_management: bool = True
    network_isolation: bool = True
    snapshot_encryption: bool = False  # Future feature
    
    # User preferences
    user_vm_preferences: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    vm_sharing_enabled: bool = False  # Future collaboration feature


# ============================================================================
# VM SETTINGS MANAGER
# ============================================================================

class VMSettingsManager:
    """Manages VM system configuration and on-demand lifecycle"""
    
    def __init__(
        self,
        pool_settings: VMPoolSettings,
        config_path: str = "data/vm_settings.json"
    ):
        self.pool_settings = pool_settings
        self.config_path = Path(config_path)
        
        # Runtime state tracking
        self.active_vms: Dict[UUID, VMInstanceSettings] = {}
        self.vm_metrics: Dict[UUID, Dict[str, Any]] = {}
        self.resource_usage: Dict[str, float] = {}
        
        # Background tasks
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._resource_monitor_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info("VM Settings Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the VM settings management system"""
        # Load configuration
        await self._load_settings()
        
        # Initialize directories
        await self._initialize_directories()
        
        # Start background monitoring
        await self._start_monitoring_tasks()
        
        logger.info(f"VM Settings Manager initialized with capacity for {self.pool_settings.max_concurrent_vms} VMs")
    
    async def create_vm_instance(
        self,
        user_id: str,
        instance_name: str,
        hardware_spec: Optional[VMHardwareSpec] = None,
        personality: Optional[VMPersonality] = None
    ) -> VMInstanceSettings:
        """Create new VM instance configuration"""
        
        # Check resource availability
        if not await self._check_resource_availability(hardware_spec):
            raise ValueError("Insufficient resources for VM creation")
        
        # Check user limits
        user_vm_count = sum(1 for vm in self.active_vms.values() if vm.user_id == user_id)
        if user_vm_count >= self.pool_settings.max_vms_per_user:
            raise ValueError(f"User VM limit reached ({self.pool_settings.max_vms_per_user})")
        
        # Create VM configuration
        vm_settings = VMInstanceSettings(
            instance_name=instance_name,
            user_id=user_id,
            hardware_spec=hardware_spec or self.pool_settings.default_hardware_spec,
            personality=personality or self.pool_settings.default_personality
        )
        
        # Set up storage paths
        vm_settings.vm_disk_path = str(
            Path(self.pool_settings.instance_directory) / f"{vm_settings.instance_id}.qcow2"
        )
        vm_settings.snapshot_path = str(
            Path(self.pool_settings.instance_directory) / f"{vm_settings.instance_id}_snapshots"
        )
        vm_settings.backup_path = str(
            Path(self.pool_settings.backup_directory) / f"{vm_settings.instance_id}"
        )
        
        # Register VM
        self.active_vms[vm_settings.instance_id] = vm_settings
        
        # Initialize metrics tracking
        self.vm_metrics[vm_settings.instance_id] = {
            "creation_time": datetime.now(timezone.utc),
            "last_heartbeat": None,
            "resource_usage": {"cpu": 0.0, "memory": 0.0, "disk": 0.0},
            "capability_changes": []
        }
        
        # Save configuration
        await self._save_settings()
        
        logger.info(f"Created VM instance {vm_settings.instance_name} for user {user_id}")
        return vm_settings
    
    async def update_vm_settings(
        self,
        instance_id: UUID,
        updates: Dict[str, Any]
    ) -> bool:
        """Update VM instance settings"""
        
        if instance_id not in self.active_vms:
            return False
        
        vm_settings = self.active_vms[instance_id]
        
        # Apply updates with validation
        for key, value in updates.items():
            if hasattr(vm_settings, key):
                # Special handling for complex objects
                if key == "hardware_spec" and isinstance(value, dict):
                    for spec_key, spec_value in value.items():
                        if hasattr(vm_settings.hardware_spec, spec_key):
                            setattr(vm_settings.hardware_spec, spec_key, spec_value)
                elif key == "personality" and isinstance(value, dict):
                    for pers_key, pers_value in value.items():
                        if hasattr(vm_settings.personality, pers_key):
                            setattr(vm_settings.personality, pers_key, pers_value)
                else:
                    setattr(vm_settings, key, value)
        
        vm_settings.last_modified = datetime.now(timezone.utc)
        
        # Save updated configuration
        await self._save_settings()
        
        logger.info(f"Updated VM settings for instance {instance_id}")
        return True
    
    async def track_capability_acquisition(
        self,
        instance_id: UUID,
        capability: str,
        capability_type: str = "tool"
    ) -> bool:
        """Track when AI acquires new capabilities"""
        
        if instance_id not in self.active_vms:
            return False
        
        vm_settings = self.active_vms[instance_id]
        
        # Add to appropriate tracking list
        if capability_type == "tool" and capability not in vm_settings.installed_tools:
            vm_settings.installed_tools.append(capability)
        elif capability_type == "skill" and capability not in vm_settings.learned_capabilities:
            vm_settings.learned_capabilities.append(capability)
        
        # Update metrics
        if instance_id in self.vm_metrics:
            self.vm_metrics[instance_id]["capability_changes"].append({
                "capability": capability,
                "type": capability_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_capabilities": len(vm_settings.installed_tools) + len(vm_settings.learned_capabilities)
            })
        
        # Calculate efficiency improvement
        total_capabilities = len(vm_settings.installed_tools) + len(vm_settings.learned_capabilities)
        vm_settings.efficiency_rating = 1.0 + (total_capabilities * 0.1)
        
        # Update acquisition rate
        if vm_settings.total_uptime_hours > 0:
            vm_settings.capability_acquisition_rate = total_capabilities / vm_settings.total_uptime_hours
        
        await self._save_settings()
        
        logger.info(f"VM {instance_id} acquired capability: {capability} ({capability_type})")
        return True
    
    async def get_vm_status(self, instance_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get status of VM instances"""
        
        if instance_id:
            if instance_id in self.active_vms:
                vm_settings = self.active_vms[instance_id]
                metrics = self.vm_metrics.get(instance_id, {})
                
                return {
                    "instance_id": str(instance_id),
                    "instance_name": vm_settings.instance_name,
                    "user_id": vm_settings.user_id,
                    "resource_policy": vm_settings.resource_policy,
                    "hardware_spec": vm_settings.hardware_spec.__dict__,
                    "personality": vm_settings.personality.__dict__,
                    "capabilities": {
                        "installed_tools": vm_settings.installed_tools,
                        "learned_capabilities": vm_settings.learned_capabilities,
                        "efficiency_rating": vm_settings.efficiency_rating,
                        "acquisition_rate": vm_settings.capability_acquisition_rate
                    },
                    "metrics": metrics,
                    "uptime_hours": vm_settings.total_uptime_hours,
                    "created_at": vm_settings.created_at.isoformat(),
                    "last_modified": vm_settings.last_modified.isoformat()
                }
            else:
                return {"error": f"VM instance {instance_id} not found"}
        
        # Return pool status
        return {
            "pool_status": {
                "active_vms": len(self.active_vms),
                "max_concurrent": self.pool_settings.max_concurrent_vms,
                "total_capacity": self.pool_settings.global_resource_limits,
                "current_usage": self.resource_usage
            },
            "active_instances": [
                {
                    "instance_id": str(vm_id),
                    "instance_name": vm_settings.instance_name,
                    "user_id": vm_settings.user_id,
                    "efficiency_rating": vm_settings.efficiency_rating,
                    "total_capabilities": len(vm_settings.installed_tools) + len(vm_settings.learned_capabilities)
                }
                for vm_id, vm_settings in self.active_vms.items()
            ]
        }
    
    async def configure_vm_pool(self, pool_updates: Dict[str, Any]) -> bool:
        """Update VM pool configuration"""
        
        # Apply updates to pool settings
        for key, value in pool_updates.items():
            if hasattr(self.pool_settings, key):
                if key == "global_resource_limits" and isinstance(value, dict):
                    self.pool_settings.global_resource_limits.update(value)
                elif key == "default_hardware_spec" and isinstance(value, dict):
                    for spec_key, spec_value in value.items():
                        if hasattr(self.pool_settings.default_hardware_spec, spec_key):
                            setattr(self.pool_settings.default_hardware_spec, spec_key, spec_value)
                else:
                    setattr(self.pool_settings, key, value)
        
        # Save updated configuration
        await self._save_settings()
        
        logger.info("VM pool configuration updated")
        return True
    
    async def _check_resource_availability(self, hardware_spec: Optional[VMHardwareSpec]) -> bool:
        """Check if sufficient resources are available for VM creation"""
        
        if not hardware_spec:
            hardware_spec = self.pool_settings.default_hardware_spec
        
        # Calculate current resource usage
        current_cpu = sum(
            vm.hardware_spec.vcpus for vm in self.active_vms.values()
        )
        current_memory = sum(
            vm.hardware_spec.memory_gb for vm in self.active_vms.values()
        )
        current_storage = sum(
            vm.hardware_spec.storage_gb for vm in self.active_vms.values()
        )
        
        # Check against limits
        available_cpu = (
            self.pool_settings.global_resource_limits["total_cpu_cores"] - 
            self.pool_settings.global_resource_limits["reserved_cpu_cores"] - 
            current_cpu
        )
        available_memory = (
            self.pool_settings.global_resource_limits["total_memory_gb"] - 
            self.pool_settings.global_resource_limits["reserved_memory_gb"] - 
            current_memory
        )
        available_storage = (
            self.pool_settings.global_resource_limits["total_storage_gb"] - 
            current_storage
        )
        
        return (
            hardware_spec.vcpus <= available_cpu and
            hardware_spec.memory_gb <= available_memory and
            hardware_spec.storage_gb <= available_storage
        )
    
    async def _initialize_directories(self):
        """Initialize VM storage directories"""
        directories = [
            self.pool_settings.template_directory,
            self.pool_settings.instance_directory,
            self.pool_settings.backup_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        if self.pool_settings.resource_monitoring_interval > 0:
            self._resource_monitor_task = asyncio.create_task(
                self._resource_monitoring_loop()
            )
        
        if self.pool_settings.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
    
    async def _resource_monitoring_loop(self):
        """Background resource monitoring"""
        while True:
            try:
                # Update resource usage metrics
                self.resource_usage = {
                    "cpu_usage_percent": sum(vm.hardware_spec.vcpus for vm in self.active_vms.values()) / 
                                       self.pool_settings.global_resource_limits["total_cpu_cores"] * 100,
                    "memory_usage_percent": sum(vm.hardware_spec.memory_gb for vm in self.active_vms.values()) / 
                                          self.pool_settings.global_resource_limits["total_memory_gb"] * 100,
                    "storage_usage_percent": sum(vm.hardware_spec.storage_gb for vm in self.active_vms.values()) / 
                                           self.pool_settings.global_resource_limits["total_storage_gb"] * 100
                }
                
                await asyncio.sleep(self.pool_settings.resource_monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _health_check_loop(self):
        """Background health monitoring for VMs"""
        while True:
            try:
                for vm_id, vm_settings in self.active_vms.items():
                    # Update uptime tracking
                    if vm_id in self.vm_metrics:
                        creation_time = self.vm_metrics[vm_id]["creation_time"]
                        current_uptime = (datetime.now(timezone.utc) - creation_time).total_seconds() / 3600
                        vm_settings.total_uptime_hours = current_uptime
                
                await asyncio.sleep(self.pool_settings.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _load_settings(self) -> None:
        """Load settings from configuration file"""
        if self.config_path.exists():
            try:
                async with aiofiles.open(self.config_path, 'r') as f:
                    settings_data = json.loads(await f.read())
                
                # Update pool settings
                if "pool_settings" in settings_data:
                    for key, value in settings_data["pool_settings"].items():
                        if hasattr(self.pool_settings, key):
                            setattr(self.pool_settings, key, value)
                
                # Load active VMs
                if "active_vms" in settings_data:
                    for vm_data in settings_data["active_vms"]:
                        vm_settings = VMInstanceSettings(**vm_data)
                        self.active_vms[vm_settings.instance_id] = vm_settings
                
                logger.info(f"VM settings loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load VM settings: {e}")
    
    async def _save_settings(self) -> None:
        """Save current settings to configuration file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            settings_data = {
                "pool_settings": self.pool_settings.dict(),
                "active_vms": [vm.dict() for vm in self.active_vms.values()],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            async with aiofiles.open(self.config_path, 'w') as f:
                await f.write(json.dumps(settings_data, indent=2, default=str))
            
            logger.debug(f"VM settings saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save VM settings: {e}")
    
    async def shutdown(self) -> None:
        """Graceful shutdown of VM settings management"""
        logger.info("Shutting down VM settings management...")
        
        # Cancel background tasks
        for task_name, task in self._monitoring_tasks.items():
            logger.debug(f"Cancelling task: {task_name}")
            task.cancel()
        
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        
        # Save final settings
        await self._save_settings()
        
        logger.info("VM settings management shutdown complete")


# ============================================================================
# FACTORY AND INITIALIZATION
# ============================================================================

async def create_vm_settings_manager(
    max_concurrent_vms: int = 10,
    base_image_path: str = "/data/vm_templates/base_ai_computer.qcow2"
) -> VMSettingsManager:
    """Create and initialize VM settings manager"""
    
    pool_settings = VMPoolSettings(
        max_concurrent_vms=max_concurrent_vms,
        base_image_path=base_image_path
    )
    
    manager = VMSettingsManager(pool_settings)
    await manager.initialize()
    
    logger.info("VM settings manager created and initialized")
    return manager


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Create VM settings manager
        manager = await create_vm_settings_manager()
        
        try:
            # Create test VM instance
            vm_settings = await manager.create_vm_instance(
                user_id="test_user",
                instance_name="Test AI Assistant",
                hardware_spec=VMHardwareSpec(vcpus=2, memory_gb=4, storage_gb=50)
            )
            print(f"Created VM instance: {vm_settings.instance_name}")
            
            # Track capability acquisition
            await manager.track_capability_acquisition(
                vm_settings.instance_id,
                "python_development",
                "skill"
            )
            
            # Get status
            status = await manager.get_vm_status(vm_settings.instance_id)
            print(f"VM Status: {status}")
            
        finally:
            # Cleanup
            await manager.shutdown()
    
    asyncio.run(main())