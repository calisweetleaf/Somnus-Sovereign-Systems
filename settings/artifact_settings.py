"""
SOMNUS SYSTEMS - Artifact System Settings Manager
On-Demand Modular Execution Ecosystem Configuration

ARCHITECTURE PHILOSOPHY:
- Lightweight orchestrator with on-demand capability activation
- Nothing runs unless explicitly triggered by user intent
- Modular sandboxed execution with intelligent resource management
- Dynamic capability discovery and instantiation
- Zero baseline resource consumption with unlimited scaling potential

This module manages configuration for the revolutionary on-demand artifact system
where capabilities spin up only when needed, maintaining zero baseline overhead
while providing unlimited execution potential.
"""

import asyncio
import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from uuid import UUID, uuid4
from dataclasses import dataclass, field

import aiofiles
import psutil
from pydantic import BaseModel, Field, validator

# Somnus system imports
from schemas.session import SessionID, UserID
from core.memory_core import MemoryManager, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)


# ============================================================================
# CAPABILITY AND ORCHESTRATION MODELS
# ============================================================================

class CapabilityState(str, Enum):
    """On-demand capability lifecycle states"""
    DORMANT = "dormant"           # Capability defined but not active
    INITIALIZING = "initializing" # Spinning up on demand
    ACTIVE = "active"            # Currently running and available
    SCALING = "scaling"          # Adjusting resources
    COOLDOWN = "cooldown"        # Gracefully shutting down
    SUSPENDED = "suspended"      # Temporarily paused
    TERMINATED = "terminated"    # Fully shut down


class CapabilityTrigger(str, Enum):
    """Triggers for on-demand capability activation"""
    USER_INTENT = "user_intent"           # User explicitly requests capability
    AI_INITIATIVE = "ai_initiative"       # AI determines capability needed
    WORKFLOW_DEPENDENCY = "workflow_dependency"  # Required by another capability
    SCHEDULED_ACTIVATION = "scheduled_activation"  # Time-based activation
    RESOURCE_AVAILABILITY = "resource_availability"  # Activated when resources free
    COLLABORATIVE_REQUEST = "collaborative_request"  # Requested by another AI instance


class ResourcePolicy(str, Enum):
    """Resource allocation policies for on-demand scaling"""
    MINIMAL = "minimal"           # Absolute minimum resources
    ADAPTIVE = "adaptive"         # Scale based on workload
    UNLIMITED = "unlimited"       # No resource constraints
    BURST = "burst"              # High resources for short duration
    SUSTAINED = "sustained"       # Consistent resource allocation


@dataclass
class CapabilityDefinition:
    """Complete definition of an on-demand capability"""
    capability_id: str
    name: str
    description: str
    
    # Activation configuration
    activation_triggers: Set[CapabilityTrigger]
    auto_activation: bool = False
    activation_delay_seconds: float = 0.0
    
    # Resource requirements
    resource_policy: ResourcePolicy = ResourcePolicy.ADAPTIVE
    cpu_cores_min: int = 1
    cpu_cores_max: Optional[int] = None
    memory_mb_min: int = 512
    memory_mb_max: Optional[int] = None
    gpu_required: bool = False
    gpu_memory_mb: Optional[int] = None
    
    # Container/execution configuration
    container_image: str = "somnus-artifact:unlimited"
    execution_timeout_seconds: Optional[int] = None
    network_access: bool = True
    file_system_access: bool = True
    
    # Lifecycle management
    idle_timeout_seconds: int = 300  # 5 minutes default
    max_lifetime_seconds: Optional[int] = None
    graceful_shutdown_seconds: int = 30
    
    # Dependencies and integration
    required_capabilities: Set[str] = field(default_factory=set)
    optional_capabilities: Set[str] = field(default_factory=set)
    conflicts_with: Set[str] = field(default_factory=set)
    
    # Security and sandboxing
    security_profile: str = "default"
    sandbox_level: str = "container"  # container, vm, process
    allowed_syscalls: Optional[Set[str]] = None
    blocked_domains: Set[str] = field(default_factory=set)


class ArtifactSystemSettings(BaseModel):
    """Comprehensive configuration for the on-demand artifact system"""
    
    # System identification
    system_id: UUID = Field(default_factory=uuid4)
    version: str = "1.0.0"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Orchestration configuration
    orchestrator_enabled: bool = True
    max_concurrent_capabilities: int = 10
    resource_monitoring_interval_seconds: int = 30
    capability_health_check_interval_seconds: int = 60
    
    # On-demand activation settings
    auto_discovery_enabled: bool = True
    capability_preloading_enabled: bool = False  # Keep dormant by default
    intelligent_scaling_enabled: bool = True
    predictive_activation_enabled: bool = True
    
    # Resource management
    global_resource_limits: Dict[str, Any] = Field(default_factory=lambda: {
        "total_cpu_cores": psutil.cpu_count(),
        "total_memory_mb": int(psutil.virtual_memory().total / (1024 * 1024)),
        "reserved_cpu_cores": 2,  # Keep for system
        "reserved_memory_mb": 2048  # Keep for system
    })
    
    # Capability catalog
    available_capabilities: Dict[str, CapabilityDefinition] = Field(default_factory=dict)
    disabled_capabilities: Set[str] = Field(default_factory=set)
    
    # User and session preferences
    user_capability_preferences: Dict[UserID, Dict[str, Any]] = Field(default_factory=dict)
    session_capability_overrides: Dict[SessionID, Dict[str, Any]] = Field(default_factory=dict)
    
    # Security and compliance
    security_policies: Dict[str, Any] = Field(default_factory=lambda: {
        "require_user_consent": True,
        "log_all_activations": True,
        "enforce_resource_limits": True,
        "enable_capability_isolation": True
    })
    
    # Performance optimization
    optimization_settings: Dict[str, Any] = Field(default_factory=lambda: {
        "enable_capability_caching": True,
        "reuse_warm_containers": True,
        "intelligent_resource_allocation": True,
        "dynamic_load_balancing": True
    })


# ============================================================================
# ON-DEMAND CAPABILITY MANAGER
# ============================================================================

class OnDemandCapabilityManager:
    """Manages on-demand activation and lifecycle of artifact capabilities"""
    
    def __init__(
        self,
        settings: ArtifactSystemSettings,
        memory_manager: Optional[MemoryManager] = None,
        config_path: str = "data/artifact_settings.json"
    ):
        self.settings = settings
        self.memory_manager = memory_manager
        self.config_path = Path(config_path)
        
        # Runtime state tracking
        self.active_capabilities: Dict[str, Dict[str, Any]] = {}
        self.capability_metrics: Dict[str, Dict[str, Any]] = {}
        self.activation_history: List[Dict[str, Any]] = []
        
        # Orchestration components
        self._orchestration_tasks: Dict[str, asyncio.Task] = {}
        self._resource_monitor_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        # Event callbacks
        self.activation_callbacks: List[Callable] = []
        self.deactivation_callbacks: List[Callable] = []
        
        logger.info("On-demand capability manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the capability management system"""
        # Load configuration
        await self._load_settings()
        
        # Initialize default capabilities
        await self._initialize_default_capabilities()
        
        # Start orchestration background tasks
        await self._start_orchestration_tasks()
        
        logger.info(f"Capability manager initialized with {len(self.settings.available_capabilities)} capabilities")
    
    async def _initialize_default_capabilities(self) -> None:
        """Initialize the default set of on-demand capabilities"""
        
        default_capabilities = {
            "unlimited_execution": CapabilityDefinition(
                capability_id="unlimited_execution",
                name="Unlimited Code Execution",
                description="Unrestricted code execution environment",
                activation_triggers={CapabilityTrigger.USER_INTENT, CapabilityTrigger.AI_INITIATIVE},
                resource_policy=ResourcePolicy.UNLIMITED,
                cpu_cores_min=2,
                memory_mb_min=2048,
                container_image="somnus-artifact:unlimited",
                idle_timeout_seconds=600
            ),
            
            "youtube_processing": CapabilityDefinition(
                capability_id="youtube_processing",
                name="YouTube Video Processing",
                description="Download and process YouTube videos with unlimited capabilities",
                activation_triggers={CapabilityTrigger.USER_INTENT},
                resource_policy=ResourcePolicy.BURST,
                cpu_cores_min=2,
                cpu_cores_max=8,
                memory_mb_min=4096,
                memory_mb_max=16384,
                container_image="somnus-artifact:media",
                network_access=True,
                idle_timeout_seconds=300
            ),
            
            "ml_training": CapabilityDefinition(
                capability_id="ml_training",
                name="AI Model Training",
                description="Machine learning model training with GPU acceleration",
                activation_triggers={CapabilityTrigger.USER_INTENT, CapabilityTrigger.AI_INITIATIVE},
                resource_policy=ResourcePolicy.SUSTAINED,
                cpu_cores_min=4,
                memory_mb_min=8192,
                gpu_required=True,
                gpu_memory_mb=8192,
                container_image="somnus-artifact:ml",
                execution_timeout_seconds=None,  # No timeout for training
                idle_timeout_seconds=1800,  # 30 minutes
                max_lifetime_seconds=86400  # 24 hours max
            ),
            
            "browser_research": CapabilityDefinition(
                capability_id="browser_research",
                name="Automated Browser Research",
                description="Visual web research with browser automation",
                activation_triggers={CapabilityTrigger.USER_INTENT, CapabilityTrigger.WORKFLOW_DEPENDENCY},
                resource_policy=ResourcePolicy.ADAPTIVE,
                cpu_cores_min=2,
                memory_mb_min=4096,
                container_image="somnus-artifact:research",
                network_access=True,
                idle_timeout_seconds=900  # 15 minutes
            ),
            
            "git_integration": CapabilityDefinition(
                capability_id="git_integration",
                name="Git Repository Management",
                description="Repository cloning, analysis, and management",
                activation_triggers={CapabilityTrigger.USER_INTENT, CapabilityTrigger.AI_INITIATIVE},
                resource_policy=ResourcePolicy.MINIMAL,
                cpu_cores_min=1,
                memory_mb_min=1024,
                container_image="somnus-artifact:git",
                network_access=True,
                idle_timeout_seconds=600
            ),
            
            "collaborative_intelligence": CapabilityDefinition(
                capability_id="collaborative_intelligence",
                name="Multi-Agent Collaboration",
                description="Enable collaboration with other AI instances",
                activation_triggers={CapabilityTrigger.COLLABORATIVE_REQUEST, CapabilityTrigger.USER_INTENT},
                resource_policy=ResourcePolicy.ADAPTIVE,
                cpu_cores_min=1,
                memory_mb_min=2048,
                container_image="somnus-artifact:collaboration",
                network_access=True,
                idle_timeout_seconds=1200,  # 20 minutes
                required_capabilities={"unlimited_execution"}
            )
        }
        
        # Add default capabilities to settings
        for cap_id, capability in default_capabilities.items():
            if cap_id not in self.settings.available_capabilities:
                self.settings.available_capabilities[cap_id] = capability
    
    async def activate_capability(
        self,
        capability_id: str,
        trigger: CapabilityTrigger,
        user_id: Optional[UserID] = None,
        session_id: Optional[SessionID] = None,
        activation_context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Activate a capability on-demand"""
        
        if capability_id not in self.settings.available_capabilities:
            raise ValueError(f"Unknown capability: {capability_id}")
        
        if capability_id in self.settings.disabled_capabilities:
            raise ValueError(f"Capability disabled: {capability_id}")
        
        capability = self.settings.available_capabilities[capability_id]
        
        # Check if already active
        if capability_id in self.active_capabilities:
            logger.info(f"Capability {capability_id} already active")
            return self.active_capabilities[capability_id]
        
        # Validate trigger
        if trigger not in capability.activation_triggers:
            raise ValueError(f"Invalid trigger {trigger} for capability {capability_id}")
        
        # Check resource availability
        if not await self._check_resource_availability(capability):
            raise RuntimeError(f"Insufficient resources for capability {capability_id}")
        
        # Check dependencies
        await self._ensure_capability_dependencies(capability)
        
        # Create activation record
        activation_record = {
            "capability_id": capability_id,
            "trigger": trigger.value,
            "user_id": user_id,
            "session_id": session_id,
            "context": activation_context or {},
            "activated_at": datetime.now(timezone.utc),
            "state": CapabilityState.INITIALIZING,
            "container_id": None,
            "resource_allocation": await self._calculate_resource_allocation(capability),
            "metrics": {
                "activation_time_ms": 0,
                "requests_handled": 0,
                "errors_encountered": 0,
                "last_activity": datetime.now(timezone.utc)
            }
        }
        
        # Add to active capabilities
        self.active_capabilities[capability_id] = activation_record
        
        # Start activation process
        activation_task = asyncio.create_task(
            self._perform_capability_activation(capability_id, capability, activation_record)
        )
        
        self._orchestration_tasks[f"activate_{capability_id}"] = activation_task
        
        # Log activation
        self.activation_history.append({
            "capability_id": capability_id,
            "trigger": trigger.value,
            "timestamp": datetime.now(timezone.utc),
            "user_id": user_id,
            "session_id": session_id
        })
        
        logger.info(f"Activating capability {capability_id} triggered by {trigger.value}")
        
        return activation_record
    
    async def _perform_capability_activation(
        self,
        capability_id: str,
        capability: CapabilityDefinition,
        activation_record: Dict[str, Any]
    ) -> None:
        """Perform the actual capability activation process"""
        
        start_time = time.time()
        
        try:
            # Apply activation delay if configured
            if capability.activation_delay_seconds > 0:
                await asyncio.sleep(capability.activation_delay_seconds)
            
            # Create container/execution environment
            container_config = await self._create_container_config(capability, activation_record)
            container_id = await self._create_capability_container(container_config)
            
            # Update activation record
            activation_record["container_id"] = container_id
            activation_record["state"] = CapabilityState.ACTIVE
            activation_record["metrics"]["activation_time_ms"] = (time.time() - start_time) * 1000
            
            # Start capability monitoring
            monitor_task = asyncio.create_task(
                self._monitor_capability_lifecycle(capability_id, capability)
            )
            self._orchestration_tasks[f"monitor_{capability_id}"] = monitor_task
            
            # Notify activation callbacks
            for callback in self.activation_callbacks:
                try:
                    await callback(capability_id, activation_record)
                except Exception as e:
                    logger.error(f"Activation callback error: {e}")
            
            logger.info(f"Capability {capability_id} activated successfully in {activation_record['metrics']['activation_time_ms']:.2f}ms")
            
        except Exception as e:
            # Handle activation failure
            activation_record["state"] = CapabilityState.TERMINATED
            activation_record["error"] = str(e)
            
            # Remove from active capabilities
            if capability_id in self.active_capabilities:
                del self.active_capabilities[capability_id]
            
            logger.error(f"Failed to activate capability {capability_id}: {e}")
            raise
    
    async def deactivate_capability(
        self,
        capability_id: str,
        reason: str = "manual",
        graceful: bool = True
    ) -> bool:
        """Deactivate a capability"""
        
        if capability_id not in self.active_capabilities:
            logger.warning(f"Capability {capability_id} not active")
            return False
        
        activation_record = self.active_capabilities[capability_id]
        activation_record["state"] = CapabilityState.COOLDOWN
        
        try:
            # Stop monitoring
            monitor_task_key = f"monitor_{capability_id}"
            if monitor_task_key in self._orchestration_tasks:
                self._orchestration_tasks[monitor_task_key].cancel()
                del self._orchestration_tasks[monitor_task_key]
            
            # Shutdown container
            if activation_record.get("container_id"):
                await self._shutdown_capability_container(
                    activation_record["container_id"],
                    graceful=graceful
                )
            
            # Update final metrics
            activation_record["deactivated_at"] = datetime.now(timezone.utc)
            activation_record["deactivation_reason"] = reason
            activation_record["state"] = CapabilityState.TERMINATED
            
            # Store metrics for analysis
            self.capability_metrics[capability_id] = activation_record["metrics"].copy()
            
            # Remove from active capabilities
            del self.active_capabilities[capability_id]
            
            # Notify deactivation callbacks
            for callback in self.deactivation_callbacks:
                try:
                    await callback(capability_id, activation_record)
                except Exception as e:
                    logger.error(f"Deactivation callback error: {e}")
            
            logger.info(f"Capability {capability_id} deactivated: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating capability {capability_id}: {e}")
            return False
    
    async def _monitor_capability_lifecycle(
        self,
        capability_id: str,
        capability: CapabilityDefinition
    ) -> None:
        """Monitor capability lifecycle and handle automatic deactivation"""
        
        while capability_id in self.active_capabilities:
            try:
                activation_record = self.active_capabilities[capability_id]
                
                # Check idle timeout
                last_activity = activation_record["metrics"]["last_activity"]
                idle_duration = (datetime.now(timezone.utc) - last_activity).total_seconds()
                
                if idle_duration > capability.idle_timeout_seconds:
                    logger.info(f"Capability {capability_id} idle timeout reached ({idle_duration:.1f}s)")
                    await self.deactivate_capability(capability_id, reason="idle_timeout")
                    break
                
                # Check maximum lifetime
                if capability.max_lifetime_seconds:
                    activated_at = activation_record["activated_at"]
                    lifetime = (datetime.now(timezone.utc) - activated_at).total_seconds()
                    
                    if lifetime > capability.max_lifetime_seconds:
                        logger.info(f"Capability {capability_id} maximum lifetime reached ({lifetime:.1f}s)")
                        await self.deactivate_capability(capability_id, reason="max_lifetime")
                        break
                
                # Update health metrics
                await self._update_capability_health_metrics(capability_id)
                
                # Wait before next check
                await asyncio.sleep(self.settings.capability_health_check_interval_seconds)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error monitoring capability {capability_id}: {e}")
                await asyncio.sleep(10)  # Brief pause before retry
    
    async def get_capability_status(self, capability_id: Optional[str] = None) -> Dict[str, Any]:
        """Get status of capabilities"""
        
        if capability_id:
            if capability_id in self.active_capabilities:
                return self.active_capabilities[capability_id]
            else:
                return {
                    "capability_id": capability_id,
                    "state": CapabilityState.DORMANT,
                    "available": capability_id in self.settings.available_capabilities,
                    "enabled": capability_id not in self.settings.disabled_capabilities
                }
        else:
            return {
                "active_capabilities": list(self.active_capabilities.keys()),
                "available_capabilities": list(self.settings.available_capabilities.keys()),
                "disabled_capabilities": list(self.settings.disabled_capabilities),
                "system_resources": await self._get_system_resource_status(),
                "orchestration_health": await self._get_orchestration_health()
            }
    
    async def configure_capability(
        self,
        capability_id: str,
        configuration_updates: Dict[str, Any]
    ) -> bool:
        """Update capability configuration"""
        
        if capability_id not in self.settings.available_capabilities:
            raise ValueError(f"Unknown capability: {capability_id}")
        
        capability = self.settings.available_capabilities[capability_id]
        
        # Apply configuration updates
        for key, value in configuration_updates.items():
            if hasattr(capability, key):
                setattr(capability, key, value)
                logger.info(f"Updated {capability_id}.{key} = {value}")
        
        # Save settings
        await self._save_settings()
        
        # If capability is active, consider restart
        if capability_id in self.active_capabilities:
            logger.info(f"Capability {capability_id} configuration updated while active - restart may be required")
        
        return True
    
    async def _load_settings(self) -> None:
        """Load settings from configuration file"""
        if self.config_path.exists():
            try:
                async with aiofiles.open(self.config_path, 'r') as f:
                    settings_data = json.loads(await f.read())
                
                # Update settings with loaded data
                for key, value in settings_data.items():
                    if hasattr(self.settings, key):
                        setattr(self.settings, key, value)
                
                logger.info(f"Settings loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load settings: {e}")
    
    async def _save_settings(self) -> None:
        """Save current settings to configuration file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Convert settings to JSON-serializable format
            settings_data = self.settings.dict()
            
            async with aiofiles.open(self.config_path, 'w') as f:
                await f.write(json.dumps(settings_data, indent=2, default=str))
            
            logger.debug(f"Settings saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save settings: {e}")
    
    async def shutdown(self) -> None:
        """Graceful shutdown of all capabilities and orchestration"""
        logger.info("Shutting down artifact system capabilities...")
        
        # Cancel all orchestration tasks
        for task_name, task in self._orchestration_tasks.items():
            logger.debug(f"Cancelling task: {task_name}")
            task.cancel()
        
        # Deactivate all active capabilities
        active_caps = list(self.active_capabilities.keys())
        for capability_id in active_caps:
            await self.deactivate_capability(capability_id, reason="system_shutdown")
        
        # Stop background tasks
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        
        # Save final settings
        await self._save_settings()
        
        logger.info("Artifact system capabilities shutdown complete")
    
    # Placeholder methods for container management (to be implemented with actual container runtime)
    async def _create_container_config(self, capability: CapabilityDefinition, activation_record: Dict[str, Any]) -> Dict[str, Any]:
        """Create container configuration for capability"""
        return {
            "image": capability.container_image,
            "cpu_limit": activation_record["resource_allocation"]["cpu_cores"],
            "memory_limit": activation_record["resource_allocation"]["memory_mb"],
            "network_access": capability.network_access,
            "file_system_access": capability.file_system_access,
            "security_profile": capability.security_profile
        }
    
    async def _create_capability_container(self, container_config: Dict[str, Any]) -> str:
        """Create and start capability container"""
        # Placeholder - integrate with actual container runtime
        container_id = f"capability_{uuid4().hex[:8]}"
        logger.debug(f"Created container {container_id} with config: {container_config}")
        return container_id
    
    async def _shutdown_capability_container(self, container_id: str, graceful: bool = True) -> None:
        """Shutdown capability container"""
        # Placeholder - integrate with actual container runtime
        logger.debug(f"Shutting down container {container_id} (graceful={graceful})")
    
    async def _check_resource_availability(self, capability: CapabilityDefinition) -> bool:
        """Check if sufficient resources are available for capability"""
        # Simplified resource check
        return True
    
    async def _calculate_resource_allocation(self, capability: CapabilityDefinition) -> Dict[str, Any]:
        """Calculate optimal resource allocation for capability"""
        return {
            "cpu_cores": capability.cpu_cores_min,
            "memory_mb": capability.memory_mb_min,
            "gpu_required": capability.gpu_required,
            "gpu_memory_mb": capability.gpu_memory_mb
        }
    
    async def _ensure_capability_dependencies(self, capability: CapabilityDefinition) -> None:
        """Ensure required capability dependencies are active"""
        for dep_id in capability.required_capabilities:
            if dep_id not in self.active_capabilities:
                logger.info(f"Activating dependency: {dep_id}")
                await self.activate_capability(dep_id, CapabilityTrigger.WORKFLOW_DEPENDENCY)
    
    async def _update_capability_health_metrics(self, capability_id: str) -> None:
        """Update health metrics for active capability"""
        if capability_id in self.active_capabilities:
            metrics = self.active_capabilities[capability_id]["metrics"]
            # Update metrics based on actual container/system status
            metrics["last_health_check"] = datetime.now(timezone.utc)
    
    async def _get_system_resource_status(self) -> Dict[str, Any]:
        """Get current system resource status"""
        return {
            "cpu_usage_percent": psutil.cpu_percent(),
            "memory_usage_percent": psutil.virtual_memory().percent,
            "disk_usage_percent": psutil.disk_usage('/').percent,
            "active_capabilities_count": len(self.active_capabilities)
        }
    
    async def _get_orchestration_health(self) -> Dict[str, Any]:
        """Get orchestration system health status"""
        return {
            "orchestrator_enabled": self.settings.orchestrator_enabled,
            "active_tasks": len(self._orchestration_tasks),
            "capabilities_active": len(self.active_capabilities),
            "capabilities_available": len(self.settings.available_capabilities),
            "uptime_seconds": time.time() - self.settings.created_at.timestamp()
        }
    
    async def _start_orchestration_tasks(self) -> None:
        """Start background orchestration tasks"""
        # Resource monitoring
        self._resource_monitor_task = asyncio.create_task(self._resource_monitoring_loop())
        
        # Health checking
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info("Orchestration background tasks started")
    
    async def _resource_monitoring_loop(self) -> None:
        """Background task for resource monitoring"""
        while True:
            try:
                await self._monitor_system_resources()
                await asyncio.sleep(self.settings.resource_monitoring_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
                await asyncio.sleep(10)
    
    async def _health_check_loop(self) -> None:
        """Background task for system health checks"""
        while True:
            try:
                await self._perform_system_health_check()
                await asyncio.sleep(self.settings.capability_health_check_interval_seconds)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Health check error: {e}")
                await asyncio.sleep(10)
    
    async def _monitor_system_resources(self) -> None:
        """Monitor system resource usage"""
        # Implement resource monitoring logic
        pass
    
    async def _perform_system_health_check(self) -> None:
        """Perform comprehensive system health check"""
        # Implement health check logic
        pass


# ============================================================================
# SETTINGS FACTORY AND UTILITIES
# ============================================================================

async def create_artifact_system_settings(
    config_path: str = "data/artifact_settings.json",
    memory_manager: Optional[MemoryManager] = None
) -> OnDemandCapabilityManager:
    """Factory function to create and initialize artifact system settings"""
    
    # Create default settings
    settings = ArtifactSystemSettings()
    
    # Create capability manager
    manager = OnDemandCapabilityManager(
        settings=settings,
        memory_manager=memory_manager,
        config_path=config_path
    )
    
    # Initialize the manager
    await manager.initialize()
    
    logger.info("Artifact system settings created and initialized")
    return manager


async def validate_settings_integrity(settings: ArtifactSystemSettings) -> List[str]:
    """Validate settings integrity and return any issues found"""
    issues = []
    
    # Check for conflicting capabilities
    for cap_id, capability in settings.available_capabilities.items():
        for conflict_id in capability.conflicts_with:
            if conflict_id in settings.available_capabilities:
                if conflict_id not in settings.disabled_capabilities:
                    issues.append(f"Conflicting capabilities both enabled: {cap_id} and {conflict_id}")
    
    # Check dependency cycles
    for cap_id, capability in settings.available_capabilities.items():
        if cap_id in capability.required_capabilities:
            issues.append(f"Capability {cap_id} has circular dependency on itself")
    
    # Validate resource limits
    total_min_cpu = sum(
        cap.cpu_cores_min for cap in settings.available_capabilities.values()
        if cap.auto_activation
    )
    
    available_cpu = settings.global_resource_limits["total_cpu_cores"] - settings.global_resource_limits["reserved_cpu_cores"]
    
    if total_min_cpu > available_cpu:
        issues.append(f"Auto-activation capabilities require more CPU than available: {total_min_cpu} > {available_cpu}")
    
    return issues


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        logging.basicConfig(level=logging.INFO)
        
        # Create artifact system settings
        manager = await create_artifact_system_settings()
        
        # Activate a capability on demand
        try:
            activation_result = await manager.activate_capability(
                "unlimited_execution",
                CapabilityTrigger.USER_INTENT,
                user_id="test_user",
                session_id="test_session"
            )
            print(f"Activated capability: {activation_result}")
            
            # Get status
            status = await manager.get_capability_status()
            print(f"System status: {status}")
            
        finally:
            # Cleanup
            await manager.shutdown()
    
    asyncio.run(main())