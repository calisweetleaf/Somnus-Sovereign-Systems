"""
SOMNUS SYSTEMS - Global Configuration Router
Centralized System Configuration Management and Routing

ARCHITECTURE PHILOSOPHY:
- Single entry point for all technical subsystem configuration
- Intelligent routing based on configuration domain and context
- Real-time configuration validation and conflict resolution
- Cross-subsystem coordination through event integration
- Complete audit trail for all configuration changes

This router handles all technical "knobs and dials" of the Somnus system,
from VM hardware allocation to research methodology parameters, while
maintaining clean separation from user personalization concerns.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Union, Tuple, Type
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
import copy

from events import (
    get_event_bus, VMEvent, MemoryEvent, ResearchEvent, ArtifactEvent,
    EventPriority, publish_vm_configuration_changed
)
from capability_broker import get_capability_broker, CapabilityBroker
from projection_registry import apply_projection_mapping

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION TYPES AND ENUMS
# ============================================================================

class ConfigurationDomain(str, Enum):
    """Configuration domains mapped to subsystems"""
    VM_SYSTEM = "vm_system"
    ARTIFACT_SYSTEM = "artifact_system"
    RESEARCH_SYSTEM = "research_system"
    MEMORY_SYSTEM = "memory_system"
    GLOBAL_SYSTEM = "global_system"

class ConfigurationScope(str, Enum):
    """Scope of configuration changes"""
    SYSTEM_WIDE = "system_wide"      # Affects entire system
    SUBSYSTEM = "subsystem"          # Affects single subsystem
    INSTANCE = "instance"            # Affects single instance
    SESSION = "session"              # Session-specific changes
    USER = "user"                    # User-specific changes

class ChangeType(str, Enum):
    """Types of configuration changes"""
    CREATE = "create"
    UPDATE = "update"
    DELETE = "delete"
    RESET = "reset"
    ENABLE = "enable"
    DISABLE = "disable"

@dataclass
class ConfigurationChange:
    """Represents a configuration change request"""
    change_id: str = field(default_factory=lambda: str(uuid4()))
    domain: ConfigurationDomain = ConfigurationDomain.GLOBAL_SYSTEM
    scope: ConfigurationScope = ConfigurationScope.SUBSYSTEM
    change_type: ChangeType = ChangeType.UPDATE
    target_path: str = ""
    new_value: Any = None
    old_value: Any = None
    context: Dict[str, Any] = field(default_factory=dict)
    requester: str = ""
    reason: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied: bool = False
    error: Optional[str] = None

@dataclass
class ConfigurationValidation:
    """Result of configuration validation"""
    is_valid: bool = False
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    suggested_alternatives: Dict[str, Any] = field(default_factory=dict)
    estimated_impact: str = "unknown"  # low, medium, high, critical


# ============================================================================
# SUBSYSTEM MANAGER INTERFACES
# ============================================================================

class SubsystemManager:
    """Base interface for subsystem configuration management"""
    
    async def validate_configuration(self, config_change: ConfigurationChange) -> ConfigurationValidation:
        """Validate proposed configuration change"""
        raise NotImplementedError
    
    async def apply_configuration(self, config_change: ConfigurationChange) -> bool:
        """Apply configuration change"""
        raise NotImplementedError
    
    async def get_current_configuration(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Get current configuration state"""
        raise NotImplementedError
    
    async def get_configuration_schema(self) -> Dict[str, Any]:
        """Get configuration schema and validation rules"""
        raise NotImplementedError
    
    async def reset_to_defaults(self, path: Optional[str] = None) -> bool:
        """Reset configuration to default values"""
        raise NotImplementedError

class VMSystemManager(SubsystemManager):
    """VM system configuration manager interface"""
    
    def __init__(self):
        self.vm_settings_manager = None  # Would be injected
    
    async def validate_configuration(self, config_change: ConfigurationChange) -> ConfigurationValidation:
        """Validate VM configuration changes"""
        validation = ConfigurationValidation()
        
        # Example validation logic for VM settings
        if "hardware_spec" in config_change.target_path:
            if isinstance(config_change.new_value, dict):
                # Validate hardware specifications
                if "memory_gb" in config_change.new_value:
                    memory_gb = config_change.new_value["memory_gb"]
                    if memory_gb < 1 or memory_gb > 128:
                        validation.errors.append("Memory allocation must be between 1GB and 128GB")
                
                if "vcpus" in config_change.new_value:
                    vcpus = config_change.new_value["vcpus"]
                    if vcpus < 1 or vcpus > 32:
                        validation.errors.append("vCPU count must be between 1 and 32")
        
        validation.is_valid = len(validation.errors) == 0
        return validation
    
    async def apply_configuration(self, config_change: ConfigurationChange) -> bool:
        """Apply VM configuration change"""
        try:
            # This would integrate with actual vm_settings.py
            logger.info(f"Applying VM configuration: {config_change.target_path} = {config_change.new_value}")
            
            # Simulate configuration application
            if "personality" in config_change.target_path:
                # Update VM personality
                pass
            elif "hardware_spec" in config_change.target_path:
                # Update hardware allocation
                pass
            
            # Publish VM configuration change event
            await publish_vm_configuration_changed(
                vm_id=config_change.context.get("vm_id", "unknown"),
                vm_name=config_change.context.get("vm_name", "unknown"),
                hardware_changes=config_change.new_value if "hardware" in config_change.target_path else None,
                personality_changes=config_change.new_value if "personality" in config_change.target_path else None,
                priority=EventPriority.HIGH,
                user_id=config_change.context.get("user_id")
            )
            
            return True
        except Exception as e:
            logger.error(f"Failed to apply VM configuration: {e}")
            return False
    
    async def get_current_configuration(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Get current VM configuration"""
        # This would integrate with actual vm_settings.py
        return {
            "max_concurrent_vms": 10,
            "default_hardware_spec": {
                "vcpus": 2,
                "memory_gb": 4,
                "storage_gb": 50
            },
            "default_personality": {
                "research_methodology": "hybrid",
                "communication_style": "adaptive"
            }
        }
    
    async def get_configuration_schema(self) -> Dict[str, Any]:
        """Get VM configuration schema"""
        return {
            "max_concurrent_vms": {"type": "integer", "min": 1, "max": 100},
            "default_hardware_spec": {
                "vcpus": {"type": "integer", "min": 1, "max": 32},
                "memory_gb": {"type": "integer", "min": 1, "max": 128},
                "storage_gb": {"type": "integer", "min": 10, "max": 1000}
            },
            "default_personality": {
                "research_methodology": {"type": "enum", "values": ["systematic", "hybrid", "creative"]},
                "communication_style": {"type": "enum", "values": ["technical", "adaptive", "casual"]}
            }
        }

class ArtifactSystemManager(SubsystemManager):
    """Artifact system configuration manager"""
    
    async def validate_configuration(self, config_change: ConfigurationChange) -> ConfigurationValidation:
        """Validate artifact system configuration"""
        validation = ConfigurationValidation()
        
        if "max_concurrent_capabilities" in config_change.target_path:
            if not isinstance(config_change.new_value, int) or config_change.new_value < 1:
                validation.errors.append("max_concurrent_capabilities must be a positive integer")
        
        if "global_resource_limits" in config_change.target_path:
            if not isinstance(config_change.new_value, dict):
                validation.errors.append("global_resource_limits must be a dictionary")
        
        validation.is_valid = len(validation.errors) == 0
        return validation
    
    async def apply_configuration(self, config_change: ConfigurationChange) -> bool:
        """Apply artifact system configuration"""
        try:
            logger.info(f"Applying artifact configuration: {config_change.target_path} = {config_change.new_value}")
            
            # This would integrate with actual artifact_settings.py
            # For now, just log the change
            
            return True
        except Exception as e:
            logger.error(f"Failed to apply artifact configuration: {e}")
            return False
    
    async def get_current_configuration(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Get current artifact system configuration"""
        return {
            "orchestrator_enabled": True,
            "max_concurrent_capabilities": 10,
            "global_resource_limits": {
                "total_cpu_cores": 8,
                "total_memory_mb": 16384
            }
        }

class ResearchSystemManager(SubsystemManager):
    """Research system configuration manager"""
    
    async def validate_configuration(self, config_change: ConfigurationChange) -> ConfigurationValidation:
        """Validate research system configuration"""
        validation = ConfigurationValidation()
        
        if "research_mode" in config_change.target_path:
            valid_modes = ["speed", "balanced", "comprehensive", "exhaustive"]
            if config_change.new_value not in valid_modes:
                validation.errors.append(f"research_mode must be one of: {valid_modes}")
        
        if "max_concurrent_searches" in config_change.target_path:
            if not isinstance(config_change.new_value, int) or config_change.new_value < 1:
                validation.errors.append("max_concurrent_searches must be a positive integer")
        
        validation.is_valid = len(validation.errors) == 0
        return validation
    
    async def apply_configuration(self, config_change: ConfigurationChange) -> bool:
        """Apply research system configuration"""
        try:
            logger.info(f"Applying research configuration: {config_change.target_path} = {config_change.new_value}")
            
            # This would integrate with actual research_settings_subsystem.py
            
            return True
        except Exception as e:
            logger.error(f"Failed to apply research configuration: {e}")
            return False
    
    async def get_current_configuration(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Get current research system configuration"""
        return {
            "research_mode": "balanced",
            "source_trust_level": "curated",
            "max_concurrent_searches": 5,
            "search_timeout_seconds": 30
        }

class MemorySystemManager(SubsystemManager):
    """Memory system configuration manager"""
    
    async def validate_configuration(self, config_change: ConfigurationChange) -> ConfigurationValidation:
        """Validate memory system configuration"""
        validation = ConfigurationValidation()
        
        if "retention_strategy" in config_change.target_path:
            valid_strategies = ["simple", "importance_based", "adaptive", "hybrid"]
            if config_change.new_value not in valid_strategies:
                validation.errors.append(f"retention_strategy must be one of: {valid_strategies}")
        
        if "max_memories_per_user" in config_change.target_path:
            if not isinstance(config_change.new_value, int) or config_change.new_value < 1:
                validation.errors.append("max_memories_per_user must be a positive integer")
        
        validation.is_valid = len(validation.errors) == 0
        return validation
    
    async def apply_configuration(self, config_change: ConfigurationChange) -> bool:
        """Apply memory system configuration"""
        try:
            logger.info(f"Applying memory configuration: {config_change.target_path} = {config_change.new_value}")
            
            # This would integrate with actual memory_system_settings.py
            
            return True
        except Exception as e:
            logger.error(f"Failed to apply memory configuration: {e}")
            return False
    
    async def get_current_configuration(self, path: Optional[str] = None) -> Dict[str, Any]:
        """Get current memory system configuration"""
        return {
            "retention_strategy": "adaptive",
            "max_memories_per_user": 50000,
            "embedding_model": "sentence_transformer",
            "enable_compression": True
        }


# ============================================================================
# GLOBAL CONFIGURATION ROUTER
# ============================================================================

class GlobalConfigurationRouter:
    """Central router for all system configuration management"""
    
    def __init__(self):
        # Subsystem managers
        self.subsystem_managers: Dict[ConfigurationDomain, SubsystemManager] = {
            ConfigurationDomain.VM_SYSTEM: VMSystemManager(),
            ConfigurationDomain.ARTIFACT_SYSTEM: ArtifactSystemManager(),
            ConfigurationDomain.RESEARCH_SYSTEM: ResearchSystemManager(),
            ConfigurationDomain.MEMORY_SYSTEM: MemorySystemManager()
        }
        
        # Configuration state tracking
        self.configuration_history: List[ConfigurationChange] = []
        self.active_configurations: Dict[ConfigurationDomain, Dict[str, Any]] = {}
        
        # Validation and conflict resolution
        self.validation_cache: Dict[str, ConfigurationValidation] = {}
        self.conflict_resolution_rules: Dict[str, Any] = {}
        
        # Event integration
        self.event_bus = get_event_bus()
        self.capability_broker = get_capability_broker()
        
        # Performance tracking
        self.operation_metrics: Dict[str, Dict[str, Any]] = {}
        
        logger.info("Global Configuration Router initialized")
    
    async def initialize(self) -> None:
        """Initialize the configuration router"""
        # Load current configurations from all subsystems
        for domain, manager in self.subsystem_managers.items():
            try:
                config = await manager.get_current_configuration()
                self.active_configurations[domain] = config
            except Exception as e:
                logger.error(f"Failed to load configuration for {domain}: {e}")
                self.active_configurations[domain] = {}
        
        # Subscribe to relevant events for cross-subsystem coordination
        self.event_bus.subscribe(VMEvent, self._handle_vm_event, priority=EventPriority.HIGH)
        self.event_bus.subscribe(MemoryEvent, self._handle_memory_event)
        self.event_bus.subscribe(ResearchEvent, self._handle_research_event)
        self.event_bus.subscribe(ArtifactEvent, self._handle_artifact_event)
        
        logger.info("Global Configuration Router initialized and event subscriptions active")
    
    async def apply_configuration_change(
        self,
        domain: ConfigurationDomain,
        target_path: str,
        new_value: Any,
        requester: str = "system",
        reason: str = "User requested",
        context: Optional[Dict[str, Any]] = None,
        validate_only: bool = False
    ) -> Tuple[bool, str, Optional[ConfigurationChange]]:
        """Apply a configuration change to the specified domain"""
        
        # Create configuration change object
        change = ConfigurationChange(
            domain=domain,
            scope=self._determine_scope(domain, target_path),
            change_type=ChangeType.UPDATE,
            target_path=target_path,
            new_value=new_value,
            old_value=self._get_current_value(domain, target_path),
            context=context or {},
            requester=requester,
            reason=reason
        )
        
        # Validate the change
        validation_result = await self._validate_configuration_change(change)
        
        if not validation_result.is_valid:
            error_msg = f"Configuration validation failed: {'; '.join(validation_result.errors)}"
            change.error = error_msg
            self.configuration_history.append(change)
            return False, error_msg, change
        
        # If validation-only mode, return success without applying
        if validate_only:
            return True, "Configuration validation passed", change
        
        # Check for conflicts with other subsystems
        conflict_result = await self._check_cross_subsystem_conflicts(change)
        if not conflict_result[0]:
            change.error = conflict_result[1]
            self.configuration_history.append(change)
            return False, conflict_result[1], change
        
        # Apply the configuration change
        manager = self.subsystem_managers.get(domain)
        if not manager:
            error_msg = f"No manager found for domain: {domain}"
            change.error = error_msg
            self.configuration_history.append(change)
            return False, error_msg, change
        
        try:
            success = await manager.apply_configuration(change)
            
            if success:
                # Update active configuration state
                self._update_active_configuration(domain, target_path, new_value)
                
                # Mark change as applied
                change.applied = True
                
                # Record metrics
                self._record_operation_metrics(change)
                
                # Trigger cross-subsystem coordination if needed
                await self._coordinate_cross_subsystem_changes(change)
                
                success_msg = f"Configuration applied successfully: {domain}.{target_path} = {new_value}"
                logger.info(success_msg)
                
                self.configuration_history.append(change)
                return True, success_msg, change
            else:
                error_msg = f"Failed to apply configuration: {domain}.{target_path}"
                change.error = error_msg
                self.configuration_history.append(change)
                return False, error_msg, change
                
        except Exception as e:
            error_msg = f"Exception applying configuration: {e}"
            change.error = error_msg
            self.configuration_history.append(change)
            logger.error(error_msg)
            return False, error_msg, change
    
    async def get_configuration(
        self,
        domain: ConfigurationDomain,
        path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get current configuration for domain or specific path"""
        
        manager = self.subsystem_managers.get(domain)
        if not manager:
            return {}
        
        try:
            config = await manager.get_current_configuration(path)
            return config
        except Exception as e:
            logger.error(f"Failed to get configuration for {domain}: {e}")
            return self.active_configurations.get(domain, {})
    
    async def get_configuration_schema(self, domain: ConfigurationDomain) -> Dict[str, Any]:
        """Get configuration schema for validation and UI generation"""
        
        manager = self.subsystem_managers.get(domain)
        if not manager:
            return {}
        
        try:
            schema = await manager.get_configuration_schema()
            return schema
        except Exception as e:
            logger.error(f"Failed to get schema for {domain}: {e}")
            return {}
    
    async def validate_configuration(
        self,
        domain: ConfigurationDomain,
        target_path: str,
        new_value: Any
    ) -> ConfigurationValidation:
        """Validate a configuration change without applying it"""
        
        change = ConfigurationChange(
            domain=domain,
            target_path=target_path,
            new_value=new_value,
            old_value=self._get_current_value(domain, target_path)
        )
        
        return await self._validate_configuration_change(change)
    
    async def reset_configuration(
        self,
        domain: ConfigurationDomain,
        path: Optional[str] = None,
        requester: str = "system"
    ) -> Tuple[bool, str]:
        """Reset configuration to default values"""
        
        manager = self.subsystem_managers.get(domain)
        if not manager:
            return False, f"No manager found for domain: {domain}"
        
        try:
            success = await manager.reset_to_defaults(path)
            
            if success:
                # Reload configuration after reset
                config = await manager.get_current_configuration()
                self.active_configurations[domain] = config
                
                # Record the reset operation
                reset_change = ConfigurationChange(
                    domain=domain,
                    change_type=ChangeType.RESET,
                    target_path=path or "all",
                    requester=requester,
                    reason="Configuration reset to defaults",
                    applied=True
                )
                self.configuration_history.append(reset_change)
                
                return True, f"Configuration reset successfully for {domain}"
            else:
                return False, f"Failed to reset configuration for {domain}"
                
        except Exception as e:
            error_msg = f"Exception during configuration reset: {e}"
            logger.error(error_msg)
            return False, error_msg
    
    async def get_configuration_history(
        self,
        domain: Optional[ConfigurationDomain] = None,
        limit: int = 100
    ) -> List[ConfigurationChange]:
        """Get configuration change history"""
        
        history = self.configuration_history
        
        if domain:
            history = [change for change in history if change.domain == domain]
        
        return history[-limit:]
    
    async def get_system_status(self) -> Dict[str, Any]:
        """Get overall system configuration status"""
        
        status = {
            "domains": {},
            "total_configurations": 0,
            "recent_changes": len([
                change for change in self.configuration_history
                if (datetime.now(timezone.utc) - change.timestamp).total_seconds() < 3600
            ]),
            "validation_cache_size": len(self.validation_cache),
            "operation_metrics": self.operation_metrics
        }
        
        for domain, manager in self.subsystem_managers.items():
            try:
                config = await manager.get_current_configuration()
                status["domains"][domain.value] = {
                    "configuration_count": len(config),
                    "last_modified": self._get_last_modification_time(domain),
                    "status": "active"
                }
                status["total_configurations"] += len(config)
            except Exception as e:
                status["domains"][domain.value] = {
                    "status": "error",
                    "error": str(e)
                }
        
        return status
    
    async def apply_user_projection_updates(
        self,
        user_id: str,
        projection: Dict[str, Any],
        scope: str = "session"
    ) -> List[Tuple[bool, str]]:
        """Apply configuration updates based on user projection changes"""
        
        results = []
        
        # Extract relevant configuration changes from projection
        vm_updates = self._extract_vm_updates_from_projection(projection)
        memory_updates = self._extract_memory_updates_from_projection(projection)
        research_updates = self._extract_research_updates_from_projection(projection)
        
        # Apply VM updates
        for path, value in vm_updates.items():
            result = await self.apply_configuration_change(
                domain=ConfigurationDomain.VM_SYSTEM,
                target_path=path,
                new_value=value,
                requester=f"user_projection:{user_id}",
                reason=f"User projection update for scope {scope}",
                context={"user_id": user_id, "scope": scope}
            )
            results.append((result[0], f"VM.{path}: {result[1]}"))
        
        # Apply memory updates
        for path, value in memory_updates.items():
            result = await self.apply_configuration_change(
                domain=ConfigurationDomain.MEMORY_SYSTEM,
                target_path=path,
                new_value=value,
                requester=f"user_projection:{user_id}",
                reason=f"User projection update for scope {scope}",
                context={"user_id": user_id, "scope": scope}
            )
            results.append((result[0], f"Memory.{path}: {result[1]}"))
        
        # Apply research updates  
        for path, value in research_updates.items():
            result = await self.apply_configuration_change(
                domain=ConfigurationDomain.RESEARCH_SYSTEM,
                target_path=path,
                new_value=value,
                requester=f"user_projection:{user_id}",
                reason=f"User projection update for scope {scope}",
                context={"user_id": user_id, "scope": scope}
            )
            results.append((result[0], f"Research.{path}: {result[1]}"))
        
        return results
    
    def _determine_scope(self, domain: ConfigurationDomain, target_path: str) -> ConfigurationScope:
        """Determine the scope of a configuration change"""
        
        if "global" in target_path.lower():
            return ConfigurationScope.SYSTEM_WIDE
        elif "instance" in target_path.lower():
            return ConfigurationScope.INSTANCE
        elif "session" in target_path.lower():
            return ConfigurationScope.SESSION
        elif "user" in target_path.lower():
            return ConfigurationScope.USER
        else:
            return ConfigurationScope.SUBSYSTEM
    
    def _get_current_value(self, domain: ConfigurationDomain, target_path: str) -> Any:
        """Get current value at target path"""
        
        config = self.active_configurations.get(domain, {})
        
        # Navigate nested path
        parts = target_path.split('.')
        current = config
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None
        
        return current
    
    def _update_active_configuration(
        self,
        domain: ConfigurationDomain,
        target_path: str,
        new_value: Any
    ) -> None:
        """Update active configuration state"""
        
        if domain not in self.active_configurations:
            self.active_configurations[domain] = {}
        
        # Navigate and update nested path
        parts = target_path.split('.')
        current = self.active_configurations[domain]
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = new_value
    
    async def _validate_configuration_change(
        self,
        change: ConfigurationChange
    ) -> ConfigurationValidation:
        """Validate configuration change"""
        
        # Check cache first
        cache_key = f"{change.domain}.{change.target_path}.{hash(str(change.new_value))}"
        if cache_key in self.validation_cache:
            return self.validation_cache[cache_key]
        
        # Get manager for domain
        manager = self.subsystem_managers.get(change.domain)
        if not manager:
            validation = ConfigurationValidation(
                is_valid=False,
                errors=[f"No manager found for domain: {change.domain}"]
            )
        else:
            validation = await manager.validate_configuration(change)
        
        # Cache result
        self.validation_cache[cache_key] = validation
        
        # Limit cache size
        if len(self.validation_cache) > 1000:
            # Remove oldest entries
            oldest_keys = list(self.validation_cache.keys())[:100]
            for key in oldest_keys:
                del self.validation_cache[key]
        
        return validation
    
    async def _check_cross_subsystem_conflicts(
        self,
        change: ConfigurationChange
    ) -> Tuple[bool, str]:
        """Check for conflicts with other subsystems"""
        
        # Example conflict checks
        if change.domain == ConfigurationDomain.VM_SYSTEM and "gpu_enabled" in change.target_path:
            # Check if artifact system also needs GPU
            artifact_config = self.active_configurations.get(ConfigurationDomain.ARTIFACT_SYSTEM, {})
            if artifact_config.get("gpu_required", False):
                return False, "GPU conflict: Artifact system already has GPU reservation"
        
        # More conflict resolution logic would go here
        
        return True, "No conflicts detected"
    
    async def _coordinate_cross_subsystem_changes(self, change: ConfigurationChange) -> None:
        """Coordinate related changes across subsystems"""
        
        # Example coordination logic
        if change.domain == ConfigurationDomain.VM_SYSTEM and "hardware_spec" in change.target_path:
            # Update capability broker about new hardware capabilities
            await self.capability_broker.request_capability(
                capability_name="hardware_update_notification",
                requester_id="global_router",
                requester_type="system",
                context={"change": change.dict() if hasattr(change, 'dict') else str(change)}
            )
    
    def _extract_vm_updates_from_projection(self, projection: Dict[str, Any]) -> Dict[str, Any]:
        """Extract VM configuration updates from user projection"""
        
        updates = {}
        capabilities = projection.get("capabilities", {}).get("flags", {})
        prefs = projection.get("prefs", {})
        
        # Map projection to VM configuration
        if capabilities.get("gpu_acceleration"):
            updates["default_hardware_spec.gpu_enabled"] = True
        
        if capabilities.get("high_memory_available"):
            updates["default_hardware_spec.memory_gb"] = 16
        
        communication_style = prefs.get("communication", {}).get("style")
        if communication_style == "technical":
            updates["default_personality.research_methodology"] = "systematic"
        elif communication_style == "casual":
            updates["default_personality.communication_style"] = "adaptive"
        
        return updates
    
    def _extract_memory_updates_from_projection(self, projection: Dict[str, Any]) -> Dict[str, Any]:
        """Extract memory system updates from user projection"""
        
        updates = {}
        prefs = projection.get("prefs", {})
        
        # Map projection preferences to memory settings
        if prefs.get("memory_retention") == "long_term":
            updates["retention_strategy"] = "importance_based"
        
        return updates
    
    def _extract_research_updates_from_projection(self, projection: Dict[str, Any]) -> Dict[str, Any]:
        """Extract research system updates from user projection"""
        
        updates = {}
        capabilities = projection.get("capabilities", {}).get("flags", {})
        
        # Map domain expertise to research configuration
        if capabilities.get("ai_expertise"):
            updates["domain_expertise.ai"] = 0.9
        
        if capabilities.get("cybersecurity"):
            updates["domain_expertise.security"] = 0.8
        
        return updates
    
    def _record_operation_metrics(self, change: ConfigurationChange) -> None:
        """Record operation metrics for performance monitoring"""
        
        domain_key = change.domain.value
        if domain_key not in self.operation_metrics:
            self.operation_metrics[domain_key] = {
                "total_changes": 0,
                "successful_changes": 0,
                "failed_changes": 0,
                "last_change": None
            }
        
        metrics = self.operation_metrics[domain_key]
        metrics["total_changes"] += 1
        
        if change.applied:
            metrics["successful_changes"] += 1
        else:
            metrics["failed_changes"] += 1
        
        metrics["last_change"] = change.timestamp.isoformat()
    
    def _get_last_modification_time(self, domain: ConfigurationDomain) -> Optional[str]:
        """Get last modification time for domain"""
        
        domain_changes = [
            change for change in self.configuration_history
            if change.domain == domain and change.applied
        ]
        
        if domain_changes:
            latest_change = max(domain_changes, key=lambda c: c.timestamp)
            return latest_change.timestamp.isoformat()
        
        return None
    
    async def _handle_vm_event(self, event: VMEvent) -> None:
        """Handle VM-related events for coordination"""
        logger.debug(f"Handling VM event: {event.vm_name}")
        
        # Example: Update related configurations based on VM changes
    
    async def _handle_memory_event(self, event: MemoryEvent) -> None:
        """Handle memory system events"""
        logger.debug(f"Handling memory event: {event.memory_type}")
    
    async def _handle_research_event(self, event: ResearchEvent) -> None:
        """Handle research system events"""
        logger.debug(f"Handling research event: {event.research_mode}")
    
    async def _handle_artifact_event(self, event: ArtifactEvent) -> None:
        """Handle artifact system events"""
        logger.debug(f"Handling artifact event: {event.capability_activated}")
    
    async def shutdown(self) -> None:
        """Graceful shutdown of configuration router"""
        
        logger.info("Shutting down Global Configuration Router...")
        
        # Save final configuration states
        for domain, config in self.active_configurations.items():
            logger.info(f"Final configuration for {domain}: {len(config)} settings")
        
        # Clear caches
        self.validation_cache.clear()
        
        logger.info("Global Configuration Router shutdown complete")


# ============================================================================
# GLOBAL ROUTER INSTANCE
# ============================================================================

_global_router: Optional[GlobalConfigurationRouter] = None

def get_global_router() -> GlobalConfigurationRouter:
    """Get global configuration router instance"""
    global _global_router
    if _global_router is None:
        _global_router = GlobalConfigurationRouter()
    return _global_router

async def initialize_global_router() -> GlobalConfigurationRouter:
    """Initialize global configuration router"""
    global _global_router
    _global_router = GlobalConfigurationRouter()
    await _global_router.initialize()
    return _global_router

async def shutdown_global_router() -> None:
    """Shutdown global configuration router"""
    global _global_router
    if _global_router:
        await _global_router.shutdown()
        _global_router = None


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize router
        router = await initialize_global_router()
        
        # Apply a VM configuration change
        success, message, change = await router.apply_configuration_change(
            domain=ConfigurationDomain.VM_SYSTEM,
            target_path="default_hardware_spec.memory_gb",
            new_value=8,
            requester="test_user",
            reason="Increase default memory allocation"
        )
        
        print(f"VM Configuration result: {success} - {message}")
        
        # Get current configuration
        vm_config = await router.get_configuration(ConfigurationDomain.VM_SYSTEM)
        print(f"Current VM config: {vm_config}")
        
        # Get system status
        status = await router.get_system_status()
        print(f"System status: {status}")
        
        # Cleanup
        await shutdown_global_router()
    
    asyncio.run(main())