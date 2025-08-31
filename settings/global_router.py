"""
SOMNUS SOVEREIGN KERNEL - Global Settings Router
==============================================

Central coordination hub for all subsystem settings management.
Routes UI commands to appropriate subsystem managers with real-time updates.

Architecture:
[UI Button Click] → [GlobalRouter] → [Specific Settings Manager] → [Live System Update]

Subsystems:
- VM Infrastructure (vm_settings.py)
- Neural Memory Runtime (memory_system_settings.py) 
- Research & Data Processing (research_and_data_processing_settings.py)
- Research System Advanced (research_settings_system.py)
- Artifact System (artifact_settings.py)
- CAS System (cas_integration_bridge.py)
- Security & Anti-Fingerprinting (ghost_model.py)
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

try:
    # Import existing settings managers
    from vm_settings import VMSettingsManager, VMPoolSettings
    from memory_system_settings import MemorySettingsManager
    from research_and_data_processing_settings import ResearchDataSettingsManager
    from research_settings_system import ResearchSettingsManager
    from artifact_settings import OnDemandCapabilityManager, ArtifactSystemSettings
    
    # Import neural memory and CAS if available
    try:
        from neural_memory_runtime import NeuralMemoryRuntime
        from cas_integration_bridge import EnhancedModelFileCreator
        from ghost_model import GhostSecurityManager
        ADVANCED_SYSTEMS_AVAILABLE = True
    except ImportError:
        ADVANCED_SYSTEMS_AVAILABLE = False
        
except ImportError as e:
    logging.warning(f"Some settings managers not available: {e}")
    # Fallback for development
    VMSettingsManager = None
    MemorySettingsManager = None
    ResearchDataSettingsManager = None
    ResearchSettingsManager = None
    OnDemandCapabilityManager = None

logger = logging.getLogger(__name__)


# ============================================================================
# GLOBAL ROUTER ENUMS AND MODELS
# ============================================================================

class SubsystemType(str, Enum):
    """Core subsystem types"""
    VM_INFRASTRUCTURE = "vm_infrastructure"
    NEURAL_MEMORY = "neural_memory"
    RESEARCH_DATA = "research_data"
    RESEARCH_ADVANCED = "research_advanced"
    ARTIFACT_SYSTEM = "artifact_system"
    CAS_SYSTEM = "cas_system"
    SECURITY_SYSTEM = "security_system"
    GLOBAL_KERNEL = "global_kernel"


class SettingCategory(str, Enum):
    """Setting category types for UI organization"""
    PERFORMANCE = "performance"
    SECURITY = "security"
    RESOURCES = "resources"
    FEATURES = "features"
    AUTOMATION = "automation"
    INTEGRATION = "integration"
    MONITORING = "monitoring"
    PRIVACY = "privacy"


class SettingType(str, Enum):
    """Setting value types for UI form generation"""
    BOOLEAN = "boolean"
    INTEGER = "integer"
    FLOAT = "float"
    STRING = "string"
    ENUM = "enum"
    ARRAY = "array"
    OBJECT = "object"
    PASSWORD = "password"
    FILE_PATH = "file_path"


class UpdateScope(str, Enum):
    """Scope of setting updates"""
    IMMEDIATE = "immediate"      # Apply instantly
    SESSION = "session"          # Apply for current session
    PERSISTENT = "persistent"    # Save to disk
    RESTART_REQUIRED = "restart_required"  # Requires system restart


@dataclass
class SettingDefinition:
    """Complete setting definition for UI generation"""
    key: str
    display_name: str
    description: str
    setting_type: SettingType
    category: SettingCategory
    default_value: Any
    current_value: Any
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    enum_options: Optional[List[str]] = None
    min_value: Optional[Union[int, float]] = None
    max_value: Optional[Union[int, float]] = None
    requires_restart: bool = False
    subsystem: SubsystemType = SubsystemType.GLOBAL_KERNEL
    dependencies: List[str] = field(default_factory=list)
    ui_hints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SettingUpdateResult:
    """Result of setting update operation"""
    success: bool
    setting_key: str
    old_value: Any
    new_value: Any
    validation_errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    requires_restart: bool = False
    affected_subsystems: List[SubsystemType] = field(default_factory=list)
    update_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class SubsystemStatus:
    """Status information for a subsystem"""
    subsystem: SubsystemType
    is_active: bool
    is_healthy: bool
    last_update: Optional[datetime]
    pending_changes: int
    error_count: int
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    configuration_version: str = "1.0.0"


# ============================================================================
# GLOBAL SETTINGS ROUTER
# ============================================================================

class GlobalSettingsRouter:
    """
    Central coordination hub for all Somnus Sovereign Kernel settings.
    Provides unified interface for UI interactions and real-time system updates.
    """
    
    def __init__(
        self, 
        user_id: str,
        config_base_path: str = "data/settings",
        enable_auto_save: bool = True
    ):
        self.user_id = user_id
        self.router_id = uuid4()
        self.config_base_path = Path(config_base_path)
        self.enable_auto_save = enable_auto_save
        
        # Subsystem managers
        self.subsystem_managers: Dict[SubsystemType, Any] = {}
        self.subsystem_status: Dict[SubsystemType, SubsystemStatus] = {}
        
        # Settings registry
        self.settings_registry: Dict[str, SettingDefinition] = {}
        self.setting_dependencies: Dict[str, Set[str]] = {}
        self.pending_updates: Dict[str, Any] = {}
        
        # Runtime state
        self.is_initialized = False
        self.background_tasks: List[asyncio.Task] = []
        self.update_history: List[SettingUpdateResult] = []
        
        # Performance tracking
        self.performance_metrics = {
            "total_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "average_update_time_ms": 0.0,
            "last_performance_check": datetime.now(timezone.utc)
        }
        
        logger.info(f"Global Settings Router initialized for user {user_id}")
    
    async def initialize(self) -> bool:
        """Initialize all subsystem managers and build settings registry"""
        try:
            logger.info("Initializing Global Settings Router...")
            
            # Create base directories
            self.config_base_path.mkdir(parents=True, exist_ok=True)
            
            # Initialize subsystem managers
            await self._initialize_subsystem_managers()
            
            # Build comprehensive settings registry
            await self._build_settings_registry()
            
            # Start background monitoring
            await self._start_background_tasks()
            
            self.is_initialized = True
            logger.info("Global Settings Router initialization complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Global Settings Router: {e}")
            return False
    
    async def _initialize_subsystem_managers(self) -> None:
        """Initialize all available subsystem managers"""
        
        # VM Infrastructure
        if VMSettingsManager:
            try:
                vm_pool_settings = VMPoolSettings()
                vm_manager = VMSettingsManager(
                    pool_settings=vm_pool_settings,
                    config_path=str(self.config_base_path / "vm_settings.json")
                )
                await vm_manager.initialize()
                self.subsystem_managers[SubsystemType.VM_INFRASTRUCTURE] = vm_manager
                logger.info("VM Infrastructure manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize VM manager: {e}")
        
        # Neural Memory System
        if MemorySettingsManager:
            try:
                memory_manager = await MemorySettingsManager.create_memory_settings(self.user_id)
                self.subsystem_managers[SubsystemType.NEURAL_MEMORY] = memory_manager
                logger.info("Neural Memory manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Memory manager: {e}")
        
        # Research & Data Processing
        if ResearchDataSettingsManager:
            try:
                research_manager = ResearchDataSettingsManager(self.user_id)
                await research_manager.activate_research_systems()
                self.subsystem_managers[SubsystemType.RESEARCH_DATA] = research_manager
                logger.info("Research Data manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Research Data manager: {e}")
        
        # Advanced Research System
        if ResearchSettingsManager:
            try:
                advanced_research = ResearchSettingsManager(self.user_id)
                await advanced_research.initialize()
                self.subsystem_managers[SubsystemType.RESEARCH_ADVANCED] = advanced_research
                logger.info("Advanced Research manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Advanced Research manager: {e}")
        
        # Artifact System
        if OnDemandCapabilityManager:
            try:
                artifact_settings = ArtifactSystemSettings()
                artifact_manager = OnDemandCapabilityManager(
                    settings=artifact_settings,
                    config_path=str(self.config_base_path / "artifact_settings.json")
                )
                await artifact_manager.initialize()
                self.subsystem_managers[SubsystemType.ARTIFACT_SYSTEM] = artifact_manager
                logger.info("Artifact System manager initialized")
            except Exception as e:
                logger.error(f"Failed to initialize Artifact manager: {e}")
        
        # Advanced systems (CAS, Neural Memory Runtime, Security)
        if ADVANCED_SYSTEMS_AVAILABLE:
            await self._initialize_advanced_systems()
        
        # Update subsystem status
        await self._update_subsystem_status()
    
    async def _initialize_advanced_systems(self) -> None:
        """Initialize advanced systems (CAS, Neural Memory Runtime, Security)"""
        try:
            # TODO: Initialize CAS System when available
            # cas_manager = CASSystemManager()
            # self.subsystem_managers[SubsystemType.CAS_SYSTEM] = cas_manager
            
            # TODO: Initialize Neural Memory Runtime when available
            # neural_runtime = NeuralMemoryRuntime()
            # self.subsystem_managers[SubsystemType.NEURAL_MEMORY] = neural_runtime
            
            # TODO: Initialize Ghost Security Manager when available
            # security_manager = GhostSecurityManager()
            # self.subsystem_managers[SubsystemType.SECURITY_SYSTEM] = security_manager
            
            logger.info("Advanced systems initialization placeholder")
            
        except Exception as e:
            logger.error(f"Failed to initialize advanced systems: {e}")
    
    async def _build_settings_registry(self) -> None:
        """Build comprehensive settings registry from all subsystems"""
        
        # Core kernel settings
        await self._register_kernel_settings()
        
        # VM Infrastructure settings
        if SubsystemType.VM_INFRASTRUCTURE in self.subsystem_managers:
            await self._register_vm_settings()
        
        # Neural Memory settings
        if SubsystemType.NEURAL_MEMORY in self.subsystem_managers:
            await self._register_memory_settings()
        
        # Research system settings
        if SubsystemType.RESEARCH_DATA in self.subsystem_managers:
            await self._register_research_settings()
        
        # Artifact system settings
        if SubsystemType.ARTIFACT_SYSTEM in self.subsystem_managers:
            await self._register_artifact_settings()
        
        logger.info(f"Settings registry built with {len(self.settings_registry)} settings")
    
    async def _register_kernel_settings(self) -> None:
        """Register core kernel-level settings"""
        
        kernel_settings = [
            SettingDefinition(
                key="kernel.auto_save_enabled",
                display_name="Auto-Save Settings",
                description="Automatically save setting changes to disk",
                setting_type=SettingType.BOOLEAN,
                category=SettingCategory.AUTOMATION,
                default_value=True,
                current_value=self.enable_auto_save,
                subsystem=SubsystemType.GLOBAL_KERNEL
            ),
            SettingDefinition(
                key="kernel.performance_monitoring",
                display_name="Performance Monitoring",
                description="Enable system-wide performance monitoring",
                setting_type=SettingType.BOOLEAN,
                category=SettingCategory.MONITORING,
                default_value=True,
                current_value=True,
                subsystem=SubsystemType.GLOBAL_KERNEL
            ),
            SettingDefinition(
                key="kernel.log_level",
                display_name="Log Level",
                description="System-wide logging verbosity",
                setting_type=SettingType.ENUM,
                category=SettingCategory.MONITORING,
                default_value="INFO",
                current_value="INFO",
                enum_options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                subsystem=SubsystemType.GLOBAL_KERNEL
            ),
            SettingDefinition(
                key="kernel.max_concurrent_operations",
                display_name="Max Concurrent Operations",
                description="Maximum number of concurrent system operations",
                setting_type=SettingType.INTEGER,
                category=SettingCategory.PERFORMANCE,
                default_value=10,
                current_value=10,
                min_value=1,
                max_value=100,
                subsystem=SubsystemType.GLOBAL_KERNEL
            )
        ]
        
        for setting in kernel_settings:
            self.settings_registry[setting.key] = setting
    
    async def _register_vm_settings(self) -> None:
        """Register VM infrastructure settings"""
        
        vm_manager = self.subsystem_managers[SubsystemType.VM_INFRASTRUCTURE]
        
        vm_settings = [
            SettingDefinition(
                key="vm.max_concurrent_vms",
                display_name="Maximum Concurrent VMs",
                description="Maximum number of VMs that can run simultaneously",
                setting_type=SettingType.INTEGER,
                category=SettingCategory.RESOURCES,
                default_value=10,
                current_value=vm_manager.pool_settings.max_concurrent_vms,
                min_value=1,
                max_value=100,
                subsystem=SubsystemType.VM_INFRASTRUCTURE
            ),
            SettingDefinition(
                key="vm.auto_scaling_enabled",
                display_name="Auto-Scaling",
                description="Automatically scale VM resources based on demand",
                setting_type=SettingType.BOOLEAN,
                category=SettingCategory.AUTOMATION,
                default_value=True,
                current_value=vm_manager.pool_settings.auto_scaling_enabled,
                subsystem=SubsystemType.VM_INFRASTRUCTURE
            ),
            SettingDefinition(
                key="vm.health_check_interval",
                display_name="Health Check Interval (seconds)",
                description="How often to check VM health status",
                setting_type=SettingType.INTEGER,
                category=SettingCategory.MONITORING,
                default_value=60,
                current_value=vm_manager.pool_settings.health_check_interval,
                min_value=10,
                max_value=3600,
                subsystem=SubsystemType.VM_INFRASTRUCTURE
            ),
            SettingDefinition(
                key="vm.auto_backup_enabled",
                display_name="Automatic Backups",
                description="Enable automatic VM state backups",
                setting_type=SettingType.BOOLEAN,
                category=SettingCategory.AUTOMATION,
                default_value=True,
                current_value=vm_manager.pool_settings.auto_backup_enabled,
                subsystem=SubsystemType.VM_INFRASTRUCTURE
            )
        ]
        
        for setting in vm_settings:
            self.settings_registry[setting.key] = setting
    
    async def _register_memory_settings(self) -> None:
        """Register neural memory system settings"""
        
        memory_manager = self.subsystem_managers[SubsystemType.NEURAL_MEMORY]
        
        memory_settings = [
            SettingDefinition(
                key="memory.max_memories_per_user",
                display_name="Max Memories Per User",
                description="Maximum number of memories stored per user",
                setting_type=SettingType.INTEGER,
                category=SettingCategory.RESOURCES,
                default_value=10000,
                current_value=memory_manager.storage.max_memories_per_user,
                min_value=1000,
                max_value=100000,
                subsystem=SubsystemType.NEURAL_MEMORY
            ),
            SettingDefinition(
                key="memory.encryption_level",
                display_name="Encryption Level",
                description="Memory encryption security level",
                setting_type=SettingType.ENUM,
                category=SettingCategory.SECURITY,
                default_value="STANDARD",
                current_value=memory_manager.privacy.encryption_level.value,
                enum_options=["NONE", "BASIC", "STANDARD", "HIGH", "MAXIMUM"],
                subsystem=SubsystemType.NEURAL_MEMORY
            ),
            SettingDefinition(
                key="memory.similarity_threshold",
                display_name="Similarity Threshold",
                description="Minimum similarity score for memory retrieval",
                setting_type=SettingType.FLOAT,
                category=SettingCategory.PERFORMANCE,
                default_value=0.7,
                current_value=memory_manager.retrieval.similarity_threshold,
                min_value=0.1,
                max_value=1.0,
                subsystem=SubsystemType.NEURAL_MEMORY
            )
        ]
        
        for setting in memory_settings:
            self.settings_registry[setting.key] = setting
    
    async def _register_research_settings(self) -> None:
        """Register research system settings"""
        
        research_manager = self.subsystem_managers[SubsystemType.RESEARCH_DATA]
        
        research_settings = [
            SettingDefinition(
                key="research.ai_browser_enabled",
                display_name="AI Browser Research",
                description="Enable AI-driven browser research capabilities",
                setting_type=SettingType.BOOLEAN,
                category=SettingCategory.FEATURES,
                default_value=True,
                current_value=research_manager.ai_browser.enabled,
                subsystem=SubsystemType.RESEARCH_DATA
            ),
            SettingDefinition(
                key="research.auto_trigger_research",
                display_name="Auto-Trigger Research",
                description="Automatically start research on relevant queries",
                setting_type=SettingType.BOOLEAN,
                category=SettingCategory.AUTOMATION,
                default_value=False,
                current_value=research_manager.ai_browser.auto_trigger_research,
                subsystem=SubsystemType.RESEARCH_DATA
            ),
            SettingDefinition(
                key="research.max_search_depth",
                display_name="Maximum Search Depth",
                description="Maximum levels of recursive search",
                setting_type=SettingType.INTEGER,
                category=SettingCategory.PERFORMANCE,
                default_value=3,
                current_value=research_manager.web_search.max_search_depth,
                min_value=1,
                max_value=5,
                subsystem=SubsystemType.RESEARCH_DATA
            )
        ]
        
        for setting in research_settings:
            self.settings_registry[setting.key] = setting
    
    async def _register_artifact_settings(self) -> None:
        """Register artifact system settings"""
        
        artifact_manager = self.subsystem_managers[SubsystemType.ARTIFACT_SYSTEM]
        
        artifact_settings = [
            SettingDefinition(
                key="artifacts.unlimited_execution",
                display_name="Unlimited Execution",
                description="Enable unlimited artifact execution capabilities",
                setting_type=SettingType.BOOLEAN,
                category=SettingCategory.FEATURES,
                default_value=True,
                current_value=artifact_manager.settings.unlimited_execution_enabled,
                subsystem=SubsystemType.ARTIFACT_SYSTEM,
                ui_hints={"warning": "Disabling this limits artifact capabilities"}
            ),
            SettingDefinition(
                key="artifacts.max_concurrent_executions",
                display_name="Max Concurrent Executions",
                description="Maximum number of concurrent artifact executions",
                setting_type=SettingType.INTEGER,
                category=SettingCategory.PERFORMANCE,
                default_value=5,
                current_value=artifact_manager.settings.max_concurrent_executions,
                min_value=1,
                max_value=20,
                subsystem=SubsystemType.ARTIFACT_SYSTEM
            ),
            SettingDefinition(
                key="artifacts.auto_cleanup_enabled",
                display_name="Auto-Cleanup",
                description="Automatically cleanup completed artifact containers",
                setting_type=SettingType.BOOLEAN,
                category=SettingCategory.AUTOMATION,
                default_value=True,
                current_value=artifact_manager.settings.auto_cleanup_enabled,
                subsystem=SubsystemType.ARTIFACT_SYSTEM
            )
        ]
        
        for setting in artifact_settings:
            self.settings_registry[setting.key] = setting
    
    # ========================================================================
    # PUBLIC API METHODS
    # ========================================================================
    
    async def get_all_settings(self) -> Dict[str, Dict[str, Any]]:
        """Get all settings organized by category for UI display"""
        if not self.is_initialized:
            await self.initialize()
        
        settings_by_category = {}
        
        for setting in self.settings_registry.values():
            category = setting.category.value
            if category not in settings_by_category:
                settings_by_category[category] = {}
            
            settings_by_category[category][setting.key] = {
                "display_name": setting.display_name,
                "description": setting.description,
                "type": setting.setting_type.value,
                "current_value": setting.current_value,
                "default_value": setting.default_value,
                "validation_rules": setting.validation_rules,
                "enum_options": setting.enum_options,
                "min_value": setting.min_value,
                "max_value": setting.max_value,
                "requires_restart": setting.requires_restart,
                "subsystem": setting.subsystem.value,
                "dependencies": setting.dependencies,
                "ui_hints": setting.ui_hints
            }
        
        return settings_by_category
    
    async def get_settings_by_subsystem(self, subsystem: SubsystemType) -> Dict[str, Any]:
        """Get all settings for a specific subsystem"""
        if not self.is_initialized:
            await self.initialize()
        
        subsystem_settings = {}
        
        for key, setting in self.settings_registry.items():
            if setting.subsystem == subsystem:
                subsystem_settings[key] = {
                    "display_name": setting.display_name,
                    "description": setting.description,
                    "current_value": setting.current_value,
                    "type": setting.setting_type.value
                }
        
        return subsystem_settings
    
    async def update_setting(
        self, 
        setting_key: str, 
        new_value: Any, 
        update_scope: UpdateScope = UpdateScope.PERSISTENT
    ) -> SettingUpdateResult:
        """Update a single setting with validation and propagation"""
        
        start_time = datetime.now(timezone.utc)
        
        try:
            if setting_key not in self.settings_registry:
                return SettingUpdateResult(
                    success=False,
                    setting_key=setting_key,
                    old_value=None,
                    new_value=new_value,
                    validation_errors=[f"Setting '{setting_key}' not found"]
                )
            
            setting_def = self.settings_registry[setting_key]
            old_value = setting_def.current_value
            
            # Validate new value
            validation_errors = await self._validate_setting_value(setting_def, new_value)
            if validation_errors:
                return SettingUpdateResult(
                    success=False,
                    setting_key=setting_key,
                    old_value=old_value,
                    new_value=new_value,
                    validation_errors=validation_errors
                )
            
            # Apply update to subsystem
            success = await self._apply_setting_to_subsystem(setting_def, new_value)
            
            if success:
                # Update registry
                setting_def.current_value = new_value
                
                # Handle persistence
                if update_scope == UpdateScope.PERSISTENT and self.enable_auto_save:
                    await self._persist_setting_change(setting_key, new_value)
                
                # Track performance
                update_time = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
                self._update_performance_metrics(update_time, True)
                
                result = SettingUpdateResult(
                    success=True,
                    setting_key=setting_key,
                    old_value=old_value,
                    new_value=new_value,
                    requires_restart=setting_def.requires_restart,
                    affected_subsystems=[setting_def.subsystem]
                )
                
                self.update_history.append(result)
                logger.info(f"Setting '{setting_key}' updated: {old_value} → {new_value}")
                
                return result
            
            else:
                return SettingUpdateResult(
                    success=False,
                    setting_key=setting_key,
                    old_value=old_value,
                    new_value=new_value,
                    validation_errors=["Failed to apply setting to subsystem"]
                )
                
        except Exception as e:
            logger.error(f"Error updating setting '{setting_key}': {e}")
            self._update_performance_metrics(0, False)
            
            return SettingUpdateResult(
                success=False,
                setting_key=setting_key,
                old_value=None,
                new_value=new_value,
                validation_errors=[f"Internal error: {str(e)}"]
            )
    
    async def update_multiple_settings(
        self, 
        settings_updates: Dict[str, Any], 
        update_scope: UpdateScope = UpdateScope.PERSISTENT
    ) -> List[SettingUpdateResult]:
        """Update multiple settings with dependency resolution"""
        
        results = []
        
        # Sort updates by dependencies
        sorted_updates = await self._resolve_setting_dependencies(settings_updates)
        
        for setting_key, new_value in sorted_updates:
            result = await self.update_setting(setting_key, new_value, update_scope)
            results.append(result)
            
            # Stop on first failure if there are dependencies
            if not result.success and setting_key in self.setting_dependencies:
                logger.warning(f"Stopping batch update due to dependency failure: {setting_key}")
                break
        
        return results
    
    async def get_subsystem_status(self) -> Dict[SubsystemType, SubsystemStatus]:
        """Get current status of all subsystems"""
        await self._update_subsystem_status()
        return self.subsystem_status.copy()
    
    async def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health and performance metrics"""
        
        subsystem_health = {}
        total_errors = 0
        healthy_subsystems = 0
        
        for subsystem_type, status in self.subsystem_status.items():
            subsystem_health[subsystem_type.value] = {
                "is_healthy": status.is_healthy,
                "error_count": status.error_count,
                "pending_changes": status.pending_changes
            }
            
            if status.is_healthy:
                healthy_subsystems += 1
            total_errors += status.error_count
        
        return {
            "router_status": "healthy" if self.is_initialized else "initializing",
            "healthy_subsystems": healthy_subsystems,
            "total_subsystems": len(self.subsystem_status),
            "total_errors": total_errors,
            "total_settings": len(self.settings_registry),
            "pending_updates": len(self.pending_updates),
            "performance_metrics": self.performance_metrics,
            "subsystem_health": subsystem_health,
            "last_health_check": datetime.now(timezone.utc).isoformat()
        }
    
    # ========================================================================
    # INTERNAL HELPER METHODS
    # ========================================================================
    
    async def _validate_setting_value(
        self, 
        setting_def: SettingDefinition, 
        value: Any
    ) -> List[str]:
        """Validate setting value against definition rules"""
        
        errors = []
        
        # Type validation
        if setting_def.setting_type == SettingType.BOOLEAN and not isinstance(value, bool):
            errors.append(f"Expected boolean, got {type(value).__name__}")
        
        elif setting_def.setting_type == SettingType.INTEGER:
            if not isinstance(value, int):
                errors.append(f"Expected integer, got {type(value).__name__}")
            else:
                if setting_def.min_value is not None and value < setting_def.min_value:
                    errors.append(f"Value {value} below minimum {setting_def.min_value}")
                if setting_def.max_value is not None and value > setting_def.max_value:
                    errors.append(f"Value {value} above maximum {setting_def.max_value}")
        
        elif setting_def.setting_type == SettingType.FLOAT:
            if not isinstance(value, (int, float)):
                errors.append(f"Expected number, got {type(value).__name__}")
            else:
                if setting_def.min_value is not None and value < setting_def.min_value:
                    errors.append(f"Value {value} below minimum {setting_def.min_value}")
                if setting_def.max_value is not None and value > setting_def.max_value:
                    errors.append(f"Value {value} above maximum {setting_def.max_value}")
        
        elif setting_def.setting_type == SettingType.ENUM:
            if setting_def.enum_options and value not in setting_def.enum_options:
                errors.append(f"Value '{value}' not in allowed options: {setting_def.enum_options}")
        
        elif setting_def.setting_type == SettingType.STRING:
            if not isinstance(value, str):
                errors.append(f"Expected string, got {type(value).__name__}")
        
        # Custom validation rules
        for rule_name, rule_value in setting_def.validation_rules.items():
            if rule_name == "max_length" and isinstance(value, str) and len(value) > rule_value:
                errors.append(f"String length {len(value)} exceeds maximum {rule_value}")
            elif rule_name == "min_length" and isinstance(value, str) and len(value) < rule_value:
                errors.append(f"String length {len(value)} below minimum {rule_value}")
        
        return errors
    
    async def _apply_setting_to_subsystem(
        self, 
        setting_def: SettingDefinition, 
        new_value: Any
    ) -> bool:
        """Apply setting change to the appropriate subsystem"""
        
        try:
            subsystem = setting_def.subsystem
            setting_key = setting_def.key
            
            if subsystem == SubsystemType.GLOBAL_KERNEL:
                return await self._apply_kernel_setting(setting_key, new_value)
            
            elif subsystem == SubsystemType.VM_INFRASTRUCTURE:
                return await self._apply_vm_setting(setting_key, new_value)
            
            elif subsystem == SubsystemType.NEURAL_MEMORY:
                return await self._apply_memory_setting(setting_key, new_value)
            
            elif subsystem == SubsystemType.RESEARCH_DATA:
                return await self._apply_research_setting(setting_key, new_value)
            
            elif subsystem == SubsystemType.ARTIFACT_SYSTEM:
                return await self._apply_artifact_setting(setting_key, new_value)
            
            else:
                logger.warning(f"Unknown subsystem: {subsystem}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to apply setting {setting_def.key}: {e}")
            return False
    
    async def _apply_kernel_setting(self, setting_key: str, new_value: Any) -> bool:
        """Apply kernel-level setting changes"""
        
        if setting_key == "kernel.auto_save_enabled":
            self.enable_auto_save = new_value
            return True
        
        elif setting_key == "kernel.log_level":
            # Update logging level for all loggers
            logging.getLogger().setLevel(getattr(logging, new_value))
            return True
        
        elif setting_key == "kernel.performance_monitoring":
            # Enable/disable performance monitoring
            return True
        
        elif setting_key == "kernel.max_concurrent_operations":
            # Update concurrent operation limits
            return True
        
        return True
    
    async def _apply_vm_setting(self, setting_key: str, new_value: Any) -> bool:
        """Apply VM infrastructure setting changes"""
        
        if SubsystemType.VM_INFRASTRUCTURE not in self.subsystem_managers:
            return False
        
        vm_manager = self.subsystem_managers[SubsystemType.VM_INFRASTRUCTURE]
        
        try:
            if setting_key == "vm.max_concurrent_vms":
                vm_manager.pool_settings.max_concurrent_vms = new_value
            elif setting_key == "vm.auto_scaling_enabled":
                vm_manager.pool_settings.auto_scaling_enabled = new_value
            elif setting_key == "vm.health_check_interval":
                vm_manager.pool_settings.health_check_interval = new_value
            elif setting_key == "vm.auto_backup_enabled":
                vm_manager.pool_settings.auto_backup_enabled = new_value
            
            # Save VM settings
            await vm_manager._save_settings()
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply VM setting {setting_key}: {e}")
            return False
    
    async def _apply_memory_setting(self, setting_key: str, new_value: Any) -> bool:
        """Apply memory system setting changes"""
        
        if SubsystemType.NEURAL_MEMORY not in self.subsystem_managers:
            return False
        
        memory_manager = self.subsystem_managers[SubsystemType.NEURAL_MEMORY]
        
        try:
            if setting_key == "memory.max_memories_per_user":
                await memory_manager.update_storage_settings(max_memories_per_user=new_value)
            elif setting_key == "memory.encryption_level":
                await memory_manager.update_privacy_settings(encryption_level=new_value)
            elif setting_key == "memory.similarity_threshold":
                await memory_manager.update_retrieval_settings(similarity_threshold=new_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply memory setting {setting_key}: {e}")
            return False
    
    async def _apply_research_setting(self, setting_key: str, new_value: Any) -> bool:
        """Apply research system setting changes"""
        
        if SubsystemType.RESEARCH_DATA not in self.subsystem_managers:
            return False
        
        research_manager = self.subsystem_managers[SubsystemType.RESEARCH_DATA]
        
        try:
            if setting_key == "research.ai_browser_enabled":
                await research_manager.update_ai_browser_settings(enabled=new_value)
            elif setting_key == "research.auto_trigger_research":
                await research_manager.update_ai_browser_settings(auto_trigger_research=new_value)
            elif setting_key == "research.max_search_depth":
                await research_manager.update_web_search_settings(max_search_depth=new_value)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply research setting {setting_key}: {e}")
            return False
    
    async def _apply_artifact_setting(self, setting_key: str, new_value: Any) -> bool:
        """Apply artifact system setting changes"""
        
        if SubsystemType.ARTIFACT_SYSTEM not in self.subsystem_managers:
            return False
        
        artifact_manager = self.subsystem_managers[SubsystemType.ARTIFACT_SYSTEM]
        
        try:
            if setting_key == "artifacts.unlimited_execution":
                artifact_manager.settings.unlimited_execution_enabled = new_value
            elif setting_key == "artifacts.max_concurrent_executions":
                artifact_manager.settings.max_concurrent_executions = new_value
            elif setting_key == "artifacts.auto_cleanup_enabled":
                artifact_manager.settings.auto_cleanup_enabled = new_value
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to apply artifact setting {setting_key}: {e}")
            return False
    
    async def _resolve_setting_dependencies(
        self, 
        settings_updates: Dict[str, Any]
    ) -> List[Tuple[str, Any]]:
        """Resolve setting dependencies and return in correct order"""
        
        # For now, return as-is. TODO: Implement dependency graph resolution
        return list(settings_updates.items())
    
    async def _persist_setting_change(self, setting_key: str, new_value: Any) -> None:
        """Persist setting change to disk"""
        
        try:
            settings_file = self.config_base_path / "global_settings.json"
            
            # Load existing settings
            settings_data = {}
            if settings_file.exists():
                with open(settings_file, 'r') as f:
                    settings_data = json.load(f)
            
            # Update setting
            settings_data[setting_key] = new_value
            settings_data["last_updated"] = datetime.now(timezone.utc).isoformat()
            
            # Save back to disk
            with open(settings_file, 'w') as f:
                json.dump(settings_data, f, indent=2, default=str)
                
        except Exception as e:
            logger.error(f"Failed to persist setting {setting_key}: {e}")
    
    async def _update_subsystem_status(self) -> None:
        """Update status information for all subsystems"""
        
        for subsystem_type, manager in self.subsystem_managers.items():
            try:
                is_active = hasattr(manager, 'is_active') and manager.is_active
                is_healthy = True  # TODO: Implement health checks
                
                self.subsystem_status[subsystem_type] = SubsystemStatus(
                    subsystem=subsystem_type,
                    is_active=is_active,
                    is_healthy=is_healthy,
                    last_update=datetime.now(timezone.utc),
                    pending_changes=0,
                    error_count=0
                )
                
            except Exception as e:
                logger.error(f"Failed to update status for {subsystem_type}: {e}")
                self.subsystem_status[subsystem_type] = SubsystemStatus(
                    subsystem=subsystem_type,
                    is_active=False,
                    is_healthy=False,
                    last_update=datetime.now(timezone.utc),
                    pending_changes=0,
                    error_count=1
                )
    
    def _update_performance_metrics(self, update_time_ms: float, success: bool) -> None:
        """Update performance tracking metrics"""
        
        self.performance_metrics["total_updates"] += 1
        
        if success:
            self.performance_metrics["successful_updates"] += 1
        else:
            self.performance_metrics["failed_updates"] += 1
        
        # Update average update time
        current_avg = self.performance_metrics["average_update_time_ms"]
        total_updates = self.performance_metrics["total_updates"]
        
        new_avg = ((current_avg * (total_updates - 1)) + update_time_ms) / total_updates
        self.performance_metrics["average_update_time_ms"] = round(new_avg, 2)
        
        self.performance_metrics["last_performance_check"] = datetime.now(timezone.utc)
    
    async def _start_background_tasks(self) -> None:
        """Start background monitoring and maintenance tasks"""
        
        # Status monitoring task
        status_task = asyncio.create_task(self._background_status_monitor())
        self.background_tasks.append(status_task)
        
        logger.info("Background tasks started")
    
    async def _background_status_monitor(self) -> None:
        """Background task for monitoring subsystem status"""
        
        while True:
            try:
                await self._update_subsystem_status()
                await asyncio.sleep(60)  # Update every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Background status monitoring error: {e}")
                await asyncio.sleep(30)
    
    async def shutdown(self) -> None:
        """Graceful shutdown of the global router"""
        
        logger.info("Shutting down Global Settings Router...")
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Wait for tasks to complete
        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)
        
        # Shutdown subsystem managers
        for manager in self.subsystem_managers.values():
            if hasattr(manager, 'shutdown'):
                try:
                    await manager.shutdown()
                except Exception as e:
                    logger.error(f"Error shutting down subsystem manager: {e}")
        
        logger.info("Global Settings Router shutdown complete")


# ============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# ============================================================================

async def create_global_router(user_id: str, config_path: str = "data/settings") -> GlobalSettingsRouter:
    """Factory function to create and initialize global settings router"""
    
    router = GlobalSettingsRouter(user_id=user_id, config_base_path=config_path)
    
    success = await router.initialize()
    if not success:
        raise RuntimeError("Failed to initialize Global Settings Router")
    
    return router


async def get_system_capabilities() -> Dict[str, bool]:
    """Get available system capabilities for UI display"""
    
    capabilities = {
        "vm_infrastructure": VMSettingsManager is not None,
        "neural_memory": MemorySettingsManager is not None,
        "research_data": ResearchDataSettingsManager is not None,
        "research_advanced": ResearchSettingsManager is not None,
        "artifact_system": OnDemandCapabilityManager is not None,
        "advanced_systems": ADVANCED_SYSTEMS_AVAILABLE
    }
    
    return capabilities


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

async def example_global_router_usage():
    """Example usage of the Global Settings Router"""
    
    # Create router for user
    router = await create_global_router(user_id="user_123")
    
    # Get all settings for UI display
    all_settings = await router.get_all_settings()
    print(f"Available settings categories: {list(all_settings.keys())}")
    
    # Update a VM setting
    result = await router.update_setting("vm.max_concurrent_vms", 15)
    print(f"VM setting update: {result.success}")
    
    # Update multiple settings
    updates = {
        "memory.max_memories_per_user": 20000,
        "research.ai_browser_enabled": True,
        "artifacts.unlimited_execution": True
    }
    results = await router.update_multiple_settings(updates)
    print(f"Batch update results: {[r.success for r in results]}")
    
    # Check system health
    health = await router.get_system_health()
    print(f"System health: {health['router_status']}")
    
    # Get subsystem status
    status = await router.get_subsystem_status()
    print(f"Active subsystems: {[s.value for s in status.keys()]}")
    
    # Cleanup
    await router.shutdown()


if __name__ == "__main__":
    import asyncio
    asyncio.run(example_global_router_usage())
