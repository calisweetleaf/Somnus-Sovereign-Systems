"""
SOMNUS SOVEREIGN SYSTEMS - Research Subsystem Settings
Production-ready settings management for the research subsystem

Features:
- Per-subsystem configuration with inheritance from global settings
- Instance-specific and per-chat/category configurations
- No JSON file management - all programmatic configuration
- Hot-reloading and runtime configuration updates
- Memory-persistent settings storage
- Environment-aware configuration profiles
"""

import asyncio
import json
import logging
import os
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field, asdict
from enum import Enum
import hashlib
from concurrent.futures import ThreadPoolExecutor
import threading

# System imports
from schemas.session import SessionID, UserID
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope

logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION ENUMS AND MODELS
# ============================================================================

class ResearchMode(str, Enum):
    """Research operation modes"""
    SPEED = "speed"              # Fast, surface-level research
    BALANCED = "balanced"        # Balance of speed and depth
    COMPREHENSIVE = "comprehensive"  # Deep, thorough research
    EXHAUSTIVE = "exhaustive"    # Maximum depth and coverage


class SourceTrustLevel(str, Enum):
    """Source trust and filtering levels"""
    UNRESTRICTED = "unrestricted"  # All sources allowed
    CURATED = "curated"           # Prefer high-quality sources
    ACADEMIC = "academic"         # Academic and peer-reviewed only
    VERIFIED = "verified"         # Only verified, authoritative sources


class AIProcessingLevel(str, Enum):
    """AI processing intensity levels"""
    MINIMAL = "minimal"           # Basic processing
    STANDARD = "standard"         # Standard AI analysis
    ADVANCED = "advanced"         # Advanced reasoning and synthesis
    MAXIMUM = "maximum"           # Full AI cognitive processing


class CacheStrategy(str, Enum):
    """Research caching strategies"""
    DISABLED = "disabled"         # No caching
    SESSION_ONLY = "session_only" # Cache for session duration
    PERSISTENT = "persistent"     # Long-term caching
    INTELLIGENT = "intelligent"   # AI-managed cache optimization


@dataclass
class ResearchPerformanceConfig:
    """Performance and resource configuration"""
    max_concurrent_searches: int = 5
    search_timeout_seconds: int = 30
    max_sources_per_query: int = 20
    max_entity_extraction_depth: int = 10
    cache_retention_hours: int = 24
    memory_usage_limit_mb: int = 512
    cpu_usage_limit_percent: int = 80
    enable_gpu_acceleration: bool = True
    background_processing: bool = True
    real_time_streaming: bool = True


@dataclass
class ResearchQualityConfig:
    """Research quality and validation settings"""
    minimum_source_reliability: float = 0.6
    required_source_diversity: int = 3
    contradiction_tolerance: float = 0.1
    fact_verification_enabled: bool = True
    bias_detection_enabled: bool = True
    plagiarism_checking: bool = True
    source_freshness_days: int = 365
    cross_reference_validation: bool = True
    confidence_threshold: float = 0.7


@dataclass
class ResearchScopeConfig:
    """Research scope and boundary settings"""
    max_research_depth: int = 8
    max_research_breadth: int = 50
    enable_recursive_expansion: bool = True
    auto_follow_citations: bool = True
    include_historical_context: bool = True
    include_future_projections: bool = False
    geographical_scope: List[str] = field(default_factory=lambda: ["global"])
    temporal_scope_years: int = 10
    domain_restrictions: List[str] = field(default_factory=list)
    language_preferences: List[str] = field(default_factory=lambda: ["en"])


@dataclass
class ResearchOutputConfig:
    """Output formatting and generation settings"""
    default_export_format: str = "research_artifact"
    auto_generate_summary: bool = True
    include_source_citations: bool = True
    include_confidence_metrics: bool = True
    include_contradiction_analysis: bool = True
    include_entity_graph: bool = True
    include_methodology_notes: bool = False
    generate_executive_summary: bool = True
    enable_transform_options: bool = True
    auto_save_to_memory: bool = True


@dataclass
class ResearchCollaborationConfig:
    """Multi-agent collaboration settings"""
    enable_collaboration: bool = True
    max_collaborating_agents: int = 5
    collaboration_timeout_minutes: int = 10
    auto_delegate_complex_tasks: bool = True
    consensus_requirement_threshold: float = 0.8
    debate_resolution_enabled: bool = True
    specialist_agent_allocation: bool = True
    collaborative_fact_checking: bool = True


@dataclass
class ResearchPrivacyConfig:
    """Privacy and security settings"""
    anonymous_browsing: bool = True
    vpn_rotation_enabled: bool = False
    search_history_retention: bool = False
    source_tracking_disabled: bool = True
    user_agent_rotation: bool = True
    request_rate_limiting: bool = True
    geo_location_masking: bool = True
    search_fingerprint_randomization: bool = True


@dataclass
class ResearchInstanceConfig:
    """Complete instance configuration"""
    instance_id: str
    user_id: str
    instance_name: str
    created_at: datetime
    last_modified: datetime
    
    # Configuration version tracking
    config_version: str = "1.0.0"
    migration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Core settings
    research_mode: ResearchMode = ResearchMode.BALANCED
    source_trust_level: SourceTrustLevel = SourceTrustLevel.CURATED
    ai_processing_level: AIProcessingLevel = AIProcessingLevel.STANDARD
    cache_strategy: CacheStrategy = CacheStrategy.INTELLIGENT
    
    # Subsystem configurations
    performance: ResearchPerformanceConfig = field(default_factory=ResearchPerformanceConfig)
    quality: ResearchQualityConfig = field(default_factory=ResearchQualityConfig)
    scope: ResearchScopeConfig = field(default_factory=ResearchScopeConfig)
    output: ResearchOutputConfig = field(default_factory=ResearchOutputConfig)
    collaboration: ResearchCollaborationConfig = field(default_factory=ResearchCollaborationConfig)
    privacy: ResearchPrivacyConfig = field(default_factory=ResearchPrivacyConfig)
    
    # Advanced customizations
    custom_prompts: Dict[str, str] = field(default_factory=dict)
    specialized_agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    domain_expertise: Dict[str, float] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Runtime state
    is_active: bool = True
    last_used: Optional[datetime] = None
    usage_statistics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoryConfig:
    """Category-specific configuration overrides"""
    category_id: str
    category_name: str
    parent_instance_id: str
    
    # Override specific settings for this category
    mode_override: Optional[ResearchMode] = None
    trust_level_override: Optional[SourceTrustLevel] = None
    processing_override: Optional[AIProcessingLevel] = None
    
    # Category-specific customizations
    domain_focus: List[str] = field(default_factory=list)
    preferred_sources: List[str] = field(default_factory=list)
    excluded_sources: List[str] = field(default_factory=list)
    custom_validation_rules: Dict[str, Any] = field(default_factory=dict)
    
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    is_active: bool = True


# ============================================================================
# MEMORY-PERSISTENT SETTINGS MANAGER
# ============================================================================

class ResearchSettingsManager:
    """Production settings manager with memory persistence"""
    
    # Configuration version management
    CURRENT_CONFIG_VERSION = "1.0.0"
    SUPPORTED_VERSIONS = ["1.0.0", "0.9.0", "0.8.0", "0.7.0"]
    MIGRATION_PATHS = {
        "0.7.0": "0.8.0",
        "0.8.0": "0.9.0", 
        "0.9.0": "1.0.0"
    }
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        self.instance_configs: Dict[str, ResearchInstanceConfig] = {}
        self.category_configs: Dict[str, CategoryConfig] = {}
        self.global_defaults = self._create_global_defaults()
        self._lock = threading.RLock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="settings")
        
        # Cache for rapid access
        self._config_cache: Dict[str, Dict[str, Any]] = {}
        self._cache_timestamps: Dict[str, datetime] = {}
        self._cache_ttl = timedelta(minutes=30)
        
        # Migration tracking
        self._migration_stats = {
            'total_migrations': 0,
            'successful_migrations': 0,
            'failed_migrations': 0,
            'rollbacks': 0
        }
        
        logger.info("ResearchSettingsManager initialized with memory persistence")
    
    def _create_global_defaults(self) -> ResearchInstanceConfig:
        """Create sensible global defaults"""
        return ResearchInstanceConfig(
            instance_id="global_defaults",
            user_id="system",
            instance_name="Global Defaults",
            created_at=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc),
            research_mode=ResearchMode.BALANCED,
            source_trust_level=SourceTrustLevel.CURATED,
            ai_processing_level=AIProcessingLevel.STANDARD,
            cache_strategy=CacheStrategy.INTELLIGENT
        )
    
    async def initialize(self) -> None:
        """Initialize settings manager and load existing configurations"""
        try:
            # Load existing instance configurations from memory
            await self._load_instance_configs_from_memory()
            
            # Load category configurations
            await self._load_category_configs_from_memory()
            
            # Validate and migrate if needed
            await self._validate_and_migrate_configs()
            
            logger.info(f"Settings manager initialized with {len(self.instance_configs)} instances and {len(self.category_configs)} categories")
            
        except Exception as e:
            logger.error(f"Failed to initialize settings manager: {e}")
            raise
    
    async def get_instance_config(
        self,
        instance_id: str,
        user_id: str,
        create_if_missing: bool = True
    ) -> ResearchInstanceConfig:
        """Get configuration for specific instance"""
        
        with self._lock:
            # Check cache first
            cache_key = f"instance_{instance_id}"
            if self._is_cache_valid(cache_key):
                cached_config = self._config_cache[cache_key]
                return ResearchInstanceConfig(**cached_config)
            
            # Check in-memory configs
            if instance_id in self.instance_configs:
                config = self.instance_configs[instance_id]
                self._update_cache(cache_key, asdict(config))
                return config
            
            # Create new if requested
            if create_if_missing:
                config = await self._create_new_instance_config(instance_id, user_id)
                self.instance_configs[instance_id] = config
                await self._persist_instance_config(config)
                self._update_cache(cache_key, asdict(config))
                return config
            
            # Return global defaults
            return self.global_defaults
    
    async def get_effective_config(
        self,
        instance_id: str,
        user_id: str,
        category_id: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Get effective configuration with inheritance and overrides"""
        
        # Start with instance config
        instance_config = await self.get_instance_config(instance_id, user_id)
        effective_config = asdict(instance_config)
        
        # Apply category overrides if specified
        if category_id:
            category_config = await self.get_category_config(category_id, instance_id)
            if category_config:
                effective_config = self._apply_category_overrides(effective_config, category_config)
        
        # Apply session-specific overrides if needed
        if session_id:
            session_overrides = await self._get_session_overrides(session_id, user_id)
            if session_overrides:
                effective_config = self._merge_configs(effective_config, session_overrides)
        
        return effective_config
    
    async def update_instance_config(
        self,
        instance_id: str,
        user_id: str,
        updates: Dict[str, Any]
    ) -> ResearchInstanceConfig:
        """Update instance configuration"""
        
        with self._lock:
            config = await self.get_instance_config(instance_id, user_id)
            
            # Apply updates
            updated_config = self._apply_config_updates(config, updates)
            updated_config.last_modified = datetime.now(timezone.utc)
            
            # Validate updates
            self._validate_config(updated_config)
            
            # Save changes
            self.instance_configs[instance_id] = updated_config
            await self._persist_instance_config(updated_config)
            
            # Clear cache
            cache_key = f"instance_{instance_id}"
            if cache_key in self._config_cache:
                del self._config_cache[cache_key]
            
            logger.info(f"Updated instance config {instance_id}")
            return updated_config
    
    async def create_category_config(
        self,
        category_name: str,
        parent_instance_id: str,
        overrides: Dict[str, Any] = False
    ) -> CategoryConfig:
        """Create new category configuration"""
        
        category_config = CategoryConfig(
            category_id=str(uuid4()),
            category_name=category_name,
            parent_instance_id=parent_instance_id
        )
        
        if overrides:
            category_config = self._apply_category_updates(category_config, overrides)
        
        self.category_configs[category_config.category_id] = category_config
        await self._persist_category_config(category_config)
        
        logger.info(f"Created category config {category_config.category_id}: {category_name}")
        return category_config
    
    async def get_category_config(
        self,
        category_id: str,
        parent_instance_id: str
    ) -> Optional[CategoryConfig]:
        """Get category configuration"""
        
        cache_key = f"category_{category_id}"
        if self._is_cache_valid(cache_key):
            cached_config = self._config_cache[cache_key]
            return CategoryConfig(**cached_config)
        
        if category_id in self.category_configs:
            config = self.category_configs[category_id]
            self._update_cache(cache_key, asdict(config))
            return config
        
        # Try loading from memory
        config = await self._load_category_config_from_memory(category_id)
        if config:
            self.category_configs[category_id] = config
            self._update_cache(cache_key, asdict(config))
            return config
        
        return None
    
    async def get_user_instances(self, user_id: str) -> List[ResearchInstanceConfig]:
        """Get all instances for a user"""
        
        user_instances = []
        for config in self.instance_configs.values():
            if config.user_id == user_id:
                user_instances.append(config)
        
        # Sort by last used (most recent first)
        user_instances.sort(key=lambda x: x.last_used or x.created_at, reverse=True)
        return user_instances
    
    async def delete_instance_config(self, instance_id: str, user_id: str) -> bool:
        """Delete instance configuration"""
        
        with self._lock:
            if instance_id in self.instance_configs:
                config = self.instance_configs[instance_id]
                if config.user_id != user_id:
                    raise PermissionError("User does not own this instance configuration")
                
                # Remove from memory
                del self.instance_configs[instance_id]
                
                # Remove from persistent storage
                await self._delete_instance_config_from_memory(instance_id)
                
                # Clear cache
                cache_key = f"instance_{instance_id}"
                if cache_key in self._config_cache:
                    del self._config_cache[cache_key]
                
                logger.info(f"Deleted instance config {instance_id}")
                return True
        
        return False
    
    async def export_user_configs(self, user_id: str) -> Dict[str, Any]:
        """Export all user configurations"""
        
        user_instances = await self.get_user_instances(user_id)
        user_categories = [c for c in self.category_configs.values() 
                          if any(i.instance_id == c.parent_instance_id for i in user_instances)]
        
        export_data = {
            'export_version': '1.0',
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'user_id': user_id,
            'instances': [asdict(config) for config in user_instances],
            'categories': [asdict(config) for config in user_categories]
        }
        
        return export_data
    
    async def import_user_configs(
        self,
        user_id: str,
        import_data: Dict[str, Any],
        overwrite_existing: bool = False
    ) -> Dict[str, List[str]]:
        """Import user configurations"""
        
        results = {'imported_instances': [], 'imported_categories': [], 'errors': []}
        
        try:
            # Import instances
            for instance_data in import_data.get('instances', []):
                try:
                    # Update user_id to importing user
                    instance_data['user_id'] = user_id
                    instance_data['last_modified'] = datetime.now(timezone.utc)
                    
                    config = ResearchInstanceConfig(**instance_data)
                    
                    if config.instance_id in self.instance_configs and not overwrite_existing:
                        # Generate new ID to avoid conflicts
                        config.instance_id = str(uuid4())
                        config.instance_name += " (Imported)"
                    
                    self.instance_configs[config.instance_id] = config
                    await self._persist_instance_config(config)
                    results['imported_instances'].append(config.instance_id)
                    
                except Exception as e:
                    results['errors'].append(f"Failed to import instance: {e}")
            
            # Import categories
            for category_data in import_data.get('categories', []):
                try:
                    config = CategoryConfig(**category_data)
                    
                    if config.category_id in self.category_configs and not overwrite_existing:
                        config.category_id = str(uuid4())
                        config.category_name += " (Imported)"
                    
                    self.category_configs[config.category_id] = config
                    await self._persist_category_config(config)
                    results['imported_categories'].append(config.category_id)
                    
                except Exception as e:
                    results['errors'].append(f"Failed to import category: {e}")
            
            logger.info(f"Imported {len(results['imported_instances'])} instances and {len(results['imported_categories'])} categories for user {user_id}")
            
        except Exception as e:
            logger.error(f"Import failed for user {user_id}: {e}")
            results['errors'].append(f"Import failed: {e}")
        
        return results
    
    # Private methods for memory persistence
    
    async def _load_instance_configs_from_memory(self) -> None:
        """Load instance configurations from memory system"""
        
        try:
            memories = await self.memory_manager.retrieve_memories(
                user_id="system",
                query="research_instance_config",
                memory_type=MemoryType.SYSTEM_CONFIG,
                limit=1000
            )
            
            for memory in memories:
                try:
                    config_data = json.loads(memory.get('content', '{}'))
                    config = ResearchInstanceConfig(**config_data)
                    self.instance_configs[config.instance_id] = config
                except Exception as e:
                    logger.warning(f"Failed to load instance config from memory: {e}")
            
            logger.info(f"Loaded {len(self.instance_configs)} instance configs from memory")
            
        except Exception as e:
            logger.error(f"Failed to load instance configs from memory: {e}")
    
    async def _load_category_configs_from_memory(self) -> None:
        """Load category configurations from memory system"""
        
        try:
            memories = await self.memory_manager.retrieve_memories(
                user_id="system",
                query="research_category_config",
                memory_type=MemoryType.SYSTEM_CONFIG,
                limit=1000
            )
            
            for memory in memories:
                try:
                    config_data = json.loads(memory.get('content', '{}'))
                    config = CategoryConfig(**config_data)
                    self.category_configs[config.category_id] = config
                except Exception as e:
                    logger.warning(f"Failed to load category config from memory: {e}")
            
            logger.info(f"Loaded {len(self.category_configs)} category configs from memory")
            
        except Exception as e:
            logger.error(f"Failed to load category configs from memory: {e}")
    
    async def _persist_instance_config(self, config: ResearchInstanceConfig) -> None:
        """Persist instance configuration to memory"""
        
        try:
            config_json = json.dumps(asdict(config), default=str, indent=2)
            
            await self.memory_manager.store_memory(
                user_id="system",
                content=config_json,
                memory_type=MemoryType.SYSTEM_CONFIG,
                importance=MemoryImportance.HIGH,
                scope=MemoryScope.GLOBAL,
                tags=['research_instance_config', config.instance_id, config.user_id],
                metadata={
                    'config_type': 'research_instance',
                    'instance_id': config.instance_id,
                    'user_id': config.user_id,
                    'instance_name': config.instance_name,
                    'last_modified': config.last_modified.isoformat(),
                    'checksum': hashlib.sha256(config_json.encode()).hexdigest()[:16]
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to persist instance config {config.instance_id}: {e}")
            raise
    
    async def _persist_category_config(self, config: CategoryConfig) -> None:
        """Persist category configuration to memory"""
        
        try:
            config_json = json.dumps(asdict(config), default=str, indent=2)
            
            await self.memory_manager.store_memory(
                user_id="system",
                content=config_json,
                memory_type=MemoryType.SYSTEM_CONFIG,
                importance=MemoryImportance.MEDIUM,
                scope=MemoryScope.GLOBAL,
                tags=['research_category_config', config.category_id, config.parent_instance_id],
                metadata={
                    'config_type': 'research_category',
                    'category_id': config.category_id,
                    'category_name': config.category_name,
                    'parent_instance_id': config.parent_instance_id,
                    'checksum': hashlib.sha256(config_json.encode()).hexdigest()[:16]
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to persist category config {config.category_id}: {e}")
            raise
    
    async def _create_new_instance_config(
        self,
        instance_id: str,
        user_id: str
    ) -> ResearchInstanceConfig:
        """Create new instance configuration with intelligent defaults"""
        
        # Generate intelligent instance name
        existing_count = len([c for c in self.instance_configs.values() if c.user_id == user_id])
        instance_name = f"Research Instance #{existing_count + 1}"
        
        config = ResearchInstanceConfig(
            instance_id=instance_id,
            user_id=user_id,
            instance_name=instance_name,
            created_at=datetime.now(timezone.utc),
            last_modified=datetime.now(timezone.utc)
        )
        
        # Apply user-specific intelligent defaults based on usage patterns
        await self._apply_intelligent_defaults(config, user_id)
        
        return config
    
    async def _apply_intelligent_defaults(
        self,
        config: ResearchInstanceConfig,
        user_id: str
    ) -> None:
        """Apply intelligent defaults based on user patterns"""
        
        try:
            # Analyze user's previous research patterns
            user_memories = await self.memory_manager.retrieve_memories(
                user_id=user_id,
                query="research_report OR research_session",
                limit=50
            )
            
            if user_memories:
                # Analyze patterns and adjust defaults
                research_topics = []
                complexity_scores = []
                
                for memory in user_memories:
                    metadata = memory.get('metadata', {})
                    
                    # Extract research complexity
                    if 'depth_score' in metadata:
                        complexity_scores.append(metadata['depth_score'])
                    
                    # Extract topics
                    if 'primary_topics' in metadata:
                        research_topics.extend(metadata['primary_topics'])
                
                # Adjust configuration based on patterns
                if complexity_scores:
                    avg_complexity = sum(complexity_scores) / len(complexity_scores)
                    
                    if avg_complexity >= 8:
                        config.research_mode = ResearchMode.COMPREHENSIVE
                        config.ai_processing_level = AIProcessingLevel.ADVANCED
                    elif avg_complexity >= 6:
                        config.research_mode = ResearchMode.BALANCED
                        config.ai_processing_level = AIProcessingLevel.STANDARD
                    else:
                        config.research_mode = ResearchMode.SPEED
                        config.ai_processing_level = AIProcessingLevel.MINIMAL
                
                # Set domain expertise based on research history
                if research_topics:
                    topic_counts = {}
                    for topic in research_topics:
                        topic_counts[topic] = topic_counts.get(topic, 0) + 1
                    
                    # Set expertise levels based on frequency
                    total_topics = len(research_topics)
                    for topic, count in topic_counts.items():
                        expertise_level = min(1.0, count / total_topics * 5)  # Scale to 0-1
                        if expertise_level > 0.3:  # Only include significant topics
                            config.domain_expertise[topic] = expertise_level
            
        except Exception as e:
            logger.warning(f"Failed to apply intelligent defaults for user {user_id}: {e}")
    
    def _apply_config_updates(
        self,
        config: ResearchInstanceConfig,
        updates: Dict[str, Any]
    ) -> ResearchInstanceConfig:
        """Apply configuration updates safely"""
        
        # Create a copy to avoid modifying original
        updated_config = ResearchInstanceConfig(**asdict(config))
        
        # Apply direct field updates
        for field, value in updates.items():
            if hasattr(updated_config, field):
                setattr(updated_config, field, value)
            elif '.' in field:
                # Handle nested field updates (e.g., "performance.max_concurrent_searches")
                self._set_nested_field(updated_config, field, value)
        
        return updated_config
    
    def _set_nested_field(self, obj: Any, field_path: str, value: Any) -> None:
        """Set nested field using dot notation"""
        
        parts = field_path.split('.')
        current = obj
        
        for part in parts[:-1]:
            if hasattr(current, part):
                current = getattr(current, part)
            else:
                return  # Invalid path
        
        final_field = parts[-1]
        if hasattr(current, final_field):
            setattr(current, final_field, value)
    
    def _validate_config(self, config: ResearchInstanceConfig) -> None:
        """Validate configuration values"""
        
        # Validate performance limits
        if config.performance.max_concurrent_searches < 1 or config.performance.max_concurrent_searches > 20:
            raise ValueError("max_concurrent_searches must be between 1 and 20")
        
        if config.performance.search_timeout_seconds < 5 or config.performance.search_timeout_seconds > 300:
            raise ValueError("search_timeout_seconds must be between 5 and 300")
        
        # Validate quality thresholds
        if not 0.0 <= config.quality.minimum_source_reliability <= 1.0:
            raise ValueError("minimum_source_reliability must be between 0.0 and 1.0")
        
        if not 0.0 <= config.quality.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0.0 and 1.0")
        
        # Validate scope limits
        if config.scope.max_research_depth < 1 or config.scope.max_research_depth > 20:
            raise ValueError("max_research_depth must be between 1 and 20")
        
        if config.scope.max_research_breadth < 1 or config.scope.max_research_breadth > 200:
            raise ValueError("max_research_breadth must be between 1 and 200")
    
    def _apply_category_overrides(
        self,
        base_config: Dict[str, Any],
        category_config: CategoryConfig
    ) -> Dict[str, Any]:
        """Apply category-specific overrides to base configuration"""
        
        config = base_config.copy()
        
        # Apply enum overrides
        if category_config.mode_override:
            config['research_mode'] = category_config.mode_override.value
        
        if category_config.trust_level_override:
            config['source_trust_level'] = category_config.trust_level_override.value
        
        if category_config.processing_override:
            config['ai_processing_level'] = category_config.processing_override.value
        
        # Apply custom validation rules
        if category_config.custom_validation_rules:
            if 'quality' not in config:
                config['quality'] = {}
            config['quality'].update(category_config.custom_validation_rules)
        
        # Apply source preferences
        if category_config.preferred_sources:
            if 'scope' not in config:
                config['scope'] = {}
            config['scope']['preferred_sources'] = category_config.preferred_sources
        
        if category_config.excluded_sources:
            if 'scope' not in config:
                config['scope'] = {}
            config['scope']['excluded_sources'] = category_config.excluded_sources
        
        return config
    
    def _merge_configs(self, base_config: Dict[str, Any], overrides: Dict[str, Any]) -> Dict[str, Any]:
        """Deep merge configuration dictionaries"""
        
        merged = base_config.copy()
        
        for key, value in overrides.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self._merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    async def _get_session_overrides(self, session_id: str, user_id: str) -> Optional[Dict[str, Any]]:
        """Get session-specific configuration overrides"""
        
        try:
            memories = await self.memory_manager.retrieve_memories(
                user_id=user_id,
                query=f"session_config:{session_id}",
                memory_type=MemoryType.SESSION_CONFIG,
                limit=1
            )
            
            if memories:
                return json.loads(memories[0].get('content', '{}'))
            
        except Exception as e:
            logger.warning(f"Failed to load session overrides for {session_id}: {e}")
        
        return None
    
    async def _validate_and_migrate_configs(self) -> None:
        """Validate and migrate existing configurations"""
        
        migration_count = 0
        
        for instance_id, config in list(self.instance_configs.items()):
            try:
                # Validate current config
                self._validate_config(config)
                
                # Check if migration is needed (version checking would go here)
                if self._needs_migration(config):
                    migrated_config = self._migrate_config(config)
                    self.instance_configs[instance_id] = migrated_config
                    await self._persist_instance_config(migrated_config)
                    migration_count += 1
                    
            except Exception as e:
                logger.error(f"Failed to validate/migrate config {instance_id}: {e}")
                # Remove invalid configs
                del self.instance_configs[instance_id]
        
        if migration_count > 0:
            logger.info(f"Migrated {migration_count} configuration(s)")
    
    def _needs_migration(self, config: ResearchInstanceConfig) -> bool:
        """Check if configuration needs migration with comprehensive validation"""
        
        try:
            # Check version compatibility
            if not hasattr(config, 'config_version') or not config.config_version:
                logger.info(f"Config {config.instance_id} missing version - migration needed")
                return True
            
            current_version = self._parse_version(self.CURRENT_CONFIG_VERSION)
            config_version = self._parse_version(config.config_version)
            
            # Migration needed if config version is older than current
            if config_version < current_version:
                logger.info(f"Config {config.instance_id} version {config.config_version} < {self.CURRENT_CONFIG_VERSION} - migration needed")
                return True
            
            # Check for missing required fields based on current version
            if self._has_missing_required_fields(config):
                logger.info(f"Config {config.instance_id} missing required fields - migration needed")
                return True
            
            # Check for deprecated fields that need cleanup
            if self._has_deprecated_fields(config):
                logger.info(f"Config {config.instance_id} has deprecated fields - migration needed")
                return True
            
            # Check for invalid enum values that need updating
            if self._has_invalid_enum_values(config):
                logger.info(f"Config {config.instance_id} has invalid enum values - migration needed")
                return True
            
            # Check for configuration schema changes
            if self._schema_validation_required(config):
                logger.info(f"Config {config.instance_id} requires schema validation - migration needed")
                return True
            
            # Check for performance optimizations
            if self._performance_optimizations_available(config):
                logger.debug(f"Config {config.instance_id} has performance optimizations available")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Error checking migration needs for config {config.instance_id}: {e}")
            # If we can't determine, assume migration is needed for safety
            return True
    
    def _migrate_config(self, config: ResearchInstanceConfig) -> ResearchInstanceConfig:
        """Migrate configuration to current version with comprehensive upgrade path"""
        
        migration_start_time = datetime.now(timezone.utc)
        original_version = getattr(config, 'config_version', '0.7.0')
        
        try:
            logger.info(f"Starting migration of config {config.instance_id} from version {original_version} to {self.CURRENT_CONFIG_VERSION}")
            
            # Create backup of original config
            backup_config = self._create_config_backup(config)
            
            # Start with current config
            migrated_config = config
            current_version = original_version
            
            # Follow migration path to current version
            migration_chain = self._get_migration_chain(current_version, self.CURRENT_CONFIG_VERSION)
            
            for target_version in migration_chain:
                logger.info(f"Migrating config {config.instance_id} from {current_version} to {target_version}")
                
                # Apply version-specific migration
                migrated_config = self._apply_version_migration(migrated_config, current_version, target_version)
                
                # Validate migration step
                if not self._validate_migration_step(migrated_config, target_version):
                    raise ValueError(f"Migration validation failed for version {target_version}")
                
                # Update version tracking
                migrated_config.config_version = target_version
                current_version = target_version
                
                # Record migration step
                migration_record = {
                    'from_version': current_version,
                    'to_version': target_version,
                    'timestamp': datetime.now(timezone.utc).isoformat(),
                    'success': True
                }
                
                if not hasattr(migrated_config, 'migration_history'):
                    migrated_config.migration_history = []
                migrated_config.migration_history.append(migration_record)
            
            # Final validation and cleanup
            migrated_config = self._finalize_migration(migrated_config)
            
            # Update metadata
            migrated_config.last_modified = datetime.now(timezone.utc)
            
            # Log successful migration
            migration_duration = (datetime.now(timezone.utc) - migration_start_time).total_seconds()
            logger.info(f"Successfully migrated config {config.instance_id} from {original_version} to {self.CURRENT_CONFIG_VERSION} in {migration_duration:.2f}s")
            
            # Update stats
            self._migration_stats['total_migrations'] += 1
            self._migration_stats['successful_migrations'] += 1
            
            return migrated_config
            
        except Exception as e:
            logger.error(f"Migration failed for config {config.instance_id}: {e}")
            
            # Attempt rollback if possible
            try:
                rolled_back_config = self._rollback_migration(backup_config, config.instance_id)
                logger.warning(f"Rolled back config {config.instance_id} to original state")
                
                self._migration_stats['rollbacks'] += 1
                return rolled_back_config
                
            except Exception as rollback_error:
                logger.error(f"Rollback failed for config {config.instance_id}: {rollback_error}")
            
            # Update stats
            self._migration_stats['total_migrations'] += 1
            self._migration_stats['failed_migrations'] += 1
            
            # Return original config if all else fails
            return config
    
    def _parse_version(self, version_string: str) -> Tuple[int, int, int]:
        """Parse version string into comparable tuple"""
        try:
            parts = version_string.split('.')
            major = int(parts[0]) if len(parts) > 0 else 0
            minor = int(parts[1]) if len(parts) > 1 else 0
            patch = int(parts[2]) if len(parts) > 2 else 0
            return (major, minor, patch)
        except (ValueError, IndexError):
            logger.warning(f"Invalid version string: {version_string}, defaulting to (0, 0, 0)")
            return (0, 0, 0)
    
    def _has_missing_required_fields(self, config: ResearchInstanceConfig) -> bool:
        """Check for missing required fields in current version"""
        
        required_fields_v1 = [
            'instance_id', 'user_id', 'instance_name', 'created_at', 'last_modified',
            'research_mode', 'source_trust_level', 'ai_processing_level', 'cache_strategy',
            'performance', 'quality', 'scope', 'output', 'collaboration', 'privacy'
        ]
        
        for field in required_fields_v1:
            if not hasattr(config, field) or getattr(config, field) is None:
                logger.debug(f"Missing required field: {field}")
                return True
        
        # Check nested required fields
        nested_checks = [
            ('performance', ['max_concurrent_searches', 'search_timeout_seconds']),
            ('quality', ['minimum_source_reliability', 'confidence_threshold']),
            ('scope', ['max_research_depth', 'max_research_breadth']),
            ('output', ['default_export_format', 'auto_generate_summary']),
            ('collaboration', ['enable_collaboration', 'max_collaborating_agents']),
            ('privacy', ['anonymous_browsing', 'search_history_retention'])
        ]
        
        for parent_field, child_fields in nested_checks:
            if hasattr(config, parent_field):
                parent_obj = getattr(config, parent_field)
                for child_field in child_fields:
                    if not hasattr(parent_obj, child_field):
                        logger.debug(f"Missing nested field: {parent_field}.{child_field}")
                        return True
        
        return False
    
    def _has_deprecated_fields(self, config: ResearchInstanceConfig) -> bool:
        """Check for deprecated fields that need removal"""
        
        deprecated_fields = [
            'legacy_search_engine',
            'old_ai_model',
            'deprecated_cache_settings',
            'legacy_output_format',
            'old_collaboration_mode'
        ]
        
        config_dict = asdict(config)
        
        for field in deprecated_fields:
            if self._field_exists_in_dict(config_dict, field):
                logger.debug(f"Found deprecated field: {field}")
                return True
        
        return False
    
    def _has_invalid_enum_values(self, config: ResearchInstanceConfig) -> bool:
        """Check for invalid enum values that need updating"""
        
        try:
            # Check main enum fields
            enum_checks = [
                ('research_mode', ResearchMode),
                ('source_trust_level', SourceTrustLevel),
                ('ai_processing_level', AIProcessingLevel),
                ('cache_strategy', CacheStrategy)
            ]
            
            for field_name, enum_class in enum_checks:
                if hasattr(config, field_name):
                    field_value = getattr(config, field_name)
                    if isinstance(field_value, str):
                        # Check if string value is valid enum
                        try:
                            enum_class(field_value)
                        except ValueError:
                            logger.debug(f"Invalid enum value for {field_name}: {field_value}")
                            return True
                    elif field_value is not None and not isinstance(field_value, enum_class):
                        logger.debug(f"Invalid enum type for {field_name}: {type(field_value)}")
                        return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Error checking enum values: {e}")
            return True
    
    def _schema_validation_required(self, config: ResearchInstanceConfig) -> bool:
        """Check if schema validation and updates are required"""
        
        try:
            # Check if all dataclass fields are properly typed
            config_dict = asdict(config)
            
            # Validate structure integrity
            if not self._validate_config_structure(config_dict):
                return True
            
            # Check for schema version compatibility
            if not self._validate_schema_compatibility(config):
                return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Schema validation check failed: {e}")
            return True
    
    def _performance_optimizations_available(self, config: ResearchInstanceConfig) -> bool:
        """Check if performance optimizations are available"""
        
        try:
            # Check for outdated performance settings
            if hasattr(config, 'performance'):
                perf = config.performance
                
                # Check if using old default values that can be optimized
                if (hasattr(perf, 'max_concurrent_searches') and 
                    perf.max_concurrent_searches < 3):
                    return True
                
                if (hasattr(perf, 'cache_retention_hours') and 
                    perf.cache_retention_hours < 12):
                    return True
                
                if (hasattr(perf, 'enable_gpu_acceleration') and 
                    not perf.enable_gpu_acceleration):
                    return True
            
            return False
            
        except Exception as e:
            logger.warning(f"Performance optimization check failed: {e}")
            return False
    
    def _get_migration_chain(self, from_version: str, to_version: str) -> List[str]:
        """Get the migration chain from one version to another"""
        
        if from_version == to_version:
            return []
        
        chain = []
        current = from_version
        
        # Follow migration path
        while current != to_version and current in self.MIGRATION_PATHS:
            next_version = self.MIGRATION_PATHS[current]
            chain.append(next_version)
            current = next_version
            
            # Prevent infinite loops
            if len(chain) > 10:
                raise ValueError(f"Migration chain too long from {from_version} to {to_version}")
        
        if current != to_version:
            raise ValueError(f"No migration path from {from_version} to {to_version}")
        
        return chain
    
    def _apply_version_migration(
        self, 
        config: ResearchInstanceConfig, 
        from_version: str, 
        to_version: str
    ) -> ResearchInstanceConfig:
        """Apply specific version migration"""
        
        migration_method_name = f"_migrate_from_{from_version.replace('.', '_')}_to_{to_version.replace('.', '_')}"
        
        if hasattr(self, migration_method_name):
            migration_method = getattr(self, migration_method_name)
            return migration_method(config)
        else:
            # Generic migration
            return self._apply_generic_migration(config, from_version, to_version)
    
    def _migrate_from_0_7_0_to_0_8_0(self, config: ResearchInstanceConfig) -> ResearchInstanceConfig:
        """Migrate from version 0.7.0 to 0.8.0"""
        
        # Add new fields introduced in 0.8.0
        if not hasattr(config, 'collaboration'):
            config.collaboration = ResearchCollaborationConfig()
        
        if not hasattr(config, 'privacy'):
            config.privacy = ResearchPrivacyConfig()
        
        # Update performance defaults
        if hasattr(config, 'performance'):
            if not hasattr(config.performance, 'background_processing'):
                config.performance.background_processing = True
            if not hasattr(config.performance, 'real_time_streaming'):
                config.performance.real_time_streaming = True
        
        # Migrate old cache settings
        if hasattr(config, 'legacy_cache_settings'):
            if hasattr(config, 'performance'):
                config.performance.cache_retention_hours = getattr(config, 'legacy_cache_settings', {}).get('retention_hours', 24)
            delattr(config, 'legacy_cache_settings')
        
        return config
    
    def _migrate_from_0_8_0_to_0_9_0(self, config: ResearchInstanceConfig) -> ResearchInstanceConfig:
        """Migrate from version 0.8.0 to 0.9.0"""
        
        # Add new quality features
        if hasattr(config, 'quality'):
            if not hasattr(config.quality, 'bias_detection_enabled'):
                config.quality.bias_detection_enabled = True
            if not hasattr(config.quality, 'plagiarism_checking'):
                config.quality.plagiarism_checking = True
        
        # Update scope configuration
        if hasattr(config, 'scope'):
            if not hasattr(config.scope, 'domain_restrictions'):
                config.scope.domain_restrictions = []
            if not hasattr(config.scope, 'language_preferences'):
                config.scope.language_preferences = ["en"]
        
        # Add specialized agents support
        if not hasattr(config, 'specialized_agents'):
            config.specialized_agents = {}
        
        return config
    
    def _migrate_from_0_9_0_to_1_0_0(self, config: ResearchInstanceConfig) -> ResearchInstanceConfig:
        """Migrate from version 0.9.0 to 1.0.0"""
        
        # Add configuration version tracking
        if not hasattr(config, 'config_version'):
            config.config_version = "1.0.0"
        
        if not hasattr(config, 'migration_history'):
            config.migration_history = []
        
        # Add usage statistics
        if not hasattr(config, 'usage_statistics'):
            config.usage_statistics = {
                'total_research_sessions': 0,
                'average_session_duration': 0,
                'preferred_research_modes': {},
                'most_used_sources': [],
                'performance_metrics': {}
            }
        
        # Update output configuration
        if hasattr(config, 'output'):
            if not hasattr(config.output, 'enable_transform_options'):
                config.output.enable_transform_options = True
            if not hasattr(config.output, 'auto_save_to_memory'):
                config.output.auto_save_to_memory = True
        
        # Add domain expertise
        if not hasattr(config, 'domain_expertise'):
            config.domain_expertise = {}
        
        return config
    
    def _apply_generic_migration(
        self, 
        config: ResearchInstanceConfig, 
        from_version: str, 
        to_version: str
    ) -> ResearchInstanceConfig:
        """Apply generic migration when no specific migration exists"""
        
        # Add any missing fields with defaults
        default_config = ResearchInstanceConfig(
            instance_id=config.instance_id,
            user_id=config.user_id,
            instance_name=config.instance_name,
            created_at=config.created_at,
            last_modified=datetime.now(timezone.utc)
        )
        
        # Copy existing fields
        for field in asdict(default_config):
            if hasattr(config, field):
                setattr(default_config, field, getattr(config, field))
        
        # Ensure version is updated
        default_config.config_version = to_version
        
        return default_config
    
    def _validate_migration_step(self, config: ResearchInstanceConfig, target_version: str) -> bool:
        """Validate that migration step was successful"""
        
        try:
            # Check that version was updated
            if hasattr(config, 'config_version') and config.config_version != target_version:
                return False
            
            # Validate configuration structure
            self._validate_config(config)
            
            # Check version-specific requirements
            if target_version == "1.0.0":
                required_fields = ['config_version', 'migration_history', 'usage_statistics']
                for field in required_fields:
                    if not hasattr(config, field):
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Migration validation failed: {e}")
            return False
    
    def _finalize_migration(self, config: ResearchInstanceConfig) -> ResearchInstanceConfig:
        """Finalize migration with cleanup and optimization"""
        
        # Remove any remaining deprecated fields
        deprecated_fields = [
            'legacy_search_engine', 'old_ai_model', 'deprecated_cache_settings',
            'legacy_output_format', 'old_collaboration_mode'
        ]
        
        for field in deprecated_fields:
            if hasattr(config, field):
                delattr(config, field)
        
        # Optimize performance settings
        if hasattr(config, 'performance'):
            self._optimize_performance_settings(config.performance)
        
        # Validate final configuration
        self._validate_config(config)
        
        return config
    
    def _optimize_performance_settings(self, performance: ResearchPerformanceConfig) -> None:
        """Optimize performance settings based on current best practices"""
        
        # Update concurrent searches based on system capabilities
        if performance.max_concurrent_searches < 3:
            performance.max_concurrent_searches = 5
        
        # Optimize timeout settings
        if performance.search_timeout_seconds < 20:
            performance.search_timeout_seconds = 30
        
        # Enable modern features
        performance.enable_gpu_acceleration = True
        performance.background_processing = True
        performance.real_time_streaming = True
    
    def _create_config_backup(self, config: ResearchInstanceConfig) -> ResearchInstanceConfig:
        """Create backup of configuration before migration"""
        
        backup_dict = asdict(config)
        backup_dict['backup_timestamp'] = datetime.now(timezone.utc).isoformat()
        
        # Create new instance from backup data
        backup_config = ResearchInstanceConfig(**{
            k: v for k, v in backup_dict.items() 
            if k in [f.name for f in ResearchInstanceConfig.__dataclass_fields__.values()]
        })
        
        return backup_config
    
    def _rollback_migration(self, backup_config: ResearchInstanceConfig, instance_id: str) -> ResearchInstanceConfig:
        """Rollback migration to backup configuration"""
        
        # Restore from backup
        restored_config = backup_config
        
        # Add rollback record
        rollback_record = {
            'rollback_timestamp': datetime.now(timezone.utc).isoformat(),
            'reason': 'Migration failed',
            'instance_id': instance_id
        }
        
        if not hasattr(restored_config, 'migration_history'):
            restored_config.migration_history = []
        
        restored_config.migration_history.append(rollback_record)
        
        return restored_config
    
    def _field_exists_in_dict(self, config_dict: Dict[str, Any], field_path: str) -> bool:
        """Check if field exists in nested dictionary"""
        
        parts = field_path.split('.')
        current = config_dict
        
        for part in parts:
            if isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return False
        
        return True
    
    def _validate_config_structure(self, config_dict: Dict[str, Any]) -> bool:
        """Validate configuration structure"""
        
        required_top_level = ['instance_id', 'user_id', 'instance_name']
        
        for field in required_top_level:
            if field not in config_dict:
                return False
        
        return True
    
    def _validate_schema_compatibility(self, config: ResearchInstanceConfig) -> bool:
        """Validate schema compatibility"""
        
        try:
            # Check if all enum fields are valid
            enum_fields = [
                ('research_mode', ResearchMode),
                ('source_trust_level', SourceTrustLevel),
                ('ai_processing_level', AIProcessingLevel),
                ('cache_strategy', CacheStrategy)
            ]
            
            for field_name, enum_class in enum_fields:
                if hasattr(config, field_name):
                    field_value = getattr(config, field_name)
                    if field_value is not None and not isinstance(field_value, enum_class):
                        if isinstance(field_value, str):
                            try:
                                enum_class(field_value)
                            except ValueError:
                                return False
                        else:
                            return False
            
            return True
            
        except Exception:
            return False
    
    async def get_migration_statistics(self) -> Dict[str, Any]:
        """Get migration statistics and health metrics"""
        
        return {
            'migration_stats': self._migration_stats.copy(),
            'current_version': self.CURRENT_CONFIG_VERSION,
            'supported_versions': self.SUPPORTED_VERSIONS.copy(),
            'migration_paths': self.MIGRATION_PATHS.copy(),
            'total_configs': len(self.instance_configs),
            'configs_by_version': self._get_config_version_distribution(),
            'migration_health': self._calculate_migration_health()
        }
    
    def _get_config_version_distribution(self) -> Dict[str, int]:
        """Get distribution of configuration versions"""
        
        version_counts = {}
        
        for config in self.instance_configs.values():
            version = getattr(config, 'config_version', 'unknown')
            version_counts[version] = version_counts.get(version, 0) + 1
        
        return version_counts
    
    def _calculate_migration_health(self) -> Dict[str, float]:
        """Calculate migration health metrics"""
        
        stats = self._migration_stats
        total = stats['total_migrations']
        
        if total == 0:
            return {
                'success_rate': 1.0,
                'failure_rate': 0.0,
                'rollback_rate': 0.0
            }
        
        return {
            'success_rate': stats['successful_migrations'] / total,
            'failure_rate': stats['failed_migrations'] / total,
            'rollback_rate': stats['rollbacks'] / total
        }