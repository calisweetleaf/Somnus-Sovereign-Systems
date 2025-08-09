"""
SOMNUS Memory System Settings
=============================

Comprehensive settings management for the persistent memory system.
Supports semantic indexing, encryption, retention policies, and cross-session continuity.

Architecture Integration:
- Memory Core: Semantic vector storage with ChromaDB and SentenceTransformers
- Memory Integration: Session-scoped memory context and cross-session continuity  
- Privacy Controls: User-scoped encryption with granular access controls
- Intelligent Retention: Importance-based forgetting and cleanup mechanisms
- Performance Optimization: Caching, monitoring, and resource management
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Union, Tuple
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator
from schemas.session import SessionID, UserID

logger = logging.getLogger(__name__)


# ================================
# Memory Configuration Models
# ================================

class MemoryType(str, Enum):
    """Memory classification for retrieval and retention strategies"""
    CORE_FACT = "core_fact"           # Persistent user facts (name, preferences)
    CONVERSATION = "conversation"      # Chat exchanges and context
    DOCUMENT = "document"             # Uploaded files and analysis
    CODE_SNIPPET = "code_snippet"    # Generated/executed code
    TOOL_RESULT = "tool_result"       # Plugin/tool outputs
    CUSTOM_INSTRUCTION = "custom_instruction"  # User-defined behaviors
    SYSTEM_EVENT = "system_event"     # Technical events and errors


class MemoryImportance(str, Enum):
    """Importance levels for retention and retrieval prioritization"""
    CRITICAL = "critical"    # Never forget (user identity, core preferences)
    HIGH = "high"           # Long-term retention (important facts, insights)
    MEDIUM = "medium"       # Medium-term retention (useful context)
    LOW = "low"            # Short-term retention (ephemeral interactions)
    TEMPORARY = "temporary" # Session-only (debugging, system messages)


class MemoryScope(str, Enum):
    """Memory access scope and sharing permissions"""
    PRIVATE = "private"     # User-only access
    SHARED = "shared"       # Shareable with other users (if permitted)
    SYSTEM = "system"       # System-wide (anonymized analytics)
    ARCHIVED = "archived"   # Compressed long-term storage


class EmbeddingModel(str, Enum):
    """Available embedding models for semantic search"""
    MINILM_L6_V2 = "sentence-transformers/all-MiniLM-L6-v2"          # Default, fast
    MPNET_BASE = "sentence-transformers/all-mpnet-base-v2"           # Better quality
    CODEBERT = "microsoft/codebert-base"                             # Code-optimized
    MULTILINGUAL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"  # Multi-language
    DISTILBERT = "sentence-transformers/all-distilroberta-v1"        # Balanced performance


class RetentionStrategy(str, Enum):
    """Memory retention and cleanup strategies"""
    IMPORTANCE_BASED = "importance_based"     # Retain based on importance scoring
    TIME_BASED = "time_based"                # Simple time-based expiration
    USAGE_BASED = "usage_based"              # Retain frequently accessed memories
    ADAPTIVE = "adaptive"                    # Learn from user patterns
    HYBRID = "hybrid"                        # Combine multiple strategies


class EncryptionLevel(str, Enum):
    """Encryption strength levels"""
    NONE = "none"           # No encryption (not recommended)
    BASIC = "basic"         # Standard encryption
    HIGH = "high"           # Enhanced encryption with stronger key derivation
    PARANOID = "paranoid"   # Maximum security with multiple layers


# ================================
# Core Settings Models
# ================================

@dataclass
class MemoryStorageSettings:
    """Core storage configuration for memory system"""
    # Database paths
    vector_db_path: str = "data/memory/vectors"
    metadata_db_path: str = "data/memory/metadata.db"
    backup_path: str = "data/memory/backups"
    
    # Storage limits
    max_memories_per_user: int = 50000
    max_storage_per_user_gb: float = 10.0
    max_document_size_mb: int = 100
    
    # Database optimization
    sqlite_wal_mode: bool = True              # Write-ahead logging
    sqlite_cache_size_mb: int = 64            # Cache size in MB
    sqlite_mmap_size_mb: int = 256            # Memory mapping size
    chromadb_persist: bool = True             # Persist vector database
    
    # Performance settings
    enable_compression: bool = True
    enable_deduplication: bool = True
    enable_archival: bool = True
    archival_threshold_days: int = 180


@dataclass
class MemoryEmbeddingSettings:
    """Semantic embedding and search configuration"""
    # Primary embedding model
    primary_model: EmbeddingModel = EmbeddingModel.MINILM_L6_V2
    model_cache_dir: str = "data/models/embeddings"
    
    # Specialized models for different content types
    code_model: EmbeddingModel = EmbeddingModel.CODEBERT
    multilingual_model: EmbeddingModel = EmbeddingModel.MULTILINGUAL
    
    # Processing settings
    batch_size: int = 32
    max_sequence_length: int = 512
    normalize_embeddings: bool = True
    
    # Performance optimization
    enable_fp16: bool = True                  # Half-precision for speed
    enable_onnx: bool = False                 # ONNX optimization (future)
    device_preference: List[str] = field(
        default_factory=lambda: ["cuda", "mps", "cpu"]
    )
    
    # Embedding generation
    auto_embedding: bool = True               # Generate embeddings automatically
    background_embedding: bool = True         # Generate embeddings in background
    embedding_cache_size: int = 5000          # Cache frequently used embeddings


@dataclass
class MemoryRetentionSettings:
    """Memory retention policies and lifecycle management"""
    # Retention strategy
    strategy: RetentionStrategy = RetentionStrategy.HYBRID
    
    # Retention periods by importance (days, -1 = never expire)
    retention_periods: Dict[MemoryImportance, int] = field(
        default_factory=lambda: {
            MemoryImportance.CRITICAL: -1,      # Never expire
            MemoryImportance.HIGH: 365,         # 1 year
            MemoryImportance.MEDIUM: 90,        # 3 months
            MemoryImportance.LOW: 30,           # 1 month
            MemoryImportance.TEMPORARY: 1       # 1 day
        }
    )
    
    # Advanced retention features
    adaptive_retention: bool = True           # Learn from user access patterns
    importance_decay: bool = True             # Gradually reduce importance over time
    redundancy_detection: bool = True         # Merge similar memories
    
    # Cleanup configuration
    cleanup_interval_hours: int = 6           # How often to run cleanup
    batch_cleanup_size: int = 1000           # Items to process per cleanup
    low_relevance_threshold: float = 0.3     # Threshold for deletion
    
    # Access pattern learning
    track_access_patterns: bool = True
    access_weight_factor: float = 0.3         # Weight of access frequency in retention
    recency_weight_factor: float = 0.3        # Weight of recency in retention
    importance_weight_factor: float = 0.4     # Weight of importance in retention


@dataclass
class MemoryPrivacySettings:
    """Privacy, encryption, and access control settings"""
    # Encryption configuration
    encryption_level: EncryptionLevel = EncryptionLevel.HIGH
    encryption_algorithm: str = "Fernet"      # Symmetric encryption
    key_derivation: str = "PBKDF2"            # Key derivation function
    key_iterations: int = 100000              # PBKDF2 iterations
    
    # User data isolation
    strict_user_isolation: bool = True        # Complete user data separation
    cross_user_sharing: bool = False          # Allow memory sharing between users
    anonymization_enabled: bool = False       # Anonymize for analytics
    
    # Data export and deletion
    export_format: str = "json"               # Default export format
    deletion_confirmation: bool = True        # Require confirmation for deletion
    secure_deletion: bool = True              # Overwrite deleted data
    
    # Audit and compliance
    audit_memory_access: bool = True          # Log all memory access
    audit_retention_days: int = 90            # Keep audit logs for 90 days
    gdpr_compliance: bool = True              # GDPR compliance features
    data_residency: str = "local"             # Keep all data local
    
    # Access controls
    require_user_consent: bool = True         # Require consent for memory storage
    allow_memory_export: bool = True          # Allow users to export their data
    allow_memory_import: bool = True          # Allow users to import data


@dataclass
class MemoryRetrievalSettings:
    """Semantic search and retrieval configuration"""
    # Search configuration
    similarity_threshold: float = 0.7         # Minimum similarity for retrieval
    max_results_per_query: int = 20           # Maximum results to return
    diversification_enabled: bool = True      # Diversify search results
    
    # Context enhancement
    auto_context_expansion: bool = True       # Automatically expand context
    max_context_memories: int = 10            # Maximum memories in context
    context_relevance_threshold: float = 0.6  # Threshold for context inclusion
    
    # Advanced search features
    temporal_weighting: bool = True           # Boost recent memories
    importance_weighting: bool = True         # Boost high-importance memories
    access_pattern_learning: bool = True      # Learn from user preferences
    
    # Hybrid search (semantic + keyword)
    hybrid_search_enabled: bool = True
    keyword_boost_factor: float = 0.3         # Weight of keyword matching
    semantic_boost_factor: float = 0.7        # Weight of semantic similarity
    
    # Performance optimization
    enable_search_cache: bool = True          # Cache search results
    cache_ttl_minutes: int = 30               # Cache time-to-live
    max_cached_queries: int = 1000            # Maximum cached queries


@dataclass
class MemoryClassificationSettings:
    """Automatic memory classification and tagging"""
    # Automatic classification
    auto_classification: bool = True
    classification_confidence_threshold: float = 0.8
    
    # Content detection patterns
    core_fact_patterns: List[str] = field(
        default_factory=lambda: [
            "my name is", "i work at", "i live in", "i prefer", "call me"
        ]
    )
    
    preference_patterns: List[str] = field(
        default_factory=lambda: [
            "i like", "i don't like", "i hate", "favorite", "prefer"
        ]
    )
    
    technical_patterns: List[str] = field(
        default_factory=lambda: [
            "code", "function", "algorithm", "programming", "debug"
        ]
    )
    
    creative_patterns: List[str] = field(
        default_factory=lambda: [
            "story", "poem", "creative", "artistic", "design"
        ]
    )
    
    # Tagging configuration
    auto_tagging: bool = True
    max_tags_per_memory: int = 10
    tag_similarity_threshold: float = 0.8
    
    # Content analysis
    extract_entities: bool = True              # Extract named entities
    extract_keywords: bool = True              # Extract important keywords
    sentiment_analysis: bool = True            # Analyze sentiment
    language_detection: bool = True            # Detect content language


@dataclass
class MemoryPerformanceSettings:
    """Performance optimization and monitoring settings"""
    # Caching strategies
    memory_cache_size: int = 1000             # In-memory cache size
    embedding_cache_size: int = 5000          # Embedding cache size
    cache_ttl_seconds: int = 3600             # Cache time-to-live
    
    # Background processing
    async_processing: bool = True             # Process memories asynchronously
    batch_processing_size: int = 100          # Batch size for operations
    max_concurrent_operations: int = 10       # Max concurrent operations
    
    # Memory usage optimization
    lazy_loading: bool = True                 # Load memories on demand
    memory_mapping: bool = True               # Use memory mapping for large files
    garbage_collection_threshold: int = 1000  # GC threshold
    
    # Monitoring and alerting
    performance_monitoring: bool = True       # Monitor performance metrics
    slow_query_threshold_ms: int = 1000       # Alert on slow queries
    memory_usage_alerting: bool = True        # Alert on high memory usage
    disk_usage_alerting: bool = True          # Alert on high disk usage
    
    # Statistics collection
    collect_usage_stats: bool = True          # Collect usage statistics
    stats_retention_days: int = 30            # Keep stats for 30 days
    anonymize_stats: bool = True              # Anonymize collected stats


@dataclass
class MemoryIntegrationSettings:
    """Integration with session management and other systems"""
    # Session integration
    auto_store_conversations: bool = True     # Automatically store conversations
    conversation_importance_threshold: MemoryImportance = MemoryImportance.LOW
    
    # Context injection
    inject_context_memories: bool = True      # Inject relevant memories into context
    max_context_tokens: int = 2048            # Maximum tokens for context
    context_injection_strategy: str = "importance_first"  # Context selection strategy
    
    # Cross-session continuity
    session_continuity: bool = True           # Enable cross-session continuity
    continuity_lookback_hours: int = 24       # Look back this many hours
    max_continuity_memories: int = 5          # Max memories for continuity
    
    # Memory triggers
    fact_extraction_enabled: bool = True      # Extract facts from conversations
    preference_learning_enabled: bool = True  # Learn user preferences
    tool_result_storage: bool = True          # Store tool/plugin results
    
    # External integrations
    web_search_integration: bool = True       # Store web search results
    file_upload_integration: bool = True      # Store uploaded file content
    artifact_integration: bool = True         # Store artifact interactions


@dataclass
class MemoryBackupSettings:
    """Backup and disaster recovery configuration"""
    # Automated backups
    enabled: bool = True
    backup_interval_hours: int = 6            # Backup every 6 hours
    retention_days: int = 30                  # Keep backups for 30 days
    
    # Backup types
    full_backup_weekly: bool = True           # Weekly full backup
    incremental_backup_daily: bool = True     # Daily incremental backup
    vector_db_backup: bool = True             # Backup vector database
    metadata_backup: bool = True              # Backup metadata database
    
    # Recovery procedures
    recovery_testing: bool = True             # Test recovery procedures
    restoration_verification: bool = True     # Verify restored data
    data_consistency_checks: bool = True      # Check data consistency
    
    # External backup storage
    cloud_backup: bool = False                # Privacy-first: no cloud backup
    local_backup_path: str = "data/backups"   # Local backup location
    compression_enabled: bool = True          # Compress backups
    encryption_enabled: bool = True           # Encrypt backups


# ================================
# Main Settings Manager
# ================================

class MemorySettingsManager:
    """Manages all memory system settings with validation and persistence"""
    
    def __init__(self, user_id: UserID):
        self.user_id = user_id
        self.settings_id = uuid4()
        self.last_updated = datetime.now(timezone.utc)
        self.is_active = False
        
        # Initialize default settings
        self.storage = MemoryStorageSettings()
        self.embedding = MemoryEmbeddingSettings()
        self.retention = MemoryRetentionSettings()
        self.privacy = MemoryPrivacySettings()
        self.retrieval = MemoryRetrievalSettings()
        self.classification = MemoryClassificationSettings()
        self.performance = MemoryPerformanceSettings()
        self.integration = MemoryIntegrationSettings()
        self.backup = MemoryBackupSettings()
        
        # System components
        self._memory_manager = None
        self._session_manager = None
        self._background_tasks = []
        
        logger.info(f"Memory settings initialized for user {user_id}")
    
    async def activate_memory_system(self) -> bool:
        """Activate the memory system with current settings"""
        try:
            if self.is_active:
                logger.warning("Memory system already active")
                return True
            
            # Initialize memory manager
            await self._initialize_memory_manager()
            
            # Initialize session integration
            await self._initialize_session_integration()
            
            # Start background tasks
            await self._start_background_tasks()
            
            self.is_active = True
            logger.info("Memory system activated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate memory system: {e}")
            return False
    
    async def deactivate_memory_system(self) -> bool:
        """Gracefully deactivate the memory system"""
        try:
            if not self.is_active:
                return True
            
            # Stop background tasks
            for task in self._background_tasks:
                task.cancel()
            
            # Shutdown memory manager
            if self._memory_manager:
                await self._memory_manager.shutdown()
            
            self.is_active = False
            logger.info("Memory system deactivated")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating memory system: {e}")
            return False
    
    async def update_storage_settings(self, **kwargs) -> bool:
        """Update storage configuration settings"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.storage, key):
                    setattr(self.storage, key, value)
            
            # Apply changes if system is active
            if self.is_active and self._memory_manager:
                await self._memory_manager.update_storage_config(self.storage)
            
            self.last_updated = datetime.now(timezone.utc)
            logger.info("Storage settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update storage settings: {e}")
            return False
    
    async def update_retention_settings(self, **kwargs) -> bool:
        """Update memory retention settings"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.retention, key):
                    setattr(self.retention, key, value)
            
            # Apply changes if system is active
            if self.is_active and self._memory_manager:
                await self._memory_manager.update_retention_policies(self.retention)
            
            self.last_updated = datetime.now(timezone.utc)
            logger.info("Retention settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update retention settings: {e}")
            return False
    
    async def update_privacy_settings(self, **kwargs) -> bool:
        """Update privacy and encryption settings"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.privacy, key):
                    setattr(self.privacy, key, value)
            
            # Privacy changes may require system restart
            if self.is_active:
                logger.warning("Privacy setting changes require system restart to take effect")
            
            self.last_updated = datetime.now(timezone.utc)
            logger.info("Privacy settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update privacy settings: {e}")
            return False
    
    async def update_retrieval_settings(self, **kwargs) -> bool:
        """Update search and retrieval settings"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.retrieval, key):
                    setattr(self.retrieval, key, value)
            
            # Apply changes if system is active
            if self.is_active and self._memory_manager:
                await self._memory_manager.update_retrieval_config(self.retrieval)
            
            self.last_updated = datetime.now(timezone.utc)
            logger.info("Retrieval settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update retrieval settings: {e}")
            return False
    
    async def _initialize_memory_manager(self):
        """Initialize the memory manager with current settings"""
        from memory_core import MemoryManager, MemoryConfiguration
        
        # Create configuration from settings
        config = MemoryConfiguration(
            vector_db_path=self.storage.vector_db_path,
            metadata_db_path=self.storage.metadata_db_path,
            max_memories_per_user=self.storage.max_memories_per_user,
            embedding_model=self.embedding.primary_model.value,
            retention_policies={
                k: v for k, v in self.retention.retention_periods.items()
            },
            encryption_enabled=(self.privacy.encryption_level != EncryptionLevel.NONE),
            similarity_threshold=self.retrieval.similarity_threshold,
            max_retrieval_results=self.retrieval.max_results_per_query
        )
        
        # Initialize memory manager
        self._memory_manager = MemoryManager(config)
        await self._memory_manager.initialize()
        
        logger.info("Memory manager initialized")
    
    async def _initialize_session_integration(self):
        """Initialize session-memory integration"""
        if self.integration.inject_context_memories:
            from memory_integration import create_memory_enhanced_session_manager
            
            # Create enhanced session manager
            # This would integrate with the main session manager
            logger.info("Session-memory integration initialized")
    
    async def _start_background_tasks(self):
        """Start background maintenance tasks"""
        if self.retention.cleanup_interval_hours > 0:
            cleanup_task = asyncio.create_task(self._cleanup_loop())
            self._background_tasks.append(cleanup_task)
        
        if self.backup.enabled:
            backup_task = asyncio.create_task(self._backup_loop())
            self._background_tasks.append(backup_task)
        
        if self.performance.performance_monitoring:
            monitor_task = asyncio.create_task(self._monitoring_loop())
            self._background_tasks.append(monitor_task)
        
        logger.info(f"Started {len(self._background_tasks)} background tasks")
    
    async def _cleanup_loop(self):
        """Background cleanup task"""
        while True:
            try:
                await asyncio.sleep(self.retention.cleanup_interval_hours * 3600)
                
                if self._memory_manager:
                    await self._memory_manager.cleanup_expired_memories()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup task error: {e}")
    
    async def _backup_loop(self):
        """Background backup task"""
        while True:
            try:
                await asyncio.sleep(self.backup.backup_interval_hours * 3600)
                
                if self._memory_manager:
                    await self._memory_manager.create_backup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Backup task error: {e}")
    
    async def _monitoring_loop(self):
        """Background performance monitoring task"""
        while True:
            try:
                await asyncio.sleep(300)  # Monitor every 5 minutes
                
                if self._memory_manager:
                    metrics = await self._memory_manager.get_performance_metrics()
                    # Log or alert on metrics as needed
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring task error: {e}")
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get all current settings as dictionary"""
        return {
            'settings_id': str(self.settings_id),
            'user_id': str(self.user_id),
            'last_updated': self.last_updated.isoformat(),
            'is_active': self.is_active,
            'storage': self.storage.__dict__,
            'embedding': self.embedding.__dict__,
            'retention': self.retention.__dict__,
            'privacy': self.privacy.__dict__,
            'retrieval': self.retrieval.__dict__,
            'classification': self.classification.__dict__,
            'performance': self.performance.__dict__,
            'integration': self.integration.__dict__,
            'backup': self.backup.__dict__
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current memory system status and metrics"""
        status = {
            'memory_system_active': self.is_active,
            'memory_manager_ready': self._memory_manager is not None,
            'background_tasks_running': len(self._background_tasks),
            'last_settings_update': self.last_updated.isoformat()
        }
        
        # Add performance metrics if system is active
        if self.is_active and self._memory_manager:
            # This would call the actual memory manager for real metrics
            status.update({
                'total_memories': 0,  # Would be populated with real data
                'memory_usage_mb': 0,
                'cache_hit_ratio': 0.0,
                'average_retrieval_time_ms': 0.0
            })
        
        return status


# ================================
# Factory Functions
# ================================

async def create_memory_settings(user_id: UserID) -> MemorySettingsManager:
    """Create and initialize memory settings for a user"""
    settings = MemorySettingsManager(user_id)
    await settings.activate_memory_system()
    return settings


def get_default_memory_settings() -> Dict[str, Any]:
    """Get default memory system settings"""
    default_settings = MemorySettingsManager("default")
    return default_settings.get_current_settings()


# ================================
# Settings Validation
# ================================

def validate_memory_settings(settings_dict: Dict[str, Any]) -> List[str]:
    """Validate memory settings configuration"""
    errors = []
    
    # Validate storage limits
    storage = settings_dict.get('storage', {})
    if storage.get('max_memories_per_user', 0) > 100000:
        errors.append("Maximum memories per user cannot exceed 100,000")
    
    if storage.get('max_storage_per_user_gb', 0) > 100:
        errors.append("Maximum storage per user cannot exceed 100GB")
    
    # Validate retention periods
    retention = settings_dict.get('retention', {})
    retention_periods = retention.get('retention_periods', {})
    for importance, days in retention_periods.items():
        if days > 3650 and days != -1:  # 10 years max
            errors.append(f"Retention period for {importance} cannot exceed 10 years")
    
    # Validate performance settings
    performance = settings_dict.get('performance', {})
    if performance.get('max_concurrent_operations', 0) > 50:
        errors.append("Maximum concurrent operations cannot exceed 50")
    
    # Validate retrieval settings
    retrieval = settings_dict.get('retrieval', {})
    if retrieval.get('max_results_per_query', 0) > 100:
        errors.append("Maximum results per query cannot exceed 100")
    
    return errors


# ================================
# Example Usage
# ================================

async def example_memory_settings_setup():
    """Example of setting up memory system settings"""
    
    # Create settings for a user
    user_id = "user_123"
    settings = await create_memory_settings(user_id)
    
    # Customize storage settings
    await settings.update_storage_settings(
        max_memories_per_user=25000,
        enable_compression=True,
        enable_archival=True
    )
    
    # Customize retention for long-term memory
    await settings.update_retention_settings(
        strategy=RetentionStrategy.ADAPTIVE,
        adaptive_retention=True,
        importance_decay=True
    )
    
    # Enhance privacy settings
    await settings.update_privacy_settings(
        encryption_level=EncryptionLevel.HIGH,
        strict_user_isolation=True,
        audit_memory_access=True
    )
    
    # Optimize retrieval for better results
    await settings.update_retrieval_settings(
        similarity_threshold=0.75,
        hybrid_search_enabled=True,
        temporal_weighting=True
    )
    
    # Check system status
    status = settings.get_system_status()
    print(f"Memory system active: {status['memory_system_active']}")
    
    return settings


if __name__ == "__main__":
    # Example usage
    import asyncio
    asyncio.run(example_memory_settings_setup())