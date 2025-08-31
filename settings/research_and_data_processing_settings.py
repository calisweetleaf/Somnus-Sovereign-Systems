"""
SOMNUS Research & Data Processing Settings
==========================================

Comprehensive settings management for research capabilities and data processing systems.
Supports AI browser research, web search, file processing, and privacy-first data handling.

Architecture Integration:
- AI Browser Research: Personal browser in AI's persistent VM
- Web Search: Privacy-first multi-provider search with deep research
- File Processing: GGUF embeddings, OCR, multi-modal processing  
- Accelerated Processing: Priority queues with adaptive resource management
- UI Integration: Progressive enhancement with on-demand activation
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
# Research Configuration Models
# ================================

class ResearchMode(str, Enum):
    """Research depth and methodology"""
    QUICK = "quick"           # Single-layer search, basic sources
    STANDARD = "standard"     # Two-layer research with credibility scoring
    DEEP = "deep"            # Three-layer research with fact-checking
    COMPREHENSIVE = "comprehensive"  # All capabilities + cross-validation


class SearchProvider(str, Enum):
    """Privacy-first search providers"""
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    BRAVE = "brave"
    STARTPAGE = "startpage"
    YANDEX = "yandex"          # For diverse perspectives
    BING = "bing"              # Fallback only


class BrowserCapability(str, Enum):
    """AI browser research capabilities"""
    VISUAL_BROWSING = "visual_browsing"      # Screenshots, visual analysis
    FORM_FILLING = "form_filling"            # Auto-fill research forms
    EXTENSION_INSTALL = "extension_install"   # Browser extension management
    AUTOMATION_SCRIPTS = "automation_scripts" # Custom research automation
    BOOKMARK_MANAGEMENT = "bookmark_management" # Research bookmark system
    MULTI_TAB_RESEARCH = "multi_tab_research"   # Parallel source analysis


class FileProcessingMode(str, Enum):
    """File processing strategies"""
    IMMEDIATE = "immediate"      # Process files immediately on upload
    QUEUED = "queued"           # Add to processing queue
    BATCH = "batch"             # Batch process during low usage
    ON_DEMAND = "on_demand"     # Process only when accessed


class ProcessingPriority(IntEnum):
    """File processing priority levels"""
    CRITICAL = 1    # User-requested immediate processing
    HIGH = 2        # Code/documentation files
    MEDIUM = 3      # Configuration/data files
    LOW = 4         # Images/media files
    BACKGROUND = 5  # Large repositories/batch operations


# ================================
# Core Settings Models
# ================================

@dataclass
class AIBrowserSettings:
    """AI's personal browser research configuration"""
    # Browser capabilities
    enabled: bool = True
    auto_trigger_research: bool = True  # AI can initiate browser research
    enabled_capabilities: Set[BrowserCapability] = field(
        default_factory=lambda: {
            BrowserCapability.VISUAL_BROWSING,
            BrowserCapability.BOOKMARK_MANAGEMENT,
            BrowserCapability.MULTI_TAB_RESEARCH
        }
    )
    
    # Browser configuration
    default_browser: str = "firefox"  # In AI's VM
    headless_mode: bool = False       # AI can see browser for visual research
    screenshot_quality: int = 85      # JPEG quality for research screenshots
    max_tabs: int = 10               # Maximum concurrent research tabs
    page_load_timeout: int = 30      # Seconds to wait for page loads
    
    # Research automation
    save_research_bookmarks: bool = True    # Auto-save valuable sources
    create_automation_scripts: bool = True  # AI can build custom tools
    research_session_recording: bool = True # Record research methodology
    
    # Visual analysis
    take_screenshots: bool = True           # Visual documentation
    analyze_page_layouts: bool = True       # AI understands visual structure
    extract_visual_text: bool = True        # OCR on complex pages
    
    # Security and privacy
    clear_cache_after_session: bool = True  # Privacy protection
    disable_tracking: bool = True            # Block tracking scripts
    use_privacy_extensions: bool = True      # Install privacy tools
    
    # Storage paths in AI's VM
    screenshots_dir: str = "/home/ai/research/screenshots"
    bookmarks_file: str = "/home/ai/research/bookmarks.json"
    automation_scripts_dir: str = "/home/ai/tools/research_automation"


@dataclass
class WebSearchSettings:
    """Privacy-first web search configuration"""
    # Core search settings
    enabled: bool = True
    default_provider: SearchProvider = SearchProvider.DUCKDUCKGO
    backup_providers: List[SearchProvider] = field(
        default_factory=lambda: [SearchProvider.SEARXNG, SearchProvider.BRAVE]
    )
    
    # Search behavior
    max_results_per_query: int = 20
    max_search_depth: int = 3           # Research levels
    enable_parallel_search: bool = True  # Multi-provider simultaneously
    result_caching: bool = True         # Cache for 24 hours
    
    # Privacy protection
    safe_search: bool = True
    tracking_protection: bool = True
    content_filtering: bool = True
    use_vpn_rotation: bool = False      # Advanced privacy (optional)
    
    # Quality control
    source_credibility_scoring: bool = True
    bias_detection: bool = True
    fact_checking: bool = True
    cross_reference_validation: bool = True
    
    # Research methodology
    default_research_mode: ResearchMode = ResearchMode.STANDARD
    academic_sources_preferred: bool = True
    news_sources_recency_hours: int = 72    # Prefer recent news
    technical_documentation_priority: bool = True
    
    # Rate limiting and performance
    requests_per_minute: int = 60
    concurrent_requests: int = 5
    request_timeout: int = 30
    retry_attempts: int = 3
    
    # Content processing
    extract_full_content: bool = True       # Not just snippets
    summarize_long_articles: bool = True
    extract_key_quotes: bool = True
    identify_expert_opinions: bool = True


@dataclass
class FileProcessingSettings:
    """Comprehensive file processing configuration"""
    # Core processing settings
    enabled: bool = True
    processing_mode: FileProcessingMode = FileProcessingMode.QUEUED
    default_priority: ProcessingPriority = ProcessingPriority.MEDIUM
    
    # File size and type limits
    max_file_size_mb: int = 100
    max_batch_size: int = 50        # Files per batch
    allowed_extensions: Set[str] = field(
        default_factory=lambda: {
            '.txt', '.md', '.pdf', '.docx', '.doc', '.jpg', '.jpeg', '.png', 
            '.py', '.js', '.html', '.css', '.json', '.csv', '.xlsx', '.xml'
        }
    )
    
    # Processing capabilities
    enable_ocr: bool = True         # Image text extraction
    enable_gguf_embeddings: bool = True
    enable_content_summarization: bool = True
    enable_semantic_chunking: bool = True
    
    # OCR configuration
    ocr_languages: List[str] = field(
        default_factory=lambda: ["eng", "spa", "fra", "deu"]
    )
    ocr_confidence_threshold: float = 0.7
    
    # Content extraction
    extract_metadata: bool = True
    extract_tables: bool = True
    extract_images: bool = True
    preserve_formatting: bool = True
    
    # Security validation
    virus_scanning: bool = True
    content_safety_check: bool = True
    executable_blocking: bool = True
    
    # Storage and indexing
    storage_path: str = "data/uploads"
    create_thumbnails: bool = True
    index_content: bool = True
    generate_embeddings: bool = True


@dataclass
class AcceleratedProcessingSettings:
    """Advanced processing queue and resource management"""
    # Queue management
    enabled: bool = True
    max_queue_size: int = 1000
    max_concurrent_tasks: int = 8       # Auto-adjusted based on system
    enable_priority_queue: bool = True
    
    # Resource monitoring
    enable_resource_monitoring: bool = True
    cpu_threshold_percent: float = 85.0  # Scale back processing if exceeded
    memory_threshold_percent: float = 80.0
    disk_threshold_percent: float = 90.0
    
    # Adaptive scaling
    auto_scale_threads: bool = True
    min_threads: int = 2
    max_threads: int = 16
    scale_up_threshold: float = 0.8     # Queue utilization to scale up
    scale_down_threshold: float = 0.3   # Queue utilization to scale down
    
    # Error handling and retries
    enable_retry_logic: bool = True
    max_retries: int = 3
    retry_backoff_multiplier: float = 2.0
    max_retry_delay: int = 60           # seconds
    
    # Performance optimization
    batch_similar_files: bool = True    # Group similar file types
    cache_frequent_operations: bool = True
    preload_models: bool = True         # Keep GGUF models in memory
    
    # Monitoring and statistics
    track_processing_stats: bool = True
    log_performance_metrics: bool = True
    alert_on_failures: bool = True
    
    # GitHub integration settings
    github_batch_priority: ProcessingPriority = ProcessingPriority.BACKGROUND
    github_max_files_per_repo: int = 500
    github_skip_large_files_mb: int = 50


@dataclass
class ResearchDataUISettings:
    """UI integration and user experience settings"""
    # Progressive enhancement
    enable_overlay_ui: bool = True      # Slide-in interfaces
    auto_hide_inactive: bool = True     # Hide when not in use
    animation_duration_ms: int = 300    # Smooth transitions
    
    # Research interface
    show_research_progress: bool = True  # Real-time research updates
    display_source_credibility: bool = True
    show_fact_check_results: bool = True
    enable_research_bookmarks: bool = True
    
    # File processing UI
    show_upload_progress: bool = True
    display_queue_status: bool = True
    show_processing_stats: bool = True
    enable_batch_upload: bool = True
    
    # Browser research UI
    show_browser_screenshots: bool = True
    display_automation_status: bool = True
    show_bookmark_manager: bool = True
    
    # Notifications and alerts
    notify_research_complete: bool = True
    notify_file_processed: bool = True
    alert_on_processing_errors: bool = True
    desktop_notifications: bool = False  # Browser notifications only


# ================================
# Main Settings Manager
# ================================

class ResearchDataSettingsManager:
    """Manages all research and data processing settings with validation and persistence"""
    
    def __init__(self, user_id: UserID):
        self.user_id = user_id
        self.settings_id = uuid4()
        self.last_updated = datetime.now(timezone.utc)
        self.is_active = False
        
        # Initialize default settings
        self.ai_browser = AIBrowserSettings()
        self.web_search = WebSearchSettings()
        self.file_processing = FileProcessingSettings()
        self.accelerated_processing = AcceleratedProcessingSettings()
        self.ui_settings = ResearchDataUISettings()
        
        # Integration state
        self._browser_system = None
        self._search_system = None
        self._file_manager = None
        self._accelerated_processor = None
        
        logger.info(f"Research & Data settings initialized for user {user_id}")
    
    async def activate_research_systems(self) -> bool:
        """Activate all research and data processing systems"""
        try:
            if self.is_active:
                logger.warning("Research systems already active")
                return True
            
            # Activate AI browser research
            if self.ai_browser.enabled:
                await self._activate_ai_browser()
            
            # Activate web search system
            if self.web_search.enabled:
                await self._activate_web_search()
            
            # Activate file processing
            if self.file_processing.enabled:
                await self._activate_file_processing()
            
            # Activate accelerated processing
            if self.accelerated_processing.enabled:
                await self._activate_accelerated_processing()
            
            self.is_active = True
            logger.info("Research & Data systems activated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to activate research systems: {e}")
            return False
    
    async def deactivate_research_systems(self) -> bool:
        """Gracefully deactivate all systems"""
        try:
            if not self.is_active:
                return True
            
            # Deactivate systems in reverse order
            if self._accelerated_processor:
                await self._accelerated_processor.stop()
            
            if self._file_manager:
                await self._file_manager.shutdown()
            
            if self._search_system:
                await self._search_system.cleanup()
            
            if self._browser_system:
                await self._browser_system.close_browser()
            
            self.is_active = False
            logger.info("Research & Data systems deactivated")
            return True
            
        except Exception as e:
            logger.error(f"Error deactivating research systems: {e}")
            return False
    
    async def update_ai_browser_settings(self, **kwargs) -> bool:
        """Update AI browser research settings"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.ai_browser, key):
                    setattr(self.ai_browser, key, value)
            
            # Restart browser system if active
            if self.is_active and self._browser_system:
                await self._browser_system.apply_settings(self.ai_browser)
            
            self.last_updated = datetime.now(timezone.utc)
            logger.info("AI browser settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update AI browser settings: {e}")
            return False
    
    async def update_web_search_settings(self, **kwargs) -> bool:
        """Update web search settings"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.web_search, key):
                    setattr(self.web_search, key, value)
            
            # Apply to active search system
            if self.is_active and self._search_system:
                await self._search_system.update_config(self.web_search)
            
            self.last_updated = datetime.now(timezone.utc)
            logger.info("Web search settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update web search settings: {e}")
            return False
    
    async def update_file_processing_settings(self, **kwargs) -> bool:
        """Update file processing settings"""
        try:
            for key, value in kwargs.items():
                if hasattr(self.file_processing, key):
                    setattr(self.file_processing, key, value)
            
            # Apply to active file manager
            if self.is_active and self._file_manager:
                await self._file_manager.update_settings(self.file_processing)
            
            self.last_updated = datetime.now(timezone.utc)
            logger.info("File processing settings updated")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update file processing settings: {e}")
            return False
    
    async def _activate_ai_browser(self):
        """Initialize AI browser research system"""
        from ai_browser_research_system import AIBrowserResearch
        from vm_manager import get_ai_vm_instance
        
        # Get AI's VM instance
        ai_vm = await get_ai_vm_instance(self.user_id)
        
        # Initialize browser system
        self._browser_system = AIBrowserResearch(ai_vm)
        await self._browser_system.setup_browser_environment(self.ai_browser)
        
        logger.info("AI browser research system activated")
    
    async def _activate_web_search(self):
        """Initialize web search research system"""
        from web_search_research import WebSearchEngine, DeepResearchEngine
        
        # Initialize search engines
        self._search_system = DeepResearchEngine(
            search_engine=WebSearchEngine(self.web_search),
            config=self.web_search
        )
        
        logger.info("Web search system activated")
    
    async def _activate_file_processing(self):
        """Initialize file processing system"""
        from file_upload_system import FileUploadManager
        
        # Initialize file manager
        self._file_manager = FileUploadManager(
            upload_dir=self.file_processing.storage_path,
            max_file_size_mb=self.file_processing.max_file_size_mb,
            allowed_types=list(self.file_processing.allowed_extensions)
        )
        
        logger.info("File processing system activated")
    
    async def _activate_accelerated_processing(self):
        """Initialize accelerated processing system"""
        from accelerated_file_processing import IntelligentFileProcessor
        
        # Initialize accelerated processor
        self._accelerated_processor = IntelligentFileProcessor(
            base_file_manager=self._file_manager,
            max_concurrent_tasks=self.accelerated_processing.max_concurrent_tasks,
            max_queue_size=self.accelerated_processing.max_queue_size,
            enable_monitoring=self.accelerated_processing.enable_resource_monitoring
        )
        
        await self._accelerated_processor.start()
        logger.info("Accelerated processing system activated")
    
    def get_current_settings(self) -> Dict[str, Any]:
        """Get all current settings as dictionary"""
        return {
            'settings_id': str(self.settings_id),
            'user_id': str(self.user_id),
            'last_updated': self.last_updated.isoformat(),
            'is_active': self.is_active,
            'ai_browser': self.ai_browser.__dict__,
            'web_search': self.web_search.__dict__,
            'file_processing': self.file_processing.__dict__,
            'accelerated_processing': self.accelerated_processing.__dict__,
            'ui_settings': self.ui_settings.__dict__
        }
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get current system status and performance metrics"""
        status = {
            'research_systems_active': self.is_active,
            'browser_ready': self._browser_system is not None,
            'search_ready': self._search_system is not None,
            'file_processing_ready': self._file_manager is not None,
            'accelerated_processing_ready': self._accelerated_processor is not None
        }
        
        # Add performance metrics if systems are active
        if self.is_active and self._accelerated_processor:
            queue_status = self._accelerated_processor.get_queue_status()
            status.update({
                'queue_size': queue_status['queue_size'],
                'active_tasks': queue_status['active_tasks'],
                'throughput_per_minute': queue_status['statistics']['throughput_files_per_minute'],
                'success_rate': (
                    queue_status['statistics']['total_processed'] / 
                    max(1, queue_status['statistics']['total_processed'] + queue_status['statistics']['total_failed'])
                ) * 100
            })
        
        return status


# ================================
# Factory Functions
# ================================

async def create_research_data_settings(user_id: UserID) -> ResearchDataSettingsManager:
    """Create and initialize research & data settings for a user"""
    settings = ResearchDataSettingsManager(user_id)
    await settings.activate_research_systems()
    return settings


def get_default_research_settings() -> Dict[str, Any]:
    """Get default research and data processing settings"""
    default_settings = ResearchDataSettingsManager("default")
    return default_settings.get_current_settings()


# ================================
# Settings Validation
# ================================

def validate_research_settings(settings_dict: Dict[str, Any]) -> List[str]:
    """Validate research settings configuration"""
    errors = []
    
    # Validate file size limits
    if settings_dict.get('file_processing', {}).get('max_file_size_mb', 0) > 500:
        errors.append("Maximum file size cannot exceed 500MB")
    
    # Validate queue size
    if settings_dict.get('accelerated_processing', {}).get('max_queue_size', 0) > 5000:
        errors.append("Maximum queue size cannot exceed 5000 items")
    
    # Validate thread limits
    max_threads = settings_dict.get('accelerated_processing', {}).get('max_threads', 0)
    if max_threads > 32:
        errors.append("Maximum thread count cannot exceed 32")
    
    # Validate search depth
    search_depth = settings_dict.get('web_search', {}).get('max_search_depth', 0)
    if search_depth > 5:
        errors.append("Maximum search depth cannot exceed 5 levels")
    
    return errors


# ================================
# Example Usage
# ================================

async def example_research_data_setup():
    """Example of setting up research and data processing"""
    
    # Create settings for a user
    user_id = "user_123"
    settings = await create_research_data_settings(user_id)
    
    # Customize AI browser settings
    await settings.update_ai_browser_settings(
        auto_trigger_research=True,
        take_screenshots=True,
        create_automation_scripts=True
    )
    
    # Customize search settings for deep research
    await settings.update_web_search_settings(
        default_research_mode=ResearchMode.DEEP,
        fact_checking=True,
        cross_reference_validation=True
    )
    
    # Optimize file processing for performance
    await settings.update_file_processing_settings(
        processing_mode=FileProcessingMode.QUEUED,
        enable_gguf_embeddings=True,
        enable_semantic_chunking=True
    )
    
    # Check system status
    status = settings.get_system_status()
    print(f"Research systems active: {status['research_systems_active']}")
    
    return settings


if __name__ == "__main__":
    # Example usage
    import asyncio
    asyncio.run(example_research_data_setup())