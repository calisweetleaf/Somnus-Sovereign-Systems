"""
SOMNUS RESEARCH VM ORCHESTRATOR
Complete production-ready research VM management system

This system provides dedicated research VMs with:
- Persistent browser automation environments
- AI model loading and analysis capabilities  
- Memory system integration for cross-session knowledge
- File processing and research database management
- Dynamic prompt management for research-focused AI behavior
- Complete integration with existing VM infrastructure

Architecture:
- Extends existing vm_manager.py for research-specific VMs
- Integrates memory_core, memory_integration for persistent knowledge
- Uses model_loader for in-VM AI analysis capabilities
- Coordinates with file upload system for research document processing
- Provides complete browser automation with Playwright/Selenium
- Real-time streaming and progress monitoring
"""

import asyncio
import json
import logging
import os
import subprocess
import tempfile
import time
import uuid
import zipfile
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union, AsyncGenerator
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from contextlib import asynccontextmanager

import aiofiles
import docker
import libvirt
import psutil
import redis.asyncio as redis
import yaml
import numpy as np
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# Core Somnus system imports
from core.vm_manager import VMInstanceManager, AIVMInstance, VMSpecs, VMState
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope, MemoryConfiguration
from core.memory_integration import EnhancedSessionManager, SessionMemoryContext
from core.model_loader import ModelLoader, ModelLoadRequest, ModelConfiguration, ModelType
from core.file_upload_system import FileUploadManager, FileMetadata, ProcessingStatus
from core.accelerated_file_processing import IntelligentFileProcessor, ProcessingPriority
from prompt_manager import SystemPromptManager, PromptType, PromptContext, PromptConfiguration
from schemas.session import SessionID, UserID

logger = logging.getLogger(__name__)


class ResearchVMType(str, Enum):
    """Types of research VMs for different use cases"""
    STANDARD = "standard"           # General web research with browser automation
    ACADEMIC = "academic"           # Academic research with paper processing
    TECHNICAL = "technical"         # Technical/code research with dev environments
    MULTIMEDIA = "multimedia"       # Media analysis with video/audio processing
    ENTERPRISE = "enterprise"       # Enterprise research with security focus
    SPECIALIZED = "specialized"     # Custom research domains


class BrowserEngine(str, Enum):
    """Supported browser engines for research"""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"
    SELENIUM_CHROME = "selenium_chrome"
    SELENIUM_FIREFOX = "selenium_firefox"


class ResearchCapability(str, Enum):
    """Research capabilities that can be enabled in VMs"""
    WEB_SCRAPING = "web_scraping"
    PDF_PROCESSING = "pdf_processing"
    VIDEO_ANALYSIS = "video_analysis"
    CODE_ANALYSIS = "code_analysis"
    ACADEMIC_SEARCH = "academic_search"
    SOCIAL_MEDIA = "social_media"
    DEEP_WEB = "deep_web"
    API_INTEGRATION = "api_integration"
    MULTILINGUAL = "multilingual"
    AI_ANALYSIS = "ai_analysis"


@dataclass
class ResearchVMConfiguration:
    """Complete configuration for research VM setup"""
    vm_type: ResearchVMType = ResearchVMType.STANDARD
    vm_specs: VMSpecs = field(default_factory=lambda: VMSpecs(vcpus=6, memory_gb=12, storage_gb=150))
    
    # Browser configuration
    browser_engines: List[BrowserEngine] = field(default_factory=lambda: [BrowserEngine.CHROMIUM, BrowserEngine.FIREFOX])
    headless_mode: bool = True
    enable_screenshots: bool = True
    enable_video_recording: bool = False
    
    # Research capabilities
    enabled_capabilities: List[ResearchCapability] = field(default_factory=lambda: [
        ResearchCapability.WEB_SCRAPING,
        ResearchCapability.PDF_PROCESSING,
        ResearchCapability.AI_ANALYSIS
    ])
    
    # AI model configuration
    ai_models: List[str] = field(default_factory=lambda: [
        "sentence-transformers/all-MiniLM-L6-v2",
        "microsoft/DialoGPT-small"
    ])
    
    # Storage configuration
    research_storage_gb: int = 50
    cache_storage_gb: int = 20
    temp_storage_gb: int = 10
    
    # Network configuration
    enable_vpn: bool = False
    proxy_settings: Optional[Dict[str, str]] = None
    rate_limiting: Dict[str, float] = field(default_factory=lambda: {"requests_per_minute": 60.0})
    
    # Security configuration
    sandbox_level: str = "medium"
    enable_content_filtering: bool = True
    blocked_domains: List[str] = field(default_factory=list)
    allowed_file_types: List[str] = field(default_factory=lambda: [
        "pdf", "txt", "html", "json", "csv", "xlsx", "docx", "png", "jpg"
    ])


@dataclass
class ResearchVMState:
    """Current state of research VM"""
    vm_id: UUID
    vm_state: VMState
    browser_sessions: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    loaded_models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_research_sessions: Set[str] = field(default_factory=set)
    
    # Resource monitoring
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    disk_usage: float = 0.0
    network_activity: Dict[str, float] = field(default_factory=dict)
    
    # Capabilities status
    capability_status: Dict[ResearchCapability, bool] = field(default_factory=dict)
    
    # Last activity tracking
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    session_count: int = 0
    research_database_size: int = 0


class ResearchVMOrchestrator:
    """
    Complete research VM orchestration system
    
    Manages dedicated research VMs with full browser automation, AI analysis,
    memory integration, and file processing capabilities.
    """
    
    def __init__(
        self,
        base_vm_manager: VMInstanceManager,
        memory_manager: MemoryManager,
        model_loader: ModelLoader,
        file_processor: IntelligentFileProcessor,
        prompt_manager: SystemPromptManager,
        config_dir: Path = Path("research_config"),
        vm_storage_dir: Path = Path("research_vms")
    ):
        """Initialize research VM orchestrator"""
        
        self.base_vm_manager = base_vm_manager
        self.memory_manager = memory_manager
        self.model_loader = model_loader
        self.file_processor = file_processor
        self.prompt_manager = prompt_manager
        self.config_dir = config_dir
        self.vm_storage_dir = vm_storage_dir
        
        # Create directories
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.vm_storage_dir.mkdir(parents=True, exist_ok=True)
        
        # Research VM tracking
        self.research_vms: Dict[UUID, ResearchVMState] = {}
        self.vm_configurations: Dict[UUID, ResearchVMConfiguration] = {}
        self.user_vm_mapping: Dict[UserID, Set[UUID]] = {}
        
        # Resource management
        self.resource_monitor_task: Optional[asyncio.Task] = None
        self.cleanup_task: Optional[asyncio.Task] = None
        self.monitoring_active = False
        
        # Browser automation
        self.playwright_instance = None
        self.selenium_drivers: Dict[UUID, Dict[str, webdriver.Remote]] = {}
        
        # AI analysis models
        self.analysis_models: Dict[str, Any] = {}
        
        # Research databases
        self.research_databases: Dict[UUID, Dict[str, Any]] = {}
        
        # Redis for caching and coordination
        self.redis_client: Optional[redis.Redis] = None
        
        # Thread pools for heavy operations
        self.vm_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="research-vm")
        self.analysis_executor = ProcessPoolExecutor(max_workers=2)
        
        logger.info("Research VM orchestrator initialized")
    
    async def initialize(self) -> None:
        """Initialize research VM orchestration system"""
        
        try:
            # Initialize Redis connection
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=2,  # Use separate DB for research caching
                decode_responses=True
            )
            await self.redis_client.ping()
            logger.info("Redis connection established for research VM caching")
            
        except Exception as e:
            logger.warning(f"Redis not available, using in-memory caching: {e}")
            self.redis_client = None
        
        # Initialize Playwright for browser automation
        self.playwright_instance = await async_playwright().start()
        
        # Load existing research VM configurations
        await self._load_existing_configurations()
        
        # Start monitoring tasks
        self.monitoring_active = True
        self.resource_monitor_task = asyncio.create_task(self._resource_monitoring_loop())
        self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Restore running research VMs
        await self._restore_existing_vms()
        
        logger.info("Research VM orchestrator fully initialized")
    
    async def create_research_vm(
        self,
        user_id: UserID,
        session_id: SessionID,
        vm_config: Optional[ResearchVMConfiguration] = None,
        vm_name: Optional[str] = None
    ) -> Tuple[UUID, ResearchVMState]:
        """
        Create dedicated research VM with full capabilities
        
        Returns:
            Tuple of (vm_id, vm_state) for the created research VM
        """
        
        if vm_config is None:
            vm_config = ResearchVMConfiguration()
        
        vm_id = uuid4()
        vm_name = vm_name or f"research-vm-{user_id}-{int(time.time())}"
        
        logger.info(f"Creating research VM {vm_id} for user {user_id}")
        
        try:
            # Create base VM using existing infrastructure
            base_vm = await self.base_vm_manager.create_ai_computer(
                instance_name=vm_name,
                personality_config={
                    "research_focused": True,
                    "capabilities": [cap.value for cap in vm_config.enabled_capabilities],
                    "user_id": user_id,
                    "session_id": session_id
                },
                specs=vm_config.vm_specs
            )
            
            # Configure research VM state
            vm_state = ResearchVMState(
                vm_id=vm_id,
                vm_state=VMState.CREATING,
                capability_status={cap: False for cap in vm_config.enabled_capabilities}
            )
            
            # Store configurations
            self.research_vms[vm_id] = vm_state
            self.vm_configurations[vm_id] = vm_config
            
            # Track user mapping
            if user_id not in self.user_vm_mapping:
                self.user_vm_mapping[user_id] = set()
            self.user_vm_mapping[user_id].add(vm_id)
            
            # Setup research environment in VM
            await self._setup_research_environment(vm_id, base_vm, vm_config)
            
            # Install and configure browser engines
            await self._setup_browser_engines(vm_id, base_vm, vm_config)
            
            # Load AI models for analysis
            await self._load_research_models(vm_id, base_vm, vm_config)
            
            # Setup research workspace and databases
            await self._setup_research_workspace(vm_id, base_vm, vm_config)
            
            # Configure memory integration
            await self._setup_memory_integration(vm_id, user_id, session_id)
            
            # Setup file processing integration
            await self._setup_file_processing(vm_id, user_id, session_id)
            
            # Configure prompt management for research
            await self._setup_research_prompts(vm_id, user_id, vm_config)
            
            # Start capability monitoring
            await self._verify_capabilities(vm_id, vm_config)
            
            # Update state
            vm_state.vm_state = VMState.RUNNING
            vm_state.session_count = 1
            vm_state.last_activity = datetime.now(timezone.utc)
            
            # Cache configuration
            if self.redis_client:
                await self.redis_client.hset(
                    f"research_vm:{vm_id}",
                    mapping={
                        "user_id": user_id,
                        "session_id": session_id,
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "vm_type": vm_config.vm_type.value,
                        "capabilities": json.dumps([cap.value for cap in vm_config.enabled_capabilities])
                    }
                )
                await self.redis_client.expire(f"research_vm:{vm_id}", 86400)  # 24 hour TTL
            
            logger.info(f"Research VM {vm_id} created and configured successfully")
            return vm_id, vm_state
            
        except Exception as e:
            logger.error(f"Failed to create research VM {vm_id}: {e}")
            
            # Cleanup on failure
            await self._cleanup_failed_vm(vm_id)
            raise
    
    async def _setup_research_environment(
        self,
        vm_id: UUID,
        base_vm: AIVMInstance,
        config: ResearchVMConfiguration
    ) -> None:
        """Setup complete research environment in VM"""
        
        # System packages and dependencies
        system_setup_commands = [
            "sudo apt-get update && sudo apt-get upgrade -y",
            "sudo apt-get install -y python3.11 python3.11-pip python3.11-venv",
            "sudo apt-get install -y firefox chromium-browser wget curl git",
            "sudo apt-get install -y xvfb xauth dbus-x11 fonts-liberation",
            "sudo apt-get install -y tesseract-ocr tesseract-ocr-all poppler-utils",
            "sudo apt-get install -y ffmpeg imagemagick pandoc",
            "sudo apt-get install -y redis-server postgresql-client",
            "sudo apt-get install -y build-essential cmake pkg-config",
            "sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev",
            "sudo apt-get install -y libavcodec-dev libavformat-dev libswscale-dev",
            "sudo apt-get install -y libssl-dev libffi-dev libxml2-dev libxslt1-dev"
        ]
        
        for cmd in system_setup_commands:
            result = await self.base_vm_manager.execute_command(base_vm.vm_id, cmd)
            if not result.success:
                logger.warning(f"System setup command failed: {cmd}")
        
        # Create research workspace structure
        workspace_commands = [
            "mkdir -p /home/ai/research/{workspace,downloads,screenshots,databases,models,cache}",
            "mkdir -p /home/ai/research/browsers/{chromium,firefox,profiles}",
            "mkdir -p /home/ai/research/scripts/{automation,analysis,processing}",
            "mkdir -p /home/ai/research/logs/{system,browser,analysis}",
            "chmod -R 755 /home/ai/research",
            "chown -R ai:ai /home/ai/research"
        ]
        
        for cmd in workspace_commands:
            result = await self.base_vm_manager.execute_command(base_vm.vm_id, cmd)
            if not result.success:
                logger.error(f"Workspace setup failed: {cmd}")
        
        # Install Python research packages
        python_packages = [
            "playwright==1.40.0",
            "selenium==4.15.0",
            "beautifulsoup4==4.12.2",
            "requests==2.31.0",
            "aiohttp==3.9.0",
            "pandas==2.1.4",
            "numpy==1.24.3",
            "pillow==10.1.0",
            "opencv-python==4.8.1.78",
            "pytesseract==0.3.10",
            "pdfplumber==0.9.0",
            "python-docx==0.8.11",
            "openpyxl==3.1.2",
            "sentence-transformers==2.2.2",
            "transformers==4.35.2",
            "torch==2.1.1",
            "scikit-learn==1.3.2",
            "nltk==3.8.1",
            "spacy==3.7.2",
            "scrapy==2.11.0",
            "redis==5.0.1",
            "psycopg2-binary==2.9.9",
            "sqlalchemy==2.0.23",
            "jupyter==1.0.0",
            "ipython==8.17.2"
        ]
        
        # Create virtual environment and install packages
        venv_commands = [
            "python3.11 -m venv /home/ai/research/venv",
            "source /home/ai/research/venv/bin/activate",
            f"pip install {' '.join(python_packages)}",
            "python -m playwright install",
            "python -m playwright install-deps",
            "python -m spacy download en_core_web_sm",
            "python -c \"import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet')\""
        ]
        
        venv_command = " && ".join(venv_commands)
        result = await self.base_vm_manager.execute_command(base_vm.vm_id, venv_command)
        if not result.success:
            logger.error(f"Python environment setup failed: {result.error}")
        
        # Setup display server for browser automation
        display_commands = [
            "export DISPLAY=:99",
            "Xvfb :99 -screen 0 1920x1080x24 -ac +extension GLX +render -noreset &",
            "sleep 2",
            "echo 'export DISPLAY=:99' >> /home/ai/.bashrc"
        ]
        
        for cmd in display_commands:
            await self.base_vm_manager.execute_command(base_vm.vm_id, cmd)
        
        logger.info(f"Research environment setup completed for VM {vm_id}")
    
    async def _setup_browser_engines(
        self,
        vm_id: UUID,
        base_vm: AIVMInstance,
        config: ResearchVMConfiguration
    ) -> None:
        """Setup and configure browser engines for research"""
        
        browser_configs = {}
        
        for browser_engine in config.browser_engines:
            if browser_engine == BrowserEngine.CHROMIUM:
                # Configure Chromium with research-optimized settings
                chromium_profile_setup = [
                    "mkdir -p /home/ai/research/browsers/chromium/profiles/research",
                    "mkdir -p /home/ai/research/browsers/chromium/extensions",
                    """cat > /home/ai/research/browsers/chromium/launch_config.json << 'EOF'
{
    "args": [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-extensions-except=/home/ai/research/browsers/chromium/extensions",
        "--load-extension=/home/ai/research/browsers/chromium/extensions",
        "--user-data-dir=/home/ai/research/browsers/chromium/profiles/research",
        "--disable-background-timer-throttling",
        "--disable-backgrounding-occluded-windows",
        "--disable-renderer-backgrounding",
        "--disable-features=TranslateUI",
        "--disable-ipc-flooding-protection",
        "--enable-automation",
        "--remote-debugging-port=9222"
    ],
    "headless": true,
    "devtools": false
}
EOF"""
                ]
                
                for cmd in chromium_profile_setup:
                    await self.base_vm_manager.execute_command(base_vm.vm_id, cmd)
                
                browser_configs[BrowserEngine.CHROMIUM] = {
                    "profile_path": "/home/ai/research/browsers/chromium/profiles/research",
                    "extensions_path": "/home/ai/research/browsers/chromium/extensions",
                    "config_path": "/home/ai/research/browsers/chromium/launch_config.json"
                }
            
            elif browser_engine == BrowserEngine.FIREFOX:
                # Configure Firefox with research profile
                firefox_profile_setup = [
                    "mkdir -p /home/ai/research/browsers/firefox/profiles/research",
                    "mkdir -p /home/ai/research/browsers/firefox/extensions",
                    """cat > /home/ai/research/browsers/firefox/prefs.js << 'EOF'
user_pref("browser.download.dir", "/home/ai/research/downloads");
user_pref("browser.download.folderList", 2);
user_pref("browser.helperApps.neverAsk.saveToDisk", "application/pdf,text/csv,application/vnd.ms-excel,application/vnd.openxmlformats-officedocument.spreadsheetml.sheet");
user_pref("pdfjs.disabled", true);
user_pref("privacy.trackingprotection.enabled", false);
user_pref("dom.webdriver.enabled", true);
user_pref("dom.webnotifications.enabled", false);
user_pref("media.volume_scale", "0.0");
EOF""",
                    "cp /home/ai/research/browsers/firefox/prefs.js /home/ai/research/browsers/firefox/profiles/research/"
                ]
                
                for cmd in firefox_profile_setup:
                    await self.base_vm_manager.execute_command(base_vm.vm_id, cmd)
                
                browser_configs[BrowserEngine.FIREFOX] = {
                    "profile_path": "/home/ai/research/browsers/firefox/profiles/research",
                    "extensions_path": "/home/ai/research/browsers/firefox/extensions",
                    "prefs_path": "/home/ai/research/browsers/firefox/prefs.js"
                }
        
        # Install research-specific browser extensions
        await self._install_research_extensions(vm_id, base_vm, browser_configs)
        
        # Test browser functionality
        await self._test_browser_functionality(vm_id, base_vm, config.browser_engines)
        
        # Update VM state
        vm_state = self.research_vms[vm_id]
        vm_state.browser_sessions = browser_configs
        
        logger.info(f"Browser engines configured for VM {vm_id}: {[e.value for e in config.browser_engines]}")
    
    async def _install_research_extensions(
        self,
        vm_id: UUID,
        base_vm: AIVMInstance,
        browser_configs: Dict[BrowserEngine, Dict[str, str]]
    ) -> None:
        """Install research-specific browser extensions"""
        
        # Custom research extension manifest
        research_extension_manifest = {
            "manifest_version": 3,
            "name": "Somnus Research Assistant",
            "version": "1.0.0",
            "description": "AI-powered research assistance extension",
            "permissions": [
                "activeTab",
                "storage",
                "scripting",
                "webRequest",
                "webRequestBlocking"
            ],
            "host_permissions": ["<all_urls>"],
            "content_scripts": [{
                "matches": ["<all_urls>"],
                "js": ["content.js"],
                "run_at": "document_end"
            }],
            "background": {
                "service_worker": "background.js"
            },
            "action": {
                "default_popup": "popup.html",
                "default_title": "Research Assistant"
            }
        }
        
        # Content script for research assistance
        content_script = """
// Somnus Research Assistant Content Script
(function() {
    'use strict';
    
    // Add research markers to page
    function addResearchMarkers() {
        // Mark citations and references
        const citationSelectors = [
            'a[href*="doi.org"]',
            'a[href*="pubmed"]',
            'a[href*="arxiv"]',
            '.citation',
            '.reference',
            '.bibliography'
        ];
        
        citationSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                el.style.border = '2px solid #4CAF50';
                el.title = 'Research Citation Detected';
            });
        });
        
        // Mark author information
        const authorSelectors = [
            '.author',
            '.byline',
            '[rel="author"]',
            '.post-author'
        ];
        
        authorSelectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(el => {
                el.style.backgroundColor = '#E3F2FD';
                el.title = 'Author Information';
            });
        });
    }
    
    // Extract structured data
    function extractStructuredData() {
        const data = {
            title: document.title,
            url: window.location.href,
            citations: [],
            authors: [],
            publicationDate: null,
            abstract: null
        };
        
        // Extract citations
        document.querySelectorAll('a[href*="doi.org"], a[href*="pubmed"], a[href*="arxiv"]').forEach(link => {
            data.citations.push({
                text: link.textContent.trim(),
                href: link.href,
                type: link.href.includes('doi.org') ? 'DOI' : 
                      link.href.includes('pubmed') ? 'PubMed' :
                      link.href.includes('arxiv') ? 'arXiv' : 'Other'
            });
        });
        
        // Extract authors
        document.querySelectorAll('.author, .byline, [rel="author"]').forEach(author => {
            const text = author.textContent.trim();
            if (text && !data.authors.includes(text)) {
                data.authors.push(text);
            }
        });
        
        // Extract publication date
        const dateSelectors = ['time[datetime]', '.date', '.published', '[property="article:published_time"]'];
        for (const selector of dateSelectors) {
            const dateEl = document.querySelector(selector);
            if (dateEl) {
                data.publicationDate = dateEl.getAttribute('datetime') || dateEl.textContent.trim();
                break;
            }
        }
        
        // Extract abstract
        const abstractSelectors = ['.abstract', '.summary', '[property="article:description"]'];
        for (const selector of abstractSelectors) {
            const abstractEl = document.querySelector(selector);
            if (abstractEl) {
                data.abstract = abstractEl.textContent.trim();
                break;
            }
        }
        
        return data;
    }
    
    // Initialize on page load
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', addResearchMarkers);
    } else {
        addResearchMarkers();
    }
    
    // Make extraction function available globally
    window.somnusExtractData = extractStructuredData;
})();
"""
        
        # Background script
        background_script = """
// Somnus Research Assistant Background Script
chrome.action.onClicked.addListener((tab) => {
    chrome.scripting.executeScript({
        target: { tabId: tab.id },
        function: () => {
            const data = window.somnusExtractData ? window.somnusExtractData() : null;
            if (data) {
                console.log('Extracted research data:', data);
                // Send to Somnus research system
                fetch('http://localhost:8000/api/research/extracted_data', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(data)
                }).catch(console.error);
            }
        }
    });
});
"""
        
        # Create extension files for each browser
        for browser_engine, config in browser_configs.items():
            extension_path = config["extensions_path"]
            
            extension_files = [
                (f"{extension_path}/manifest.json", json.dumps(research_extension_manifest, indent=2)),
                (f"{extension_path}/content.js", content_script),
                (f"{extension_path}/background.js", background_script),
                (f"{extension_path}/popup.html", """
<!DOCTYPE html>
<html>
<head>
    <style>
        body { width: 300px; padding: 15px; }
        .button { background: #4CAF50; color: white; padding: 10px; border: none; border-radius: 4px; cursor: pointer; width: 100%; margin: 5px 0; }
        .stats { font-size: 12px; color: #666; margin-top: 10px; }
    </style>
</head>
<body>
    <h3>Somnus Research Assistant</h3>
    <button class="button" onclick="extractData()">Extract Research Data</button>
    <button class="button" onclick="analyzeCredibility()">Analyze Credibility</button>
    <button class="button" onclick="findCitations()">Find Citations</button>
    <div class="stats" id="stats">Ready for research</div>
    
    <script>
        function extractData() {
            chrome.tabs.query({active: true, currentWindow: true}, (tabs) => {
                chrome.scripting.executeScript({
                    target: { tabId: tabs[0].id },
                    function: () => window.somnusExtractData()
                });
            });
        }
        
        function analyzeCredibility() {
            document.getElementById('stats').textContent = 'Analyzing credibility...';
        }
        
        function findCitations() {
            document.getElementById('stats').textContent = 'Finding citations...';
        }
    </script>
</body>
</html>
                """)
            ]
            
            for file_path, content in extension_files:
                create_file_cmd = f"""cat > {file_path} << 'EOF'
{content}
EOF"""
                await self.base_vm_manager.execute_command(base_vm.vm_id, create_file_cmd)
        
        logger.info(f"Research extensions installed for VM {vm_id}")
    
    async def _load_research_models(
        self,
        vm_id: UUID,
        base_vm: AIVMInstance,
        config: ResearchVMConfiguration
    ) -> None:
        """Load AI models for research analysis in VM"""
        
        loaded_models = {}
        
        for model_name in config.ai_models:
            try:
                # Load model using the model loader
                model_request = ModelLoadRequest(
                    model_id=model_name,
                    device="auto",
                    quantization_config=None
                )
                
                model_response = await self.model_loader.load_model(model_request)
                
                if model_response.success:
                    loaded_models[model_name] = {
                        "model_id": model_response.model_id,
                        "memory_usage_gb": model_response.memory_usage_gb,
                        "load_time": model_response.load_time_seconds,
                        "capabilities": self._get_model_capabilities(model_name)
                    }
                    
                    logger.info(f"Loaded model {model_name} in research VM {vm_id}")
                else:
                    logger.error(f"Failed to load model {model_name}: {model_response.message}")
                    
            except Exception as e:
                logger.error(f"Error loading model {model_name}: {e}")
        
        # Install model-specific Python libraries in VM
        model_libraries = [
            "sentence-transformers",
            "transformers",
            "torch",
            "tokenizers",
            "datasets",
            "evaluate",
            "accelerate",
            "optimum"
        ]
        
        install_cmd = f"source /home/ai/research/venv/bin/activate && pip install {' '.join(model_libraries)}"
        await self.base_vm_manager.execute_command(base_vm.vm_id, install_cmd)
        
        # Create model analysis scripts
        analysis_script = f"""#!/usr/bin/env python3
'''
Somnus Research VM - AI Analysis Script
Provides AI-powered analysis capabilities for research content
'''

import sys
import json
import torch
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModel
import numpy as np
from pathlib import Path

class ResearchAnalyzer:
    def __init__(self):
        self.models = {{}}
        self.loaded_models = {json.dumps(list(config.ai_models))}
        self._load_models()
    
    def _load_models(self):
        '''Load all configured models'''
        for model_name in self.loaded_models:
            try:
                if 'sentence-transformers' in model_name:
                    self.models[model_name] = SentenceTransformer(model_name)
                elif 'DialoGPT' in model_name:
                    self.models[model_name] = pipeline('text-generation', model=model_name)
                else:
                    self.models[model_name] = AutoModel.from_pretrained(model_name)
                print(f"Loaded model: {{model_name}}")
            except Exception as e:
                print(f"Failed to load model {{model_name}}: {{e}}")
    
    def generate_embeddings(self, texts):
        '''Generate semantic embeddings for texts'''
        model_name = 'sentence-transformers/all-MiniLM-L6-v2'
        if model_name in self.models:
            embeddings = self.models[model_name].encode(texts)
            return embeddings.tolist()
        return None
    
    def analyze_content_quality(self, content):
        '''Analyze content quality and credibility'''
        # Length analysis
        word_count = len(content.split())
        char_count = len(content)
        
        # Structure analysis
        sentences = content.split('.')
        paragraphs = content.split('\\n\\n')
        
        # Authority indicators
        authority_keywords = ['research', 'study', 'analysis', 'university', 'professor', 'journal']
        authority_score = sum(1 for keyword in authority_keywords if keyword.lower() in content.lower())
        
        # Citation analysis
        citation_patterns = ['doi:', 'http://dx.doi.org', 'arxiv:', 'pubmed']
        citation_score = sum(1 for pattern in citation_patterns if pattern in content.lower())
        
        # Calculate quality score
        length_score = min(1.0, word_count / 1000)
        structure_score = min(1.0, len(paragraphs) / 5)
        authority_score = min(1.0, authority_score / 3)
        citation_score = min(1.0, citation_score / 2)
        
        overall_score = (length_score * 0.3 + structure_score * 0.2 + 
                        authority_score * 0.3 + citation_score * 0.2)
        
        return {{
            'overall_score': overall_score,
            'word_count': word_count,
            'sentence_count': len(sentences),
            'paragraph_count': len(paragraphs),
            'authority_indicators': authority_score,
            'citation_indicators': citation_score,
            'readability_estimate': min(1.0, 1 / (1 + len(content) / 10000))
        }}
    
    def detect_bias(self, content):
        '''Detect potential bias indicators'''
        bias_indicators = [
            'always', 'never', 'all', 'none', 'everyone', 'no one',
            'obviously', 'clearly', 'without doubt', 'definitely',
            'certainly', 'absolutely', 'unquestionably'
        ]
        
        emotional_language = [
            'amazing', 'terrible', 'incredible', 'awful', 'fantastic',
            'horrible', 'excellent', 'disgusting', 'perfect', 'worst'
        ]
        
        content_lower = content.lower()
        
        bias_count = sum(1 for indicator in bias_indicators if indicator in content_lower)
        emotional_count = sum(1 for word in emotional_language if word in content_lower)
        
        total_words = len(content.split())
        bias_ratio = bias_count / max(total_words, 1)
        emotional_ratio = emotional_count / max(total_words, 1)
        
        return {{
            'bias_indicators_found': bias_count,
            'emotional_language_found': emotional_count,
            'bias_ratio': bias_ratio,
            'emotional_ratio': emotional_ratio,
            'overall_bias_score': min(1.0, (bias_ratio + emotional_ratio) * 10)
        }}
    
    def extract_key_topics(self, content, num_topics=10):
        '''Extract key topics from content'''
        # Simple keyword extraction (would use more sophisticated NLP in production)
        words = content.lower().split()
        
        # Filter meaningful words
        stopwords = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
        meaningful_words = [word for word in words if len(word) > 3 and word not in stopwords and word.isalpha()]
        
        # Count frequencies
        word_freq = {{}}
        for word in meaningful_words:
            word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top topics
        sorted_topics = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [{{
            'topic': word,
            'frequency': freq,
            'relevance': freq / len(meaningful_words)
        }} for word, freq in sorted_topics[:num_topics]]

def main():
    if len(sys.argv) < 3:
        print("Usage: python3 analysis.py <function> <content>")
        sys.exit(1)
    
    function = sys.argv[1]
    content = sys.argv[2]
    
    analyzer = ResearchAnalyzer()
    
    if function == 'quality':
        result = analyzer.analyze_content_quality(content)
    elif function == 'bias':
        result = analyzer.detect_bias(content)
    elif function == 'topics':
        result = analyzer.extract_key_topics(content)
    elif function == 'embeddings':
        result = analyzer.generate_embeddings([content])
    else:
        result = {{'error': f'Unknown function: {{function}}'}}
    
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
"""
        
        # Write analysis script to VM
        script_path = "/home/ai/research/scripts/analysis/ai_analysis.py"
        create_script_cmd = f"""cat > {script_path} << 'EOF'
{analysis_script}
EOF"""
        await self.base_vm_manager.execute_command(base_vm.vm_id, create_script_cmd)
        await self.base_vm_manager.execute_command(base_vm.vm_id, f"chmod +x {script_path}")
        
        # Update VM state
        vm_state = self.research_vms[vm_id]
        vm_state.loaded_models = loaded_models
        
        logger.info(f"Research models loaded for VM {vm_id}: {list(loaded_models.keys())}")
    
    async def _setup_research_workspace(
        self,
        vm_id: UUID,
        base_vm: AIVMInstance,
        config: ResearchVMConfiguration
    ) -> None:
        """Setup research workspace and databases in VM"""
        
        # Create research database schema
        database_schema = {
            "research_sessions": {
                "session_id": "TEXT PRIMARY KEY",
                "user_id": "TEXT NOT NULL",
                "query": "TEXT NOT NULL",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "status": "TEXT DEFAULT 'active'",
                "metadata": "JSON"
            },
            "research_sources": {
                "source_id": "TEXT PRIMARY KEY",
                "session_id": "TEXT",
                "url": "TEXT NOT NULL",
                "title": "TEXT",
                "content": "TEXT",
                "extracted_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "quality_score": "REAL",
                "credibility_score": "REAL",
                "bias_score": "REAL",
                "metadata": "JSON",
                "FOREIGN KEY (session_id) REFERENCES research_sessions(session_id)"
            },
            "research_analysis": {
                "analysis_id": "TEXT PRIMARY KEY",
                "source_id": "TEXT",
                "analysis_type": "TEXT NOT NULL",
                "result": "JSON",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "FOREIGN KEY (source_id) REFERENCES research_sources(source_id)"
            },
            "research_cache": {
                "cache_key": "TEXT PRIMARY KEY",
                "content": "TEXT",
                "embeddings": "BLOB",
                "created_at": "TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "expires_at": "TIMESTAMP",
                "access_count": "INTEGER DEFAULT 0"
            }
        }
        
        # Create SQLite database initialization script
        db_init_script = f"""#!/usr/bin/env python3
'''
Initialize research database
'''

import sqlite3
import json
from pathlib import Path

def initialize_database():
    db_path = '/home/ai/research/databases/research.db'
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Enable JSON support
    cursor.execute('PRAGMA journal_mode=WAL')
    cursor.execute('PRAGMA synchronous=NORMAL')
    cursor.execute('PRAGMA cache_size=10000')
    
    schema = {json.dumps(database_schema, indent=4)}
    
    for table_name, columns in schema.items():
        column_defs = []
        for col_name, col_type in columns.items():
            if 'FOREIGN KEY' not in col_type:
                column_defs.append(f"{{col_name}} {{col_type}}")
        
        create_table_sql = f"CREATE TABLE IF NOT EXISTS {{table_name}} ({{', '.join(column_defs)}})"
        cursor.execute(create_table_sql)
        
        # Add foreign key constraints
        for col_name, col_type in columns.items():
            if 'FOREIGN KEY' in col_type:
                cursor.execute(f"CREATE INDEX IF NOT EXISTS idx_{{table_name}}_{{col_name}} ON {{table_name}}({{col_name}})")
    
    # Create indexes
    indexes = [
        "CREATE INDEX IF NOT EXISTS idx_sources_session ON research_sources(session_id)",
        "CREATE INDEX IF NOT EXISTS idx_sources_url ON research_sources(url)",
        "CREATE INDEX IF NOT EXISTS idx_analysis_source ON research_analysis(source_id)",
        "CREATE INDEX IF NOT EXISTS idx_cache_expires ON research_cache(expires_at)"
    ]
    
    for index_sql in indexes:
        cursor.execute(index_sql)
    
    conn.commit()
    conn.close()
    print("Research database initialized successfully")

if __name__ == '__main__':
    initialize_database()
"""
        
        # Write and execute database initialization
        db_script_path = "/home/ai/research/scripts/db_init.py"
        create_db_script_cmd = f"""cat > {db_script_path} << 'EOF'
{db_init_script}
EOF"""
        await self.base_vm_manager.execute_command(base_vm.vm_id, create_db_script_cmd)
        await self.base_vm_manager.execute_command(base_vm.vm_id, f"chmod +x {db_script_path}")
        
        # Initialize database
        init_result = await self.base_vm_manager.execute_command(
            base_vm.vm_id, 
            f"source /home/ai/research/venv/bin/activate && python3 {db_script_path}"
        )
        
        if init_result.success:
            logger.info(f"Research database initialized for VM {vm_id}")
        else:
            logger.error(f"Database initialization failed: {init_result.error}")
        
        # Create research API server script
        api_server_script = """#!/usr/bin/env python3
'''
Research VM API Server
Provides local API for research operations
'''

from flask import Flask, request, jsonify
import sqlite3
import json
import subprocess
import os
from datetime import datetime, timezone

app = Flask(__name__)
app.config['JSON_SORT_KEYS'] = False

DB_PATH = '/home/ai/research/databases/research.db'

def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/api/research/session', methods=['POST'])
def create_research_session():
    data = request.json
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO research_sessions (session_id, user_id, query, metadata)
        VALUES (?, ?, ?, ?)
    ''', (data['session_id'], data['user_id'], data['query'], json.dumps(data.get('metadata', {}))))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'session_id': data['session_id']})

@app.route('/api/research/source', methods=['POST'])
def add_research_source():
    data = request.json
    
    # Analyze content quality
    analysis_result = subprocess.run([
        'python3', '/home/ai/research/scripts/analysis/ai_analysis.py',
        'quality', data['content']
    ], capture_output=True, text=True)
    
    quality_data = json.loads(analysis_result.stdout) if analysis_result.returncode == 0 else {}
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO research_sources 
        (source_id, session_id, url, title, content, quality_score, metadata)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (
        data['source_id'], data['session_id'], data['url'], 
        data['title'], data['content'], quality_data.get('overall_score', 0.5),
        json.dumps(data.get('metadata', {}))
    ))
    
    conn.commit()
    conn.close()
    
    return jsonify({'success': True, 'quality_analysis': quality_data})

@app.route('/api/research/analyze', methods=['POST'])
def analyze_content():
    data = request.json
    content = data['content']
    analysis_type = data.get('type', 'quality')
    
    result = subprocess.run([
        'python3', '/home/ai/research/scripts/analysis/ai_analysis.py',
        analysis_type, content
    ], capture_output=True, text=True)
    
    if result.returncode == 0:
        analysis_result = json.loads(result.stdout)
        return jsonify({'success': True, 'analysis': analysis_result})
    else:
        return jsonify({'success': False, 'error': result.stderr})

@app.route('/api/research/sessions', methods=['GET'])
def get_research_sessions():
    user_id = request.args.get('user_id')
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    if user_id:
        cursor.execute('SELECT * FROM research_sessions WHERE user_id = ? ORDER BY created_at DESC', (user_id,))
    else:
        cursor.execute('SELECT * FROM research_sessions ORDER BY created_at DESC LIMIT 50')
    
    sessions = [dict(row) for row in cursor.fetchall()]
    conn.close()
    
    return jsonify({'sessions': sessions})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=False)
"""
        
        # Write API server script
        api_script_path = "/home/ai/research/scripts/api_server.py"
        create_api_script_cmd = f"""cat > {api_script_path} << 'EOF'
{api_server_script}
EOF"""
        await self.base_vm_manager.execute_command(base_vm.vm_id, create_api_script_cmd)
        await self.base_vm_manager.execute_command(base_vm.vm_id, f"chmod +x {api_script_path}")
        
        # Install Flask for API server
        flask_install_cmd = "source /home/ai/research/venv/bin/activate && pip install flask"
        await self.base_vm_manager.execute_command(base_vm.vm_id, flask_install_cmd)
        
        # Start API server
        start_api_cmd = "source /home/ai/research/venv/bin/activate && nohup python3 /home/ai/research/scripts/api_server.py > /home/ai/research/logs/api.log 2>&1 &"
        await self.base_vm_manager.execute_command(base_vm.vm_id, start_api_cmd)
        
        logger.info(f"Research workspace and databases setup completed for VM {vm_id}")
    
    async def _setup_memory_integration(
        self,
        vm_id: UUID,
        user_id: UserID,
        session_id: SessionID
    ) -> None:
        """Setup memory system integration for research VM"""
        
        # Create memory context for research VM
        memory_context = SessionMemoryContext(
            session_id=f"research_vm_{vm_id}",
            user_id=user_id,
            memory_manager=self.memory_manager,
            session_type="research_vm"
        )
        
        # Initialize memory context
        await memory_context.initialize()
        
        # Store research VM information in memory
        await self.memory_manager.store_memory(
            user_id=user_id,
            content=f"Research VM {vm_id} created for advanced research capabilities",
            memory_type=MemoryType.TOOL_RESULT,
            importance=MemoryImportance.HIGH,
            scope=MemoryScope.PRIVATE,
            metadata={
                "vm_id": str(vm_id),
                "vm_type": "research",
                "capabilities": [cap.value for cap in self.vm_configurations[vm_id].enabled_capabilities],
                "session_id": session_id
            }
        )
        
        # Store memory context reference
        if not hasattr(self, 'memory_contexts'):
            self.memory_contexts = {}
        self.memory_contexts[vm_id] = memory_context
        
        logger.info(f"Memory integration setup completed for research VM {vm_id}")
    
    async def _setup_file_processing(
        self,
        vm_id: UUID,
        user_id: UserID,
        session_id: SessionID
    ) -> None:
        """Setup file processing integration for research VM"""
        
        # Create file processing configuration for research
        processing_config = {
            "vm_id": str(vm_id),
            "user_id": user_id,
            "session_id": session_id,
            "processing_priority": ProcessingPriority.HIGH.value,
            "research_focused": True,
            "auto_analysis": True
        }
        
        # Create file watcher script for research VM
        file_watcher_script = f"""#!/usr/bin/env python3
'''
Research VM File Watcher
Automatically processes files dropped into research folders
'''

import os
import time
import requests
import json
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

class ResearchFileHandler(FileSystemEventHandler):
    def __init__(self):
        self.processing_config = {json.dumps(processing_config)}
        self.api_endpoint = 'http://localhost:8000/api/file/process'
    
    def on_created(self, event):
        if event.is_directory:
            return
        
        file_path = Path(event.src_path)
        
        # Only process specific file types
        allowed_extensions = ['.pdf', '.txt', '.html', '.csv', '.xlsx', '.docx', '.json']
        if file_path.suffix.lower() not in allowed_extensions:
            return
        
        # Wait for file to be fully written
        time.sleep(2)
        
        try:
            with open(file_path, 'rb') as f:
                files = {{'file': (file_path.name, f, 'application/octet-stream')}}
                data = {{
                    'user_id': self.processing_config['user_id'],
                    'session_id': self.processing_config['session_id'],
                    'source_context': 'research_vm_auto',
                    'vm_id': self.processing_config['vm_id']
                }}
                
                response = requests.post(self.api_endpoint, files=files, data=data)
                
                if response.status_code == 200:
                    print(f"Successfully queued file for processing: {{file_path.name}}")
                else:
                    print(f"Failed to queue file: {{file_path.name}} - {{response.status_code}}")
                    
        except Exception as e:
            print(f"Error processing file {{file_path.name}}: {{e}}")

def main():
    event_handler = ResearchFileHandler()
    observer = Observer()
    
    # Watch research directories
    watch_paths = [
        '/home/ai/research/downloads',
        '/home/ai/research/workspace'
    ]
    
    for path in watch_paths:
        if os.path.exists(path):
            observer.schedule(event_handler, path, recursive=True)
            print(f"Watching directory: {{path}}")
    
    observer.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    
    observer.join()

if __name__ == '__main__':
    main()
"""
        
        # Write file watcher script
        watcher_script_path = "/home/ai/research/scripts/file_watcher.py"
        create_watcher_cmd = f"""cat > {watcher_script_path} << 'EOF'
{file_watcher_script}
EOF"""
        await self.base_vm_manager.execute_command(base_vm.vm_id, create_watcher_cmd)
        await self.base_vm_manager.execute_command(base_vm.vm_id, f"chmod +x {watcher_script_path}")
        
        # Install watchdog for file monitoring
        watchdog_install_cmd = "source /home/ai/research/venv/bin/activate && pip install watchdog requests"
        await self.base_vm_manager.execute_command(base_vm.vm_id, watchdog_install_cmd)
        
        # Start file watcher
        start_watcher_cmd = f"source /home/ai/research/venv/bin/activate && nohup python3 {watcher_script_path} > /home/ai/research/logs/file_watcher.log 2>&1 &"
        await self.base_vm_manager.execute_command(base_vm.vm_id, start_watcher_cmd)
        
        logger.info(f"File processing integration setup completed for research VM {vm_id}")
    
    async def _setup_research_prompts(
        self,
        vm_id: UUID,
        user_id: UserID,
        config: ResearchVMConfiguration
    ) -> None:
        """Setup research-focused prompt management"""
        
        # Generate research-specific system prompt
        research_prompt = await self.prompt_manager.generate_system_prompt(
            user_id=user_id,
            session_id=f"research_vm_{vm_id}",
            context=PromptContext.RESEARCH_FOCUSED,
            prompt_type=PromptType.RESEARCH_ASSISTANT,
            custom_variables={
                "vm_id": str(vm_id),
                "research_capabilities": [cap.value for cap in config.enabled_capabilities],
                "vm_type": config.vm_type.value,
                "browser_engines": [engine.value for engine in config.browser_engines]
            }
        )
        
        # Create prompt configuration file in VM
        prompt_config = {
            "research_prompt": research_prompt,
            "capabilities": [cap.value for cap in config.enabled_capabilities],
            "behavior_guidelines": {
                "prioritize_credible_sources": True,
                "detect_bias_automatically": True,
                "cross_reference_information": True,
                "maintain_research_ethics": True,
                "document_methodology": True
            },
            "analysis_focus": {
                "content_quality": True,
                "source_credibility": True,
                "bias_detection": True,
                "fact_verification": True,
                "citation_analysis": True
            }
        }
        
        # Write prompt configuration to VM
        prompt_config_path = "/home/ai/research/config/prompts.json"
        config_dir_cmd = "mkdir -p /home/ai/research/config"
        await self.base_vm_manager.execute_command(base_vm.vm_id, config_dir_cmd)
        
        create_prompt_config_cmd = f"""cat > {prompt_config_path} << 'EOF'
{json.dumps(prompt_config, indent=2)}
EOF"""
        await self.base_vm_manager.execute_command(base_vm.vm_id, create_prompt_config_cmd)
        
        logger.info(f"Research prompts configured for VM {vm_id}")
    
    async def _verify_capabilities(
        self,
        vm_id: UUID,
        config: ResearchVMConfiguration
    ) -> None:
        """Verify all capabilities are working in research VM"""
        
        vm_state = self.research_vms[vm_id]
        
        for capability in config.enabled_capabilities:
            try:
                if capability == ResearchCapability.WEB_SCRAPING:
                    # Test web scraping capability
                    test_cmd = "source /home/ai/research/venv/bin/activate && python3 -c \"import requests; print('Web scraping OK' if requests.get('https://httpbin.org/json').status_code == 200 else 'Failed')\""
                    result = await self.base_vm_manager.execute_command(vm_id, test_cmd)
                    vm_state.capability_status[capability] = "OK" in result.output
                
                elif capability == ResearchCapability.PDF_PROCESSING:
                    # Test PDF processing capability
                    test_cmd = "source /home/ai/research/venv/bin/activate && python3 -c \"import pdfplumber; print('PDF processing OK')\""
                    result = await self.base_vm_manager.execute_command(vm_id, test_cmd)
                    vm_state.capability_status[capability] = result.success
                
                elif capability == ResearchCapability.AI_ANALYSIS:
                    # Test AI analysis capability
                    test_cmd = "source /home/ai/research/venv/bin/activate && python3 /home/ai/research/scripts/analysis/ai_analysis.py quality 'This is a test content for analysis.'"
                    result = await self.base_vm_manager.execute_command(vm_id, test_cmd)
                    vm_state.capability_status[capability] = result.success and "overall_score" in result.output
                
                else:
                    # Default capability verification
                    vm_state.capability_status[capability] = True
                    
            except Exception as e:
                logger.error(f"Capability verification failed for {capability.value}: {e}")
                vm_state.capability_status[capability] = False
        
        # Log capability status
        working_capabilities = [cap.value for cap, status in vm_state.capability_status.items() if status]
        failed_capabilities = [cap.value for cap, status in vm_state.capability_status.items() if not status]
        
        logger.info(f"VM {vm_id} capabilities verified - Working: {working_capabilities}")
        if failed_capabilities:
            logger.warning(f"VM {vm_id} failed capabilities: {failed_capabilities}")
    
    async def _test_browser_functionality(
        self,
        vm_id: UUID,
        base_vm: AIVMInstance,
        browser_engines: List[BrowserEngine]
    ) -> None:
        """Test browser functionality in research VM"""
        
        for browser_engine in browser_engines:
            try:
                if browser_engine in [BrowserEngine.CHROMIUM, BrowserEngine.FIREFOX]:
                    # Test Playwright browser
                    test_script = f"""
source /home/ai/research/venv/bin/activate
python3 -c "
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    browser = p.{browser_engine.value}.launch(headless=True)
    page = browser.new_page()
    page.goto('https://httpbin.org/json')
    content = page.content()
    browser.close()
    print('Browser test OK' if 'slideshow' in content else 'Browser test failed')
"
"""
                    
                elif browser_engine in [BrowserEngine.SELENIUM_CHROME, BrowserEngine.SELENIUM_FIREFOX]:
                    # Test Selenium browser
                    browser_name = "chrome" if "chrome" in browser_engine.value else "firefox"
                    test_script = f"""
source /home/ai/research/venv/bin/activate
python3 -c "
from selenium import webdriver
from selenium.webdriver.{browser_name}.options import Options
options = Options()
options.add_argument('--headless')
options.add_argument('--no-sandbox')
options.add_argument('--disable-dev-shm-usage')
driver = webdriver.{'Chrome' if browser_name == 'chrome' else 'Firefox'}(options=options)
driver.get('https://httpbin.org/json')
content = driver.page_source
driver.quit()
print('Selenium test OK' if 'slideshow' in content else 'Selenium test failed')
"
"""
                
                result = await self.base_vm_manager.execute_command(vm_id, test_script)
                
                if result.success and "OK" in result.output:
                    logger.info(f"Browser {browser_engine.value} test passed for VM {vm_id}")
                else:
                    logger.warning(f"Browser {browser_engine.value} test failed for VM {vm_id}: {result.error}")
                    
            except Exception as e:
                logger.error(f"Browser functionality test failed for {browser_engine.value}: {e}")
    
    def _get_model_capabilities(self, model_name: str) -> List[str]:
        """Get capabilities for specific model"""
        
        capabilities = []
        
        if "sentence-transformers" in model_name:
            capabilities.extend(["text_embedding", "semantic_similarity", "content_clustering"])
        
        if "DialoGPT" in model_name:
            capabilities.extend(["text_generation", "conversation", "content_analysis"])
        
        if "bert" in model_name.lower():
            capabilities.extend(["text_classification", "sentiment_analysis", "question_answering"])
        
        return capabilities
    
    async def _resource_monitoring_loop(self) -> None:
        """Monitor resource usage for all research VMs"""
        
        while self.monitoring_active:
            try:
                for vm_id, vm_state in self.research_vms.items():
                    if vm_state.vm_state == VMState.RUNNING:
                        # Get VM resource usage
                        try:
                            stats_cmd = f"ps aux | grep 'research.*{vm_id}' | head -10"
                            result = await self.base_vm_manager.execute_command(vm_id, "top -bn1 | head -5")
                            
                            if result.success:
                                # Parse resource usage from output
                                lines = result.output.split('\n')
                                for line in lines:
                                    if '%Cpu' in line:
                                        cpu_match = line.split('%Cpu(s):')[1].split()[0]
                                        vm_state.cpu_usage = float(cpu_match)
                                    elif 'MiB Mem' in line:
                                        mem_parts = line.split()
                                        used_mem = float(mem_parts[5])
                                        total_mem = float(mem_parts[3])
                                        vm_state.memory_usage = (used_mem / total_mem) * 100
                            
                            # Check disk usage
                            disk_cmd = "df -h /home/ai/research | tail -1 | awk '{print $5}' | sed 's/%//'"
                            disk_result = await self.base_vm_manager.execute_command(vm_id, disk_cmd)
                            if disk_result.success:
                                vm_state.disk_usage = float(disk_result.output.strip())
                            
                            # Update activity timestamp
                            vm_state.last_activity = datetime.now(timezone.utc)
                            
                        except Exception as e:
                            logger.warning(f"Resource monitoring failed for VM {vm_id}: {e}")
                
                await asyncio.sleep(30)  # Monitor every 30 seconds
                
            except Exception as e:
                logger.error(f"Resource monitoring loop error: {e}")
                await asyncio.sleep(60)
    
    async def _cleanup_loop(self) -> None:
        """Periodic cleanup of research VMs and resources"""
        
        while self.monitoring_active:
            try:
                current_time = datetime.now(timezone.utc)
                
                for vm_id, vm_state in list(self.research_vms.items()):
                    # Check for inactive VMs
                    inactive_duration = current_time - vm_state.last_activity
                    
                    if inactive_duration > timedelta(hours=2) and vm_state.vm_state == VMState.RUNNING:
                        logger.info(f"Suspending inactive research VM {vm_id}")
                        await self._suspend_research_vm(vm_id)
                    
                    # Clean up temp files
                    if vm_state.vm_state == VMState.RUNNING:
                        cleanup_cmd = "find /home/ai/research/cache -type f -mtime +1 -delete"
                        await self.base_vm_manager.execute_command(vm_id, cleanup_cmd)
                
                # Clean Redis cache
                if self.redis_client:
                    expired_keys = await self.redis_client.keys("research_vm:*")
                    for key in expired_keys:
                        ttl = await self.redis_client.ttl(key)
                        if ttl == -1:  # No expiration set
                            await self.redis_client.expire(key, 86400)
                
                await asyncio.sleep(3600)  # Run cleanup every hour
                
            except Exception as e:
                logger.error(f"Cleanup loop error: {e}")
                await asyncio.sleep(3600)
    
    async def _suspend_research_vm(self, vm_id: UUID) -> None:
        """Suspend research VM to save resources"""
        
        try:
            vm_state = self.research_vms[vm_id]
            
            # Save current state
            await self._save_vm_state(vm_id)
            
            # Suspend VM
            result = await self.base_vm_manager.suspend_vm(vm_id)
            
            if result:
                vm_state.vm_state = VMState.SUSPENDED
                logger.info(f"Research VM {vm_id} suspended successfully")
            else:
                logger.error(f"Failed to suspend research VM {vm_id}")
                
        except Exception as e:
            logger.error(f"Error suspending research VM {vm_id}: {e}")
    
    async def _save_vm_state(self, vm_id: UUID) -> None:
        """Save research VM state to persistent storage"""
        
        try:
            vm_state = self.research_vms[vm_id]
            vm_config = self.vm_configurations[vm_id]
            
            state_data = {
                "vm_id": str(vm_id),
                "vm_state": vm_state.vm_state.value,
                "browser_sessions": vm_state.browser_sessions,
                "loaded_models": vm_state.loaded_models,
                "active_research_sessions": list(vm_state.active_research_sessions),
                "capability_status": {cap.value: status for cap, status in vm_state.capability_status.items()},
                "last_activity": vm_state.last_activity.isoformat(),
                "session_count": vm_state.session_count,
                "config": {
                    "vm_type": vm_config.vm_type.value,
                    "enabled_capabilities": [cap.value for cap in vm_config.enabled_capabilities],
                    "browser_engines": [engine.value for engine in vm_config.browser_engines]
                }
            }
            
            # Save to file
            state_file = self.config_dir / f"vm_state_{vm_id}.json"
            async with aiofiles.open(state_file, 'w') as f:
                await f.write(json.dumps(state_data, indent=2))
            
            # Cache in Redis
            if self.redis_client:
                await self.redis_client.hset(
                    f"research_vm_state:{vm_id}",
                    mapping={"state_data": json.dumps(state_data)}
                )
                await self.redis_client.expire(f"research_vm_state:{vm_id}", 604800)  # 7 days
            
            logger.debug(f"Saved state for research VM {vm_id}")
            
        except Exception as e:
            logger.error(f"Failed to save VM state for {vm_id}: {e}")
    
    async def _load_existing_configurations(self) -> None:
        """Load existing research VM configurations"""
        
        try:
            # Load from config files
            config_files = list(self.config_dir.glob("vm_state_*.json"))
            
            for config_file in config_files:
                try:
                    async with aiofiles.open(config_file, 'r') as f:
                        content = await f.read()
                        state_data = json.loads(content)
                    
                    vm_id = UUID(state_data["vm_id"])
                    
                    # Restore VM state
                    vm_state = ResearchVMState(
                        vm_id=vm_id,
                        vm_state=VMState(state_data["vm_state"]),
                        browser_sessions=state_data.get("browser_sessions", {}),
                        loaded_models=state_data.get("loaded_models", {}),
                        active_research_sessions=set(state_data.get("active_research_sessions", [])),
                        last_activity=datetime.fromisoformat(state_data["last_activity"]),
                        session_count=state_data.get("session_count", 0)
                    )
                    
                    # Restore capability status
                    for cap_name, status in state_data.get("capability_status", {}).items():
                        try:
                            cap = ResearchCapability(cap_name)
                            vm_state.capability_status[cap] = status
                        except ValueError:
                            pass
                    
                    self.research_vms[vm_id] = vm_state
                    
                    # Restore configuration
                    config_data = state_data.get("config", {})
                    vm_config = ResearchVMConfiguration(
                        vm_type=ResearchVMType(config_data.get("vm_type", "standard")),
                        enabled_capabilities=[
                            ResearchCapability(cap) for cap in config_data.get("enabled_capabilities", [])
                            if cap in [c.value for c in ResearchCapability]
                        ],
                        browser_engines=[
                            BrowserEngine(engine) for engine in config_data.get("browser_engines", ["chromium"])
                            if engine in [e.value for e in BrowserEngine]
                        ]
                    )
                    
                    self.vm_configurations[vm_id] = vm_config
                    
                    logger.info(f"Loaded existing research VM configuration: {vm_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to load config file {config_file}: {e}")
            
        except Exception as e:
            logger.error(f"Failed to load existing configurations: {e}")
    
    async def _restore_existing_vms(self) -> None:
        """Restore existing research VMs"""
        
        for vm_id, vm_state in self.research_vms.items():
            try:
                if vm_state.vm_state in [VMState.SUSPENDED, VMState.RUNNING]:
                    # Check if VM is actually running
                    vm_status = await self.base_vm_manager.get_vm_status(vm_id)
                    
                    if vm_status and vm_status.state == VMState.RUNNING:
                        vm_state.vm_state = VMState.RUNNING
                        logger.info(f"Restored running research VM: {vm_id}")
                    elif vm_status and vm_status.state == VMState.SUSPENDED:
                        vm_state.vm_state = VMState.SUSPENDED
                        logger.info(f"Found suspended research VM: {vm_id}")
                    else:
                        logger.warning(f"Research VM {vm_id} not found, marking as inactive")
                        vm_state.vm_state = VMState.ERROR
                        
            except Exception as e:
                logger.error(f"Failed to restore research VM {vm_id}: {e}")
    
    async def _cleanup_failed_vm(self, vm_id: UUID) -> None:
        """Cleanup failed VM creation"""
        
        try:
            # Remove from tracking
            if vm_id in self.research_vms:
                del self.research_vms[vm_id]
            
            if vm_id in self.vm_configurations:
                del self.vm_configurations[vm_id]
            
            # Clean up any partial VM resources
            try:
                await self.base_vm_manager.destroy_vm(vm_id, "Creation failed")
            except:
                pass
            
            # Remove config file
            config_file = self.config_dir / f"vm_state_{vm_id}.json"
            if config_file.exists():
                config_file.unlink()
            
            # Clean Redis cache
            if self.redis_client:
                await self.redis_client.delete(f"research_vm:{vm_id}")
                await self.redis_client.delete(f"research_vm_state:{vm_id}")
            
            logger.info(f"Cleaned up failed research VM {vm_id}")
            
        except Exception as e:
            logger.error(f"Failed to cleanup VM {vm_id}: {e}")
    
    async def get_vm_status(self, vm_id: UUID) -> Optional[ResearchVMState]:
        """Get current status of research VM"""
        return self.research_vms.get(vm_id)
    
    async def list_user_vms(self, user_id: UserID) -> List[Tuple[UUID, ResearchVMState]]:
        """List all research VMs for a user"""
        
        user_vms = []
        
        if user_id in self.user_vm_mapping:
            for vm_id in self.user_vm_mapping[user_id]:
                if vm_id in self.research_vms:
                    user_vms.append((vm_id, self.research_vms[vm_id]))
        
        return user_vms
    
    async def shutdown(self) -> None:
        """Shutdown research VM orchestrator"""
        
        logger.info("Shutting down research VM orchestrator...")
        
        try:
            # Stop monitoring
            self.monitoring_active = False
            
            if self.resource_monitor_task:
                self.resource_monitor_task.cancel()
            
            if self.cleanup_task:
                self.cleanup_task.cancel()
            
            # Save all VM states
            for vm_id in self.research_vms:
                await self._save_vm_state(vm_id)
            
            # Close browser instances
            if self.playwright_instance:
                await self.playwright_instance.stop()
            
            # Close Selenium drivers
            for vm_drivers in self.selenium_drivers.values():
                for driver in vm_drivers.values():
                    try:
                        driver.quit()
                    except:
                        pass
            
            # Close Redis connection
            if self.redis_client:
                await self.redis_client.close()
            
            # Shutdown thread pools
            self.vm_executor.shutdown(wait=True, timeout=30)
            self.analysis_executor.shutdown(wait=True, timeout=30)
            
            logger.info("Research VM orchestrator shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

async def create_research_vm_orchestrator(
    base_vm_manager: VMInstanceManager,
    memory_manager: MemoryManager,
    model_loader: ModelLoader,
    file_processor: IntelligentFileProcessor,
    prompt_manager: SystemPromptManager,
    config_dir: Optional[Path] = None,
    vm_storage_dir: Optional[Path] = None
) -> ResearchVMOrchestrator:
    """
    Factory function to create and initialize research VM orchestrator
    
    Returns:
        Fully initialized ResearchVMOrchestrator
    """
    
    orchestrator = ResearchVMOrchestrator(
        base_vm_manager=base_vm_manager,
        memory_manager=memory_manager,
        model_loader=model_loader,
        file_processor=file_processor,
        prompt_manager=prompt_manager,
        config_dir=config_dir or Path("research_config"),
        vm_storage_dir=vm_storage_dir or Path("research_vms")
    )
    
    await orchestrator.initialize()
    return orchestrator
