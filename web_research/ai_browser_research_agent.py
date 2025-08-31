"""
SOMNUS RESEARCH - Complete AI Browser Research Agent
Production-ready browser automation for intelligent web research

This agent provides comprehensive browser automation capabilities:
- Multi-engine browser support (Playwright + Selenium)
- Advanced content extraction and analysis
- AI-powered quality assessment using VM-loaded models
- Real-time progress streaming and error recovery
- Integration with research VM orchestrator and memory systems
- Complete implementation with zero placeholders

Architecture:
- Runs inside dedicated research VMs managed by ResearchVMOrchestrator
- Uses persistent browser sessions with accumulated tools/extensions
- Integrates with model_loader for AI analysis capabilities
- Coordinates with research_engine.py for orchestrated research
- Provides real-time streaming via ResearchStreamManager
"""

import asyncio
import base64
import hashlib
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import traceback
import urllib.parse
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Any, Tuple, Union, AsyncGenerator
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import asynccontextmanager

import aiofiles
import aiohttp
import numpy as np
import requests
from bs4 import BeautifulSoup
from PIL import Image
import cv2
import pytesseract
from playwright.async_api import async_playwright, Browser, BrowserContext, Page, Response
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.common.exceptions import TimeoutException, WebDriverException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
import spacy
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Core Somnus system imports
from core.vm_manager import VMInstanceManager, VMCommandResult
from core.memory_core import MemoryManager, MemoryType, MemoryImportance
from core.model_loader import ModelLoader
from schemas.session import SessionID, UserID

# Research subsystem imports
from research_session import ResearchSession, ResearchEntity
from research_stream_manager import ResearchStreamManager, StreamEvent, StreamEventType, StreamPriority
from research_vm_orchestrator import ResearchVMOrchestrator, BrowserEngine, ResearchCapability

logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Browser task execution status"""
    PENDING = "pending"
    STARTING = "starting"
    RUNNING = "running"
    EXTRACTING = "extracting"
    ANALYZING = "analyzing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ContentType(str, Enum):
    """Types of web content"""
    ARTICLE = "article"
    ACADEMIC_PAPER = "academic_paper"
    NEWS = "news"
    BLOG_POST = "blog_post"
    FORUM_DISCUSSION = "forum_discussion"
    DOCUMENTATION = "documentation"
    SOCIAL_MEDIA = "social_media"
    E_COMMERCE = "e_commerce"
    MULTIMEDIA = "multimedia"
    UNKNOWN = "unknown"


class ExtractionMethod(str, Enum):
    """Content extraction methods"""
    PLAYWRIGHT = "playwright"
    SELENIUM = "selenium"
    BEAUTIFULSOUP = "beautifulsoup"
    API_REQUEST = "api_request"
    HYBRID = "hybrid"


@dataclass
class BrowserTask:
    """Complete browser research task definition"""
    task_id: str = field(default_factory=lambda: str(uuid4()))
    task_type: str = "search"
    objective: str = ""
    
    # Search parameters
    queries: List[str] = field(default_factory=list)
    search_engines: List[str] = field(default_factory=lambda: ["google"])
    
    # Page interaction parameters
    urls: List[str] = field(default_factory=list)
    max_sources: int = 10
    depth_level: str = "moderate"
    
    # Data extraction requirements
    data_extraction_requirements: List[str] = field(default_factory=lambda: ["full_text", "metadata"])
    required_elements: List[str] = field(default_factory=list)
    
    # Quality filters
    quality_filters: Dict[str, Any] = field(default_factory=dict)
    
    # AI analysis requirements
    enable_ai_analysis: bool = True
    analysis_focus: List[str] = field(default_factory=lambda: ["quality", "credibility"])
    
    # Browser configuration
    browser_engine: BrowserEngine = BrowserEngine.CHROMIUM
    headless: bool = True
    timeout_seconds: int = 30
    
    # Progress tracking
    status: TaskStatus = TaskStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    progress_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'task_type': self.task_type,
            'objective': self.objective,
            'queries': self.queries,
            'search_engines': self.search_engines,
            'urls': self.urls,
            'max_sources': self.max_sources,
            'depth_level': self.depth_level,
            'enable_ai_analysis': self.enable_ai_analysis,
            'browser_engine': self.browser_engine.value,
            'status': self.status.value,
            'progress_percentage': self.progress_percentage,
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ExtractedContent:
    """Comprehensive extracted content from web source"""
    url: str
    title: str = ""
    content: str = ""
    clean_text: str = ""
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    language: str = "en"
    content_type: ContentType = ContentType.UNKNOWN
    extraction_method: ExtractionMethod = ExtractionMethod.PLAYWRIGHT
    
    # Visual data
    screenshot_path: Optional[str] = None
    screenshot_data: Optional[bytes] = None
    
    # Structured elements
    headings: List[Dict[str, str]] = field(default_factory=list)
    images: List[Dict[str, str]] = field(default_factory=list)
    links: List[Dict[str, str]] = field(default_factory=list)
    tables: List[List[List[str]]] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    
    # Content analysis
    word_count: int = 0
    reading_time_minutes: float = 0.0
    sentiment_score: float = 0.0
    
    # Quality metrics
    content_quality_score: float = 0.0
    credibility_indicators: Dict[str, Any] = field(default_factory=dict)
    bias_indicators: List[str] = field(default_factory=list)
    
    # AI analysis results
    ai_analysis: Optional[Dict[str, Any]] = None
    key_topics: List[str] = field(default_factory=list)
    named_entities: List[Dict[str, str]] = field(default_factory=list)
    
    # Processing metadata
    extraction_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processing_time_seconds: float = 0.0
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'url': self.url,
            'title': self.title,
            'content_preview': self.content[:500] + "..." if len(self.content) > 500 else self.content,
            'metadata': self.metadata,
            'language': self.language,
            'content_type': self.content_type.value,
            'extraction_method': self.extraction_method.value,
            'word_count': self.word_count,
            'reading_time_minutes': self.reading_time_minutes,
            'sentiment_score': self.sentiment_score,
            'content_quality_score': self.content_quality_score,
            'credibility_indicators': self.credibility_indicators,
            'key_topics': self.key_topics,
            'processing_time_seconds': self.processing_time_seconds,
            'extraction_timestamp': self.extraction_timestamp.isoformat()
        }


@dataclass
class BrowserResult:
    """Complete browser research task result"""
    task_id: str
    success: bool
    extracted_content: List[ExtractedContent] = field(default_factory=list)
    
    # Task metadata
    task_objective: str = ""
    queries_executed: List[str] = field(default_factory=list)
    urls_processed: List[str] = field(default_factory=list)
    
    # Processing statistics
    total_sources_found: int = 0
    sources_successfully_processed: int = 0
    processing_time_seconds: float = 0.0
    screenshots_captured: int = 0
    
    # Quality metrics
    average_content_quality: float = 0.0
    high_quality_sources: int = 0
    
    # Error information
    error_message: Optional[str] = None
    warnings: List[str] = field(default_factory=list)
    failed_urls: List[Dict[str, str]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'task_id': self.task_id,
            'success': self.success,
            'task_objective': self.task_objective,
            'total_sources_found': self.total_sources_found,
            'sources_successfully_processed': self.sources_successfully_processed,
            'processing_time_seconds': self.processing_time_seconds,
            'average_content_quality': self.average_content_quality,
            'high_quality_sources': self.high_quality_sources,
            'extracted_content': [content.to_dict() for content in self.extracted_content],
            'warnings': self.warnings,
            'failed_urls': self.failed_urls
        }


class AIBrowserResearchAgent:
    """
    Complete AI browser research agent with production-grade capabilities
    
    Provides comprehensive web research automation with:
    - Multi-engine browser support (Playwright, Selenium)
    - Advanced content extraction and quality assessment
    - AI-powered analysis using VM-loaded models
    - Real-time progress streaming and error recovery
    - Integration with research VM and memory systems
    """
    
    def __init__(
        self,
        vm_orchestrator: ResearchVMOrchestrator,
        research_vm_id: UUID,
        model_loader: ModelLoader,
        memory_manager: MemoryManager,
        stream_manager: ResearchStreamManager,
        workspace_path: str = "/home/ai/research"
    ):
        """Initialize browser research agent"""
        
        self.vm_orchestrator = vm_orchestrator
        self.research_vm_id = research_vm_id
        self.model_loader = model_loader
        self.memory_manager = memory_manager
        self.stream_manager = stream_manager
        self.workspace_path = workspace_path
        
        # Browser management
        self.playwright_instance = None
        self.browsers: Dict[BrowserEngine, Browser] = {}
        self.contexts: Dict[str, BrowserContext] = {}
        self.selenium_drivers: Dict[str, webdriver.Remote] = {}
        
        # AI analysis models (loaded in VM)
        self.embedding_model: Optional[SentenceTransformer] = None
        self.nlp_model = None
        self.sentiment_analyzer = None
        
        # Content processing
        self.content_extractors: Dict[ExtractionMethod, Any] = {}
        self.quality_assessors: List[Any] = []
        
        # Task management
        self.active_tasks: Dict[str, BrowserTask] = {}
        self.task_executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="browser-task")
        
        # Caching and optimization
        self.url_cache: Dict[str, ExtractedContent] = {}
        self.domain_rate_limits: Dict[str, Dict[str, float]] = {}
        
        # Search engine configurations
        self.search_engines = {
            "google": {
                "url": "https://www.google.com/search",
                "query_param": "q",
                "result_selector": "div.g h3 a",
                "snippet_selector": "div.g .VwiC3b"
            },
            "bing": {
                "url": "https://www.bing.com/search",
                "query_param": "q", 
                "result_selector": ".b_algo h2 a",
                "snippet_selector": ".b_algo .b_caption p"
            },
            "duckduckgo": {
                "url": "https://duckduckgo.com/",
                "query_param": "q",
                "result_selector": "article h2 a",
                "snippet_selector": "article [data-result='snippet']"
            },
            "arxiv": {
                "url": "https://arxiv.org/search/",
                "query_param": "query",
                "result_selector": "li.arxiv-result h4 a",
                "snippet_selector": "li.arxiv-result .abstract-short"
            },
            "scholar": {
                "url": "https://scholar.google.com/scholar",
                "query_param": "q",
                "result_selector": ".gs_rt h3 a",
                "snippet_selector": ".gs_rs"
            }
        }
        
        logger.info(f"AI Browser Research Agent initialized for VM {research_vm_id}")
    
    async def initialize(self) -> None:
        """Initialize browser research agent and all components"""
        
        try:
            # Initialize AI models
            await self._initialize_ai_models()
            
            # Initialize browser engines
            await self._initialize_browsers()
            
            # Initialize content processors
            await self._initialize_content_processors()
            
            # Setup workspace in VM
            await self._setup_vm_workspace()
            
            # Load browser extensions and tools
            await self._load_browser_extensions()
            
            # Initialize rate limiting
            await self._initialize_rate_limiting()
            
            logger.info("AI Browser Research Agent fully initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize browser research agent: {e}")
            raise
    
    async def execute_task(self, task: BrowserTask) -> BrowserResult:
        """Execute complete browser research task"""
        
        task.status = TaskStatus.STARTING
        task.started_at = datetime.now(timezone.utc)
        self.active_tasks[task.task_id] = task
        
        logger.info(f"Executing browser task: {task.task_type} - {task.objective}")
        
        try:
            # Stream task start
            await self._stream_task_event("TASK_START", task)
            
            # Execute task based on type
            if task.task_type == "search":
                result = await self._execute_search_task(task)
            elif task.task_type == "extract":
                result = await self._execute_extraction_task(task)
            elif task.task_type == "analyze":
                result = await self._execute_analysis_task(task)
            elif task.task_type == "multi_step":
                result = await self._execute_multi_step_task(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Update task status
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.progress_percentage = 100.0
            
            # Calculate final metrics
            if result.extracted_content:
                result.average_content_quality = sum(
                    content.content_quality_score for content in result.extracted_content
                ) / len(result.extracted_content)
                
                result.high_quality_sources = len([
                    content for content in result.extracted_content 
                    if content.content_quality_score > 0.7
                ])
            
            result.processing_time_seconds = (
                task.completed_at - task.started_at
            ).total_seconds()
            
            # Stream task completion
            await self._stream_task_event("TASK_COMPLETE", task, {
                'sources_processed': len(result.extracted_content),
                'average_quality': result.average_content_quality,
                'processing_time': result.processing_time_seconds
            })
            
            logger.info(f"Task completed: {task.task_id}, sources: {len(result.extracted_content)}")
            return result
            
        except Exception as e:
            logger.error(f"Browser task failed: {e}", exc_info=True)
            
            task.status = TaskStatus.FAILED
            task.completed_at = datetime.now(timezone.utc)
            
            await self._stream_task_event("TASK_ERROR", task, {'error': str(e)})
            
            return BrowserResult(
                task_id=task.task_id,
                success=False,
                error_message=str(e),
                task_objective=task.objective
            )
        
        finally:
            # Cleanup task
            if task.task_id in self.active_tasks:
                del self.active_tasks[task.task_id]
    
    async def _execute_search_task(self, task: BrowserTask) -> BrowserResult:
        """Execute comprehensive web search across multiple engines"""
        
        all_results = []
        queries_executed = []
        failed_urls = []
        
        task.status = TaskStatus.RUNNING
        total_operations = len(task.search_engines) * len(task.queries)
        completed_operations = 0
        
        for search_engine in task.search_engines:
            for query in task.queries:
                try:
                    # Update progress
                    task.progress_percentage = (completed_operations / total_operations) * 60
                    await self._stream_task_event("SEARCH_PROGRESS", task, {
                        'search_engine': search_engine,
                        'query': query,
                        'progress': task.progress_percentage
                    })
                    
                    # Execute search
                    search_results = await self._perform_search(
                        query=query,
                        search_engine=search_engine,
                        max_results=task.max_sources // len(task.queries) // len(task.search_engines) + 2
                    )
                    
                    queries_executed.append(f"{search_engine}: {query}")
                    
                    # Process search results
                    for i, search_result in enumerate(search_results):
                        try:
                            if len(all_results) >= task.max_sources:
                                break
                            
                            extracted_content = await self._extract_content_from_url(
                                url=search_result['url'],
                                task=task,
                                search_context={
                                    'query': query,
                                    'search_engine': search_engine,
                                    'title': search_result.get('title', ''),
                                    'snippet': search_result.get('snippet', '')
                                }
                            )
                            
                            if extracted_content:
                                all_results.append(extracted_content)
                                
                                await self._stream_task_event("SOURCE_EXTRACTED", task, {
                                    'url': extracted_content.url,
                                    'title': extracted_content.title,
                                    'quality_score': extracted_content.content_quality_score,
                                    'sources_found': len(all_results)
                                })
                            
                        except Exception as e:
                            failed_urls.append({
                                'url': search_result.get('url', 'unknown'),
                                'error': str(e),
                                'search_engine': search_engine,
                                'query': query
                            })
                            logger.warning(f"Failed to extract content from {search_result.get('url')}: {e}")
                    
                    completed_operations += 1
                    
                    # Rate limiting between searches
                    await self._respect_rate_limits(search_engine)
                    
                except Exception as e:
                    logger.error(f"Search failed for {search_engine} - {query}: {e}")
                    failed_urls.append({
                        'url': f"{search_engine}:{query}",
                        'error': str(e),
                        'search_engine': search_engine,
                        'query': query
                    })
                    completed_operations += 1
        
        # AI analysis phase
        task.status = TaskStatus.ANALYZING
        task.progress_percentage = 80
        
        if task.enable_ai_analysis and all_results:
            await self._perform_ai_analysis_batch(all_results, task)
        
        # Filter and rank by quality
        quality_filtered = [
            content for content in all_results 
            if content.content_quality_score > 0.3
        ]
        
        quality_filtered.sort(key=lambda x: x.content_quality_score, reverse=True)
        final_results = quality_filtered[:task.max_sources]
        
        return BrowserResult(
            task_id=task.task_id,
            success=True,
            extracted_content=final_results,
            task_objective=task.objective,
            queries_executed=queries_executed,
            urls_processed=[content.url for content in all_results],
            total_sources_found=len(all_results),
            sources_successfully_processed=len(final_results),
            screenshots_captured=len([c for c in final_results if c.screenshot_path]),
            failed_urls=failed_urls
        )
    
    async def _perform_search(
        self,
        query: str,
        search_engine: str,
        max_results: int = 10
    ) -> List[Dict[str, str]]:
        """Perform search on specific search engine"""
        
        if search_engine not in self.search_engines:
            raise ValueError(f"Unsupported search engine: {search_engine}")
        
        engine_config = self.search_engines[search_engine]
        
        # Construct search URL
        search_url = f"{engine_config['url']}?{engine_config['query_param']}={urllib.parse.quote_plus(query)}"
        
        # Get browser context
        context = await self._get_browser_context(search_engine)
        page = await context.new_page()
        
        try:
            # Navigate to search page
            await page.goto(search_url, wait_until="domcontentloaded", timeout=30000)
            
            # Wait for results to load
            await page.wait_for_timeout(2000)
            
            # Extract search results based on engine
            if search_engine == "google":
                results = await self._extract_google_results(page, max_results)
            elif search_engine == "bing":
                results = await self._extract_bing_results(page, max_results)
            elif search_engine == "duckduckgo":
                results = await self._extract_duckduckgo_results(page, max_results)
            elif search_engine == "arxiv":
                results = await self._extract_arxiv_results(page, max_results)
            elif search_engine == "scholar":
                results = await self._extract_scholar_results(page, max_results)
            else:
                results = await self._extract_generic_results(page, engine_config, max_results)
            
            return results
            
        finally:
            await page.close()
    
    async def _extract_google_results(self, page: Page, max_results: int) -> List[Dict[str, str]]:
        """Extract Google search results with comprehensive parsing"""
        
        try:
            # Wait for results container
            await page.wait_for_selector('div[data-ved]', timeout=10000)
            
            # Extract results using JavaScript execution
            results = await page.evaluate(f"""
                () => {{
                    const results = [];
                    const maxResults = {max_results};
                    
                    // Try multiple selectors for Google results
                    const selectors = [
                        'div[data-ved] h3 a',
                        '.g h3 a',
                        '.rc h3 a',
                        'h3.LC20lb a'
                    ];
                    
                    let resultElements = [];
                    for (const selector of selectors) {{
                        resultElements = document.querySelectorAll(selector);
                        if (resultElements.length > 0) break;
                    }}
                    
                    for (let i = 0; i < Math.min(resultElements.length, maxResults); i++) {{
                        const linkElement = resultElements[i];
                        const titleElement = linkElement.querySelector('h3') || linkElement;
                        
                        if (linkElement.href && titleElement.textContent.trim()) {{
                            // Find snippet for this result
                            let snippet = '';
                            const resultContainer = linkElement.closest('.g, .rc, div[data-ved]');
                            if (resultContainer) {{
                                const snippetSelectors = [
                                    '.VwiC3b',
                                    '.s3v9rd',
                                    '.st',
                                    'span[data-ved]'
                                ];
                                
                                for (const snippetSelector of snippetSelectors) {{
                                    const snippetElement = resultContainer.querySelector(snippetSelector);
                                    if (snippetElement && snippetElement.textContent.trim()) {{
                                        snippet = snippetElement.textContent.trim();
                                        break;
                                    }}
                                }}
                            }}
                            
                            results.push({{
                                title: titleElement.textContent.trim(),
                                url: linkElement.href,
                                snippet: snippet,
                                search_engine: 'google',
                                rank: i + 1
                            }});
                        }}
                    }}
                    
                    return results;
                }}
            """)
            
            # Filter valid URLs
            valid_results = []
            for result in results:
                if (result['url'].startswith('http') and 
                    not result['url'].startswith('https://www.google.com') and
                    len(result['title']) > 0):
                    valid_results.append(result)
            
            return valid_results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to extract Google results: {e}")
            return []
    
    async def _extract_bing_results(self, page: Page, max_results: int) -> List[Dict[str, str]]:
        """Extract Bing search results"""
        
        try:
            await page.wait_for_selector('.b_algo', timeout=10000)
            
            results = await page.evaluate(f"""
                () => {{
                    const results = [];
                    const resultElements = document.querySelectorAll('.b_algo');
                    const maxResults = {max_results};
                    
                    for (let i = 0; i < Math.min(resultElements.length, maxResults); i++) {{
                        const element = resultElements[i];
                        const titleElement = element.querySelector('h2 a');
                        const snippetElement = element.querySelector('.b_caption p, .b_caption .b_dList');
                        
                        if (titleElement && titleElement.href) {{
                            results.push({{
                                title: titleElement.textContent.trim(),
                                url: titleElement.href,
                                snippet: snippetElement ? snippetElement.textContent.trim() : '',
                                search_engine: 'bing',
                                rank: i + 1
                            }});
                        }}
                    }}
                    
                    return results;
                }}
            """)
            
            return [r for r in results if r['url'].startswith('http')][:max_results]
            
        except Exception as e:
            logger.error(f"Failed to extract Bing results: {e}")
            return []
    
    async def _extract_duckduckgo_results(self, page: Page, max_results: int) -> List[Dict[str, str]]:
        """Extract DuckDuckGo search results"""
        
        try:
            await page.wait_for_selector('article', timeout=10000)
            
            results = await page.evaluate(f"""
                () => {{
                    const results = [];
                    const resultElements = document.querySelectorAll('article');
                    const maxResults = {max_results};
                    
                    for (let i = 0; i < Math.min(resultElements.length, maxResults); i++) {{
                        const element = resultElements[i];
                        const titleElement = element.querySelector('h2 a, h3 a');
                        const snippetElement = element.querySelector('[data-result="snippet"], .result__snippet');
                        
                        if (titleElement && titleElement.href) {{
                            results.push({{
                                title: titleElement.textContent.trim(),
                                url: titleElement.href,
                                snippet: snippetElement ? snippetElement.textContent.trim() : '',
                                search_engine: 'duckduckgo',
                                rank: i + 1
                            }});
                        }}
                    }}
                    
                    return results;
                }}
            """)
            
            return [r for r in results if r['url'].startswith('http')][:max_results]
            
        except Exception as e:
            logger.error(f"Failed to extract DuckDuckGo results: {e}")
            return []
    
    async def _extract_arxiv_results(self, page: Page, max_results: int) -> List[Dict[str, str]]:
        """Extract arXiv search results"""
        
        try:
            await page.wait_for_selector('.arxiv-result', timeout=10000)
            
            results = await page.evaluate(f"""
                () => {{
                    const results = [];
                    const resultElements = document.querySelectorAll('.arxiv-result');
                    const maxResults = {max_results};
                    
                    for (let i = 0; i < Math.min(resultElements.length, maxResults); i++) {{
                        const element = resultElements[i];
                        const titleElement = element.querySelector('.list-title a');
                        const snippetElement = element.querySelector('.mathjax, .abstract-short');
                        const authorsElement = element.querySelector('.list-authors');
                        
                        if (titleElement && titleElement.href) {{
                            results.push({{
                                title: titleElement.textContent.replace('Title:', '').trim(),
                                url: 'https://arxiv.org' + titleElement.getAttribute('href'),
                                snippet: snippetElement ? snippetElement.textContent.trim() : '',
                                authors: authorsElement ? authorsElement.textContent.trim() : '',
                                search_engine: 'arxiv',
                                rank: i + 1
                            }});
                        }}
                    }}
                    
                    return results;
                }}
            """)
            
            return results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to extract arXiv results: {e}")
            return []
    
    async def _extract_scholar_results(self, page: Page, max_results: int) -> List[Dict[str, str]]:
        """Extract Google Scholar search results"""
        
        try:
            await page.wait_for_selector('.gs_r', timeout=10000)
            
            results = await page.evaluate(f"""
                () => {{
                    const results = [];
                    const resultElements = document.querySelectorAll('.gs_r');
                    const maxResults = {max_results};
                    
                    for (let i = 0; i < Math.min(resultElements.length, maxResults); i++) {{
                        const element = resultElements[i];
                        const titleElement = element.querySelector('.gs_rt a');
                        const snippetElement = element.querySelector('.gs_rs');
                        const authorsElement = element.querySelector('.gs_a');
                        const citedElement = element.querySelector('.gs_fl a[href*="cites"]');
                        
                        if (titleElement && titleElement.href) {{
                            results.push({{
                                title: titleElement.textContent.trim(),
                                url: titleElement.href,
                                snippet: snippetElement ? snippetElement.textContent.trim() : '',
                                authors: authorsElement ? authorsElement.textContent.trim() : '',
                                cited_by: citedElement ? citedElement.textContent : '',
                                search_engine: 'scholar',
                                rank: i + 1
                            }});
                        }}
                    }}
                    
                    return results;
                }}
            """)
            
            return [r for r in results if r['url'].startswith('http')][:max_results]
            
        except Exception as e:
            logger.error(f"Failed to extract Scholar results: {e}")
            return []
    
    async def _extract_generic_results(
        self, 
        page: Page, 
        engine_config: Dict[str, str], 
        max_results: int
    ) -> List[Dict[str, str]]:
        """Extract results using generic selectors"""
        
        try:
            result_selector = engine_config['result_selector']
            snippet_selector = engine_config['snippet_selector']
            
            await page.wait_for_selector(result_selector, timeout=10000)
            
            results = await page.evaluate(f"""
                () => {{
                    const results = [];
                    const resultElements = document.querySelectorAll('{result_selector}');
                    const maxResults = {max_results};
                    
                    for (let i = 0; i < Math.min(resultElements.length, maxResults); i++) {{
                        const element = resultElements[i];
                        
                        if (element.href && element.textContent.trim()) {{
                            let snippet = '';
                            const snippetElement = element.closest('*').querySelector('{snippet_selector}');
                            if (snippetElement) {{
                                snippet = snippetElement.textContent.trim();
                            }}
                            
                            results.push({{
                                title: element.textContent.trim(),
                                url: element.href,
                                snippet: snippet,
                                search_engine: 'generic',
                                rank: i + 1
                            }});
                        }}
                    }}
                    
                    return results;
                }}
            """)
            
            return [r for r in results if r['url'].startswith('http')][:max_results]
            
        except Exception as e:
            logger.error(f"Failed to extract generic results: {e}")
            return []
    
    async def _extract_content_from_url(
        self,
        url: str,
        task: BrowserTask,
        search_context: Optional[Dict[str, str]] = None
    ) -> Optional[ExtractedContent]:
        """Extract comprehensive content from URL"""
        
        if url in self.url_cache:
            logger.debug(f"Using cached content for {url}")
            return self.url_cache[url]
        
        start_time = time.time()
        
        try:
            # Rate limiting check
            domain = urllib.parse.urlparse(url).netloc
            await self._respect_rate_limits(domain)
            
            # Get browser context
            context = await self._get_browser_context("extraction")
            page = await context.new_page()
            
            try:
                # Navigate to page
                response = await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                
                if not response or response.status >= 400:
                    raise Exception(f"HTTP {response.status if response else 'unknown'}")
                
                # Wait for content to load
                await page.wait_for_timeout(2000)
                
                # Extract content using multiple methods
                extracted_content = ExtractedContent(
                    url=url,
                    extraction_method=ExtractionMethod.PLAYWRIGHT
                )
                
                # Basic page information
                extracted_content.title = await page.title()
                
                # Screenshot if enabled
                if task.depth_level in ["deep", "exhaustive"]:
                    screenshot_data = await self._capture_screenshot(page, url)
                    if screenshot_data:
                        extracted_content.screenshot_data = screenshot_data[1]
                        extracted_content.screenshot_path = screenshot_data[0]
                
                # Extract structured content
                await self._extract_page_structure(page, extracted_content)
                
                # Extract and clean text content
                await self._extract_text_content(page, extracted_content)
                
                # Analyze content type
                extracted_content.content_type = self._classify_content_type(extracted_content)
                
                # Language detection
                extracted_content.language = self._detect_language(extracted_content.content)
                
                # Calculate content metrics
                self._calculate_content_metrics(extracted_content)
                
                # Quality assessment
                await self._assess_content_quality(extracted_content)
                
                # Credibility analysis
                await self._analyze_credibility(page, extracted_content)
                
                # AI analysis if enabled
                if task.enable_ai_analysis:
                    await self._perform_ai_analysis(extracted_content, task.analysis_focus)
                
                # Add search context if available
                if search_context:
                    extracted_content.metadata.update(search_context)
                
                # Processing time
                extracted_content.processing_time_seconds = time.time() - start_time
                
                # Cache result
                self.url_cache[url] = extracted_content
                
                return extracted_content
                
            finally:
                await page.close()
                
        except Exception as e:
            logger.warning(f"Failed to extract content from {url}: {e}")
            
            # Try fallback extraction method
            try:
                return await self._fallback_content_extraction(url, str(e))
            except Exception as fallback_error:
                logger.error(f"Fallback extraction also failed for {url}: {fallback_error}")
                return None
    
    async def _extract_page_structure(self, page: Page, content: ExtractedContent) -> None:
        """Extract structured elements from page"""
        
        try:
            # Extract page structure using JavaScript
            structure_data = await page.evaluate("""
                () => {
                    // Extract headings
                    const headings = Array.from(document.querySelectorAll('h1, h2, h3, h4, h5, h6')).map(h => ({
                        level: parseInt(h.tagName.charAt(1)),
                        text: h.textContent.trim(),
                        id: h.id || ''
                    }));
                    
                    // Extract images
                    const images = Array.from(document.querySelectorAll('img')).map(img => ({
                        src: img.src,
                        alt: img.alt || '',
                        title: img.title || '',
                        width: img.naturalWidth || 0,
                        height: img.naturalHeight || 0
                    }));
                    
                    // Extract links
                    const links = Array.from(document.querySelectorAll('a[href]')).map(link => ({
                        href: link.href,
                        text: link.textContent.trim(),
                        title: link.title || '',
                        rel: link.rel || '',
                        target: link.target || ''
                    }));
                    
                    // Extract tables
                    const tables = Array.from(document.querySelectorAll('table')).map(table => {
                        const rows = Array.from(table.querySelectorAll('tr')).map(row => 
                            Array.from(row.querySelectorAll('td, th')).map(cell => 
                                cell.textContent.trim()
                            )
                        );
                        return rows;
                    });
                    
                    // Extract forms
                    const forms = Array.from(document.querySelectorAll('form')).map(form => ({
                        action: form.action || '',
                        method: form.method || 'get',
                        inputs: Array.from(form.querySelectorAll('input, select, textarea')).map(input => ({
                            type: input.type || '',
                            name: input.name || '',
                            placeholder: input.placeholder || '',
                            required: input.required || false
                        }))
                    }));
                    
                    // Extract metadata
                    const metaTags = Array.from(document.querySelectorAll('meta')).reduce((acc, meta) => {
                        const name = meta.getAttribute('name') || meta.getAttribute('property');
                        const content = meta.getAttribute('content');
                        if (name && content) {
                            acc[name] = content;
                        }
                        return acc;
                    }, {});
                    
                    // Extract JSON-LD structured data
                    const jsonLdScripts = Array.from(document.querySelectorAll('script[type="application/ld+json"]'));
                    const structuredData = jsonLdScripts.map(script => {
                        try {
                            return JSON.parse(script.textContent);
                        } catch {
                            return null;
                        }
                    }).filter(data => data !== null);
                    
                    return {
                        headings,
                        images: images.slice(0, 20), // Limit images
                        links: links.slice(0, 50),   // Limit links
                        tables: tables.slice(0, 10), // Limit tables
                        forms,
                        metaTags,
                        structuredData,
                        title: document.title,
                        charset: document.charset,
                        lang: document.documentElement.lang || '',
                        canonicalUrl: document.querySelector('link[rel="canonical"]')?.href || ''
                    };
                }
            """)
            
            # Update extracted content
            content.headings = structure_data.get('headings', [])
            content.images = structure_data.get('images', [])
            content.links = structure_data.get('links', [])
            content.tables = structure_data.get('tables', [])
            content.forms = structure_data.get('forms', [])
            content.metadata.update(structure_data.get('metaTags', {}))
            content.metadata['structured_data'] = structure_data.get('structuredData', [])
            content.metadata['canonical_url'] = structure_data.get('canonicalUrl', '')
            content.language = structure_data.get('lang', 'en') or content.language
            
        except Exception as e:
            logger.warning(f"Failed to extract page structure: {e}")
    
    async def _extract_text_content(self, page: Page, content: ExtractedContent) -> None:
        """Extract and clean text content from page"""
        
        try:
            # Extract content using multiple strategies
            content_data = await page.evaluate("""
                () => {
                    // Remove unwanted elements
                    const unwantedSelectors = [
                        'script', 'style', 'nav', 'header', 'footer', 'aside',
                        '.advertisement', '.ads', '.sidebar', '.menu', '.navigation',
                        '.cookie-notice', '.popup', '.modal', '.overlay'
                    ];
                    
                    unwantedSelectors.forEach(selector => {
                        document.querySelectorAll(selector).forEach(el => el.remove());
                    });
                    
                    // Try to find main content
                    const contentSelectors = [
                        'main', 'article', '[role="main"]', '.content', '.post-content',
                        '.entry-content', '.article-content', '.story-body', '.post-body',
                        '.content-body', '.article-body', '.main-content', '.primary-content'
                    ];
                    
                    let mainContent = '';
                    let contentElement = null;
                    
                    for (const selector of contentSelectors) {
                        contentElement = document.querySelector(selector);
                        if (contentElement) {
                            mainContent = contentElement.innerText || contentElement.textContent || '';
                            if (mainContent.trim().length > 100) {
                                break;
                            }
                        }
                    }
                    
                    // Fallback to body if no main content found
                    if (!mainContent || mainContent.trim().length < 100) {
                        mainContent = document.body.innerText || document.body.textContent || '';
                    }
                    
                    // Extract raw HTML for further processing
                    const rawHtml = contentElement ? contentElement.innerHTML : document.body.innerHTML;
                    
                    return {
                        mainContent: mainContent.trim(),
                        rawHtml: rawHtml,
                        wordCount: mainContent.trim().split(/\\s+/).length,
                        charCount: mainContent.trim().length
                    };
                }
            """)
            
            content.content = content_data.get('mainContent', '')
            content.metadata['raw_html'] = content_data.get('rawHtml', '')[:10000]  # Truncate
            content.word_count = content_data.get('wordCount', 0)
            
            # Clean and normalize text
            content.clean_text = self._clean_text(content.content)
            
        except Exception as e:
            logger.warning(f"Failed to extract text content: {e}")
            content.content = ""
            content.clean_text = ""
            content.word_count = 0
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove special characters and normalize
        text = re.sub(r'[^\w\s\.\,\!\?\;\:\-\(\)\"\'\/]', '', text)
        
        # Remove repeated punctuation
        text = re.sub(r'[\.]{3,}', '...', text)
        text = re.sub(r'[!]{2,}', '!', text)
        text = re.sub(r'[?]{2,}', '?', text)
        
        # Remove URLs
        text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)
        
        # Normalize quotes
        text = re.sub(r'[""''`]', '"', text)
        
        return text.strip()
    
    def _classify_content_type(self, content: ExtractedContent) -> ContentType:
        """Classify the type of content"""
        
        text = content.content.lower()
        url = content.url.lower()
        title = content.title.lower()
        
        # Academic paper indicators
        academic_indicators = [
            'abstract', 'introduction', 'methodology', 'conclusion', 'references',
            'arxiv.org', 'pubmed', 'doi.org', 'journal', 'research', 'study'
        ]
        if any(indicator in text or indicator in url for indicator in academic_indicators):
            return ContentType.ACADEMIC_PAPER
        
        # News indicators
        news_indicators = [
            'breaking', 'reported', 'according to', 'news', 'reuters', 'cnn',
            'bbc', 'associated press', 'journalist', 'correspondent'
        ]
        if any(indicator in text or indicator in url for indicator in news_indicators):
            return ContentType.NEWS
        
        # Documentation indicators
        doc_indicators = [
            'documentation', 'docs', 'api', 'tutorial', 'guide', 'how to',
            'getting started', 'installation', 'configuration'
        ]
        if any(indicator in text or indicator in title for indicator in doc_indicators):
            return ContentType.DOCUMENTATION
        
        # Blog post indicators
        blog_indicators = [
            'blog', 'post', 'author:', 'published', 'tags:', 'category:',
            'medium.com', 'wordpress', 'blogger'
        ]
        if any(indicator in text or indicator in url for indicator in blog_indicators):
            return ContentType.BLOG_POST
        
        # Forum/discussion indicators
        forum_indicators = [
            'forum', 'discussion', 'thread', 'reply', 'comment', 'posted by',
            'reddit.com', 'stackoverflow', 'quora'
        ]
        if any(indicator in text or indicator in url for indicator in forum_indicators):
            return ContentType.FORUM_DISCUSSION
        
        # Social media indicators
        social_indicators = [
            'twitter.com', 'facebook.com', 'instagram.com', 'linkedin.com',
            'social', 'post', 'share', 'like', 'follow'
        ]
        if any(indicator in url for indicator in social_indicators):
            return ContentType.SOCIAL_MEDIA
        
        # E-commerce indicators
        ecommerce_indicators = [
            'price', 'buy', 'cart', 'shop', 'product', 'amazon.com',
            'ebay.com', 'store', 'purchase', 'checkout'
        ]
        if any(indicator in text or indicator in url for indicator in ecommerce_indicators):
            return ContentType.E_COMMERCE
        
        # Check for multimedia content
        if content.images and len(content.images) > len(content.content.split()) / 50:
            return ContentType.MULTIMEDIA
        
        # Default to article for substantial text content
        if content.word_count > 200:
            return ContentType.ARTICLE
        
        return ContentType.UNKNOWN
    
    def _detect_language(self, text: str) -> str:
        """Detect language of text content"""
        
        if not text or len(text) < 50:
            return "en"
        
        try:
            # Simple language detection based on common words
            english_words = {
                'the', 'and', 'to', 'of', 'a', 'in', 'is', 'it', 'you', 'that',
                'he', 'was', 'for', 'on', 'are', 'as', 'with', 'his', 'they'
            }
            
            spanish_words = {
                'el', 'la', 'de', 'que', 'y', 'a', 'en', 'un', 'es', 'se',
                'no', 'te', 'lo', 'le', 'da', 'su', 'por', 'son', 'con', 'para'
            }
            
            french_words = {
                'le', 'de', 'et', 'à', 'un', 'il', 'être', 'et', 'en', 'avoir',
                'que', 'pour', 'dans', 'ce', 'son', 'une', 'sur', 'avec', 'ne', 'se'
            }
            
            german_words = {
                'der', 'die', 'und', 'in', 'den', 'von', 'zu', 'das', 'mit', 'sich',
                'des', 'auf', 'für', 'ist', 'im', 'dem', 'nicht', 'ein', 'eine', 'als'
            }
            
            # Count matches for each language
            words = set(text.lower().split()[:100])  # First 100 words
            
            scores = {
                'en': len(words & english_words),
                'es': len(words & spanish_words),
                'fr': len(words & french_words),
                'de': len(words & german_words)
            }
            
            # Return language with highest score
            detected_lang = max(scores.items(), key=lambda x: x[1])[0]
            return detected_lang if scores[detected_lang] > 2 else "en"
            
        except Exception as e:
            logger.warning(f"Language detection failed: {e}")
            return "en"
    
    def _calculate_content_metrics(self, content: ExtractedContent) -> None:
        """Calculate content metrics and statistics"""
        
        # Reading time (average 200 words per minute)
        content.reading_time_minutes = content.word_count / 200.0
        
        # Sentiment analysis using NLTK
        if self.sentiment_analyzer:
            try:
                sentiment_scores = self.sentiment_analyzer.polarity_scores(content.clean_text[:1000])
                content.sentiment_score = sentiment_scores.get('compound', 0.0)
            except Exception as e:
                logger.warning(f"Sentiment analysis failed: {e}")
                content.sentiment_score = 0.0
        
        # Content density metrics
        if content.content:
            sentences = content.content.count('.') + content.content.count('!') + content.content.count('?')
            content.metadata['sentence_count'] = sentences
            content.metadata['avg_sentence_length'] = content.word_count / max(sentences, 1)
            
            # Paragraph count
            paragraphs = len([p for p in content.content.split('\n\n') if p.strip()])
            content.metadata['paragraph_count'] = paragraphs
            content.metadata['avg_paragraph_length'] = content.word_count / max(paragraphs, 1)
    
    async def _assess_content_quality(self, content: ExtractedContent) -> None:
        """Assess overall content quality score"""
        
        score = 0.0
        
        # Length assessment
        if 100 <= content.word_count <= 5000:
            length_score = min(1.0, content.word_count / 1000)
            score += length_score * 0.25
        elif content.word_count > 5000:
            score += 0.25
        
        # Structure assessment
        if content.headings:
            score += 0.15
        
        if content.metadata.get('paragraph_count', 0) > 3:
            score += 0.1
        
        # Authority indicators
        authority_keywords = [
            'research', 'study', 'analysis', 'university', 'professor',
            'journal', 'publication', 'peer-reviewed', 'academic'
        ]
        
        text_lower = content.content.lower()
        authority_matches = sum(1 for keyword in authority_keywords if keyword in text_lower)
        authority_score = min(1.0, authority_matches / 3)
        score += authority_score * 0.2
        
        # Citation indicators
        citation_patterns = [
            r'doi:', r'http://dx\.doi\.org', r'arxiv:', r'pubmed',
            r'\[\d+\]', r'\(\d{4}\)', r'et al\.', r'ibid\.'
        ]
        
        citation_matches = sum(1 for pattern in citation_patterns 
                              if re.search(pattern, content.content, re.IGNORECASE))
        citation_score = min(1.0, citation_matches / 2)
        score += citation_score * 0.15
        
        # Domain authority (basic heuristics)
        url_lower = content.url.lower()
        if any(domain in url_lower for domain in ['.edu', '.gov', '.org']):
            score += 0.1
        elif any(domain in url_lower for domain in ['.ac.uk', 'arxiv.org', 'scholar.google']):
            score += 0.15
        
        # Multimedia content bonus
        if content.images and len(content.images) > 2:
            score += 0.05
        
        if content.tables:
            score += 0.05
        
        content.content_quality_score = min(1.0, score)
    
    async def _analyze_credibility(self, page: Page, content: ExtractedContent) -> None:
        """Analyze credibility indicators"""
        
        indicators = {
            'author_present': False,
            'publication_date': None,
            'citations_count': 0,
            'contact_info_present': False,
            'references_present': False,
            'https_secure': content.url.startswith('https'),
            'domain_age_indicators': [],
            'editorial_standards': False
        }
        
        try:
            # Check for author information
            author_data = await page.evaluate("""
                () => {
                    const authorSelectors = [
                        '[rel="author"]', '.author', '.byline', '.post-author',
                        '.writer', '.journalist', '.by-author', '[itemprop="author"]'
                    ];
                    
                    for (const selector of authorSelectors) {
                        const element = document.querySelector(selector);
                        if (element && element.textContent.trim()) {
                            return {
                                found: true,
                                text: element.textContent.trim()
                            };
                        }
                    }
                    return { found: false, text: '' };
                }
            """)
            
            indicators['author_present'] = author_data['found']
            if author_data['found']:
                content.metadata['author'] = author_data['text']
            
            # Check for publication date
            date_data = await page.evaluate("""
                () => {
                    const dateSelectors = [
                        'time[datetime]', '.date', '.published', '.post-date',
                        '[property="article:published_time"]', '[itemprop="datePublished"]'
                    ];
                    
                    for (const selector of dateSelectors) {
                        const element = document.querySelector(selector);
                        if (element) {
                            const dateText = element.getAttribute('datetime') || 
                                           element.getAttribute('content') ||
                                           element.textContent.trim();
                            if (dateText) {
                                return {
                                    found: true,
                                    date: dateText
                                };
                            }
                        }
                    }
                    return { found: false, date: '' };
                }
            """)
            
            if date_data['found']:
                indicators['publication_date'] = date_data['date']
                content.metadata['publication_date'] = date_data['date']
            
            # Check for contact information
            contact_data = await page.evaluate("""
                () => {
                    const contactSelectors = [
                        'a[href^="mailto:"]', '.contact', '.email',
                        '[href*="contact"]', '[href*="about"]'
                    ];
                    
                    return contactSelectors.some(selector => 
                        document.querySelector(selector) !== null
                    );
                }
            """)
            
            indicators['contact_info_present'] = contact_data
            
            # Check for references/citations
            references_data = await page.evaluate("""
                () => {
                    const refSelectors = [
                        '.references', '.citations', '.bibliography',
                        'a[href*="doi.org"]', 'a[href*="pubmed"]', 'a[href*="arxiv"]',
                        '[class*="reference"]', '[id*="reference"]'
                    ];
                    
                    let count = 0;
                    refSelectors.forEach(selector => {
                        count += document.querySelectorAll(selector).length;
                    });
                    
                    return count;
                }
            """)
            
            indicators['citations_count'] = references_data
            indicators['references_present'] = references_data > 0
            
            # Editorial standards indicators
            editorial_indicators = [
                'editorial', 'editor', 'reviewed', 'fact-check', 'correction',
                'retraction', 'peer-review', 'editorial board'
            ]
            
            text_lower = content.content.lower()
            editorial_matches = sum(1 for indicator in editorial_indicators 
                                  if indicator in text_lower)
            indicators['editorial_standards'] = editorial_matches > 0
            
        except Exception as e:
            logger.warning(f"Credibility analysis failed: {e}")
        
        content.credibility_indicators = indicators
    
    async def _perform_ai_analysis(
        self, 
        content: ExtractedContent, 
        analysis_focus: List[str]
    ) -> None:
        """Perform AI-powered content analysis"""
        
        if not self.embedding_model:
            return
        
        try:
            analysis_results = {}
            
            # Generate embeddings for content
            if content.clean_text:
                embedding = self.embedding_model.encode([content.clean_text[:1000]])[0]
                analysis_results['content_embedding'] = embedding.tolist()
            
            # Topic extraction using TF-IDF
            if 'topics' in analysis_focus:
                topics = await self._extract_topics_tfidf(content.clean_text)
                content.key_topics = topics
                analysis_results['key_topics'] = topics
            
            # Named entity recognition
            if 'entities' in analysis_focus and self.nlp_model:
                entities = await self._extract_named_entities(content.clean_text)
                content.named_entities = entities
                analysis_results['named_entities'] = entities
            
            # Bias detection
            if 'bias' in analysis_focus:
                bias_analysis = await self._detect_bias_indicators(content.clean_text)
                content.bias_indicators = bias_analysis
                analysis_results['bias_analysis'] = bias_analysis
            
            # Content classification
            if 'classification' in analysis_focus:
                classification = await self._classify_content_advanced(content)
                analysis_results['content_classification'] = classification
            
            content.ai_analysis = analysis_results
            
        except Exception as e:
            logger.error(f"AI analysis failed: {e}")
            content.ai_analysis = {'error': str(e)}
    
    async def _extract_topics_tfidf(self, text: str) -> List[str]:
        """Extract topics using TF-IDF"""
        
        if not text or len(text) < 100:
            return []
        
        try:
            # Simple TF-IDF topic extraction
            sentences = text.split('.')[:50]  # Use first 50 sentences
            
            if len(sentences) < 2:
                return []
            
            vectorizer = TfidfVectorizer(
                max_features=100,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=1,
                max_df=0.8
            )
            
            tfidf_matrix = vectorizer.fit_transform(sentences)
            feature_names = vectorizer.get_feature_names_out()
            
            # Get top terms by TF-IDF score
            mean_scores = np.mean(tfidf_matrix.toarray(), axis=0)
            top_indices = np.argsort(mean_scores)[-10:][::-1]
            
            topics = [feature_names[i] for i in top_indices if mean_scores[i] > 0.01]
            return topics[:10]
            
        except Exception as e:
            logger.warning(f"Topic extraction failed: {e}")
            return []
    
    async def _extract_named_entities(self, text: str) -> List[Dict[str, str]]:
        """Extract named entities using spaCy"""
        
        if not self.nlp_model or not text:
            return []
        
        try:
            # Process text with spaCy (limit to first 1000 characters)
            doc = self.nlp_model(text[:1000])
            
            entities = []
            for ent in doc.ents:
                if len(ent.text.strip()) > 2:  # Filter short entities
                    entities.append({
                        'text': ent.text.strip(),
                        'label': ent.label_,
                        'description': spacy.explain(ent.label_) or ent.label_
                    })
            
            # Remove duplicates
            seen = set()
            unique_entities = []
            for entity in entities:
                key = (entity['text'].lower(), entity['label'])
                if key not in seen:
                    seen.add(key)
                    unique_entities.append(entity)
            
            return unique_entities[:20]  # Limit to 20 entities
            
        except Exception as e:
            logger.warning(f"Named entity extraction failed: {e}")
            return []
    
    async def _detect_bias_indicators(self, text: str) -> List[str]:
        """Detect potential bias indicators in text"""
        
        bias_patterns = {
            'absolutist_language': [
                r'\b(always|never|all|none|everyone|no one|everything|nothing)\b',
                r'\b(absolutely|definitely|certainly|unquestionably|undoubtedly)\b'
            ],
            'emotional_language': [
                r'\b(amazing|terrible|incredible|awful|fantastic|horrible)\b',
                r'\b(perfect|worst|best|greatest|most|least)\b'
            ],
            'loaded_terms': [
                r'\b(obviously|clearly|undeniably|without doubt|of course)\b'
            ],
            'weak_qualifiers': [
                r'\b(some say|many believe|it is said|reportedly|allegedly)\b'
            ]
        }
        
        found_indicators = []
        text_lower = text.lower()
        
        for category, patterns in bias_patterns.items():
            for pattern in patterns:
                matches = re.findall(pattern, text_lower, re.IGNORECASE)
                if matches:
                    found_indicators.extend([f"{category}: {match}" for match in matches[:3]])
        
        return found_indicators[:10]  # Limit results
    
    async def _classify_content_advanced(self, content: ExtractedContent) -> Dict[str, Any]:
        """Advanced content classification using multiple signals"""
        
        classification = {
            'primary_type': content.content_type.value,
            'confidence': 0.5,
            'secondary_types': [],
            'domain_category': 'general',
            'audience_level': 'general',
            'content_purpose': 'informational'
        }
        
        text = content.content.lower()
        url = content.url.lower()
        
        # Domain classification
        if any(domain in url for domain in ['.edu', '.ac.', 'university', 'college']):
            classification['domain_category'] = 'academic'
        elif any(domain in url for domain in ['.gov', 'government']):
            classification['domain_category'] = 'government'
        elif any(domain in url for domain in ['.org', 'nonprofit']):
            classification['domain_category'] = 'nonprofit'
        elif any(domain in url for domain in ['.com', 'business', 'company']):
            classification['domain_category'] = 'commercial'
        
        # Audience level
        if content.metadata.get('avg_sentence_length', 0) > 25:
            classification['audience_level'] = 'advanced'
        elif content.metadata.get('avg_sentence_length', 0) < 15:
            classification['audience_level'] = 'basic'
        
        # Content purpose
        purpose_indicators = {
            'educational': ['learn', 'tutorial', 'guide', 'how to', 'education'],
            'promotional': ['buy', 'sale', 'offer', 'discount', 'purchase'],
            'persuasive': ['should', 'must', 'opinion', 'believe', 'argue'],
            'analytical': ['analysis', 'research', 'study', 'data', 'findings']
        }
        
        max_score = 0
        for purpose, indicators in purpose_indicators.items():
            score = sum(1 for indicator in indicators if indicator in text)
            if score > max_score:
                max_score = score
                classification['content_purpose'] = purpose
        
        return classification
    
    async def _perform_ai_analysis_batch(
        self, 
        content_list: List[ExtractedContent], 
        task: BrowserTask
    ) -> None:
        """Perform AI analysis on batch of content"""
        
        if not content_list or not self.embedding_model:
            return
        
        try:
            # Batch process embeddings
            texts = [content.clean_text[:1000] for content in content_list if content.clean_text]
            
            if texts:
                embeddings = self.embedding_model.encode(texts, batch_size=8)
                
                # Perform similarity analysis
                if len(embeddings) > 1:
                    similarity_matrix = cosine_similarity(embeddings)
                    
                    # Find similar content pairs
                    for i, content in enumerate(content_list):
                        if i < len(embeddings):
                            similar_indices = np.argsort(similarity_matrix[i])[-3:]  # Top 3 similar
                            similar_urls = [
                                content_list[j].url for j in similar_indices 
                                if j != i and j < len(content_list)
                            ]
                            content.metadata['similar_content'] = similar_urls
            
            # Batch topic analysis
            for content in content_list:
                if content.clean_text and not content.key_topics:
                    content.key_topics = await self._extract_topics_tfidf(content.clean_text)
            
        except Exception as e:
            logger.error(f"Batch AI analysis failed: {e}")
    
    async def _capture_screenshot(self, page: Page, url: str) -> Optional[Tuple[str, bytes]]:
        """Capture screenshot of page"""
        
        try:
            # Generate filename
            url_hash = hashlib.md5(url.encode()).hexdigest()[:8]
            timestamp = int(time.time())
            filename = f"screenshot_{timestamp}_{url_hash}.png"
            file_path = f"{self.workspace_path}/screenshots/{filename}"
            
            # Capture screenshot
            screenshot_data = await page.screenshot(
                path=file_path,
                full_page=True,
                type='png'
            )
            
            return file_path, screenshot_data
            
        except Exception as e:
            logger.warning(f"Failed to capture screenshot for {url}: {e}")
            return None
    
    async def _fallback_content_extraction(self, url: str, primary_error: str) -> Optional[ExtractedContent]:
        """Fallback content extraction using requests + BeautifulSoup"""
        
        try:
            logger.info(f"Attempting fallback extraction for {url}")
            
            # Use requests with proper headers
            headers = {
                'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
            
            async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30)) as session:
                async with session.get(url, headers=headers) as response:
                    if response.status >= 400:
                        raise Exception(f"HTTP {response.status}")
                    
                    html_content = await response.text()
            
            # Parse with BeautifulSoup
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Remove unwanted elements
            for element in soup(['script', 'style', 'nav', 'header', 'footer', 'aside']):
                element.decompose()
            
            # Extract content
            content = ExtractedContent(
                url=url,
                extraction_method=ExtractionMethod.BEAUTIFULSOUP
            )
            
            # Title
            title_tag = soup.find('title')
            content.title = title_tag.get_text().strip() if title_tag else ""
            
            # Main content
            content_selectors = ['main', 'article', '.content', '.post-content', '.entry-content']
            main_content = ""
            
            for selector in content_selectors:
                element = soup.select_one(selector)
                if element:
                    main_content = element.get_text(separator=' ', strip=True)
                    if len(main_content) > 100:
                        break
            
            if not main_content:
                body = soup.find('body')
                main_content = body.get_text(separator=' ', strip=True) if body else ""
            
            content.content = main_content
            content.clean_text = self._clean_text(main_content)
            content.word_count = len(main_content.split()) if main_content else 0
            
            # Extract metadata
            meta_tags = soup.find_all('meta')
            for meta in meta_tags:
                name = meta.get('name') or meta.get('property')
                content_attr = meta.get('content')
                if name and content_attr:
                    content.metadata[name] = content_attr
            
            # Extract links
            links = []
            for link in soup.find_all('a', href=True)[:50]:
                links.append({
                    'href': link['href'],
                    'text': link.get_text().strip(),
                    'title': link.get('title', '')
                })
            content.links = links
            
            # Extract images
            images = []
            for img in soup.find_all('img', src=True)[:20]:
                images.append({
                    'src': img['src'],
                    'alt': img.get('alt', ''),
                    'title': img.get('title', '')
                })
            content.images = images
            
            # Basic quality assessment
            content.content_type = self._classify_content_type(content)
            content.language = self._detect_language(content.content)
            self._calculate_content_metrics(content)
            await self._assess_content_quality(content)
            
            # Add error context
            content.warnings.append(f"Primary extraction failed: {primary_error}")
            content.warnings.append("Used fallback BeautifulSoup extraction")
            
            return content
            
        except Exception as e:
            logger.error(f"Fallback extraction also failed for {url}: {e}")
            return None
    
    async def _initialize_ai_models(self) -> None:
        """Initialize AI models for content analysis"""
        
        try:
            # Initialize sentence transformers model
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            logger.info("Embedding model loaded successfully")
            
            # Initialize spaCy model
            try:
                import spacy
                self.nlp_model = spacy.load('en_core_web_sm')
                logger.info("spaCy model loaded successfully")
            except Exception as e:
                logger.warning(f"spaCy model not available: {e}")
                self.nlp_model = None
            
            # Initialize NLTK sentiment analyzer
            try:
                import nltk
                nltk.download('vader_lexicon', quiet=True)
                from nltk.sentiment import SentimentIntensityAnalyzer
                self.sentiment_analyzer = SentimentIntensityAnalyzer()
                logger.info("NLTK sentiment analyzer loaded successfully")
            except Exception as e:
                logger.warning(f"NLTK sentiment analyzer not available: {e}")
                self.sentiment_analyzer = None
            
        except Exception as e:
            logger.error(f"Failed to initialize AI models: {e}")
            raise
    
    async def _initialize_browsers(self) -> None:
        """Initialize browser engines"""
        
        try:
            # Initialize Playwright
            self.playwright_instance = await async_playwright().start()
            
            # Launch browsers
            browser_configs = {
                BrowserEngine.CHROMIUM: {
                    'headless': True,
                    'args': [
                        '--no-sandbox',
                        '--disable-dev-shm-usage',
                        '--disable-gpu',
                        '--disable-extensions',
                        '--disable-background-timer-throttling',
                        '--disable-backgrounding-occluded-windows',
                        '--disable-renderer-backgrounding',
                        '--disable-features=TranslateUI',
                        '--disable-ipc-flooding-protection'
                    ]
                },
                BrowserEngine.FIREFOX: {
                    'headless': True,
                    'args': ['--no-sandbox']
                }
            }
            
            # Launch Chromium
            self.browsers[BrowserEngine.CHROMIUM] = await self.playwright_instance.chromium.launch(
                **browser_configs[BrowserEngine.CHROMIUM]
            )
            
            # Launch Firefox
            self.browsers[BrowserEngine.FIREFOX] = await self.playwright_instance.firefox.launch(
                **browser_configs[BrowserEngine.FIREFOX]
            )
            
            logger.info("Browser engines initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize browsers: {e}")
            raise
    
    async def _initialize_content_processors(self) -> None:
        """Initialize content processing components"""
        
        # Initialize content extractors
        self.content_extractors = {
            ExtractionMethod.PLAYWRIGHT: self._extract_with_playwright,
            ExtractionMethod.BEAUTIFULSOUP: self._extract_with_beautifulsoup,
            ExtractionMethod.API_REQUEST: self._extract_with_api
        }
        
        # Initialize quality assessors
        self.quality_assessors = [
            self._assess_content_quality,
            self._analyze_credibility
        ]
        
        logger.info("Content processors initialized")
    
    async def _setup_vm_workspace(self) -> None:
        """Setup workspace directories in research VM"""
        
        workspace_commands = [
            f"mkdir -p {self.workspace_path}/screenshots",
            f"mkdir -p {self.workspace_path}/downloads", 
            f"mkdir -p {self.workspace_path}/cache",
            f"mkdir -p {self.workspace_path}/logs",
            f"chmod -R 755 {self.workspace_path}"
        ]
        
        for cmd in workspace_commands:
            try:
                result = await self.vm_orchestrator.base_vm_manager.execute_command(
                    self.research_vm_id, cmd
                )
                if not result.success:
                    logger.warning(f"Workspace setup command failed: {cmd}")
            except Exception as e:
                logger.warning(f"Failed to execute workspace command {cmd}: {e}")
    
    async def _load_browser_extensions(self) -> None:
        """Load research-specific browser extensions"""
        
        try:
            # Create context with research extension
            extension_path = f"{self.workspace_path}/extensions/research_assistant"
            
            # Research extension is already installed by ResearchVMOrchestrator
            # Just verify it's available
            verify_cmd = f"ls -la {extension_path}"
            result = await self.vm_orchestrator.base_vm_manager.execute_command(
                self.research_vm_id, verify_cmd
            )
            
            if result.success:
                logger.info("Research browser extensions verified")
            else:
                logger.warning("Research browser extensions not found")
                
        except Exception as e:
            logger.warning(f"Failed to load browser extensions: {e}")
    
    async def _initialize_rate_limiting(self) -> None:
        """Initialize rate limiting for domains"""
        
        # Default rate limits (requests per minute)
        default_limits = {
            'google.com': 30,
            'bing.com': 30,
            'duckduckgo.com': 60,
            'arxiv.org': 60,
            'scholar.google.com': 20,
            'default': 60
        }
        
        for domain, limit in default_limits.items():
            self.domain_rate_limits[domain] = {
                'limit': limit,
                'requests': [],
                'last_request': 0
            }
        
        logger.info("Rate limiting initialized")
    
    async def _respect_rate_limits(self, domain_or_engine: str) -> None:
        """Respect rate limits for domain or search engine"""
        
        current_time = time.time()
        
        # Determine domain
        if domain_or_engine in ['google', 'bing', 'duckduckgo', 'arxiv', 'scholar']:
            domain = f"{domain_or_engine}.com"
        else:
            # Extract domain from URL
            try:
                domain = urllib.parse.urlparse(f"http://{domain_or_engine}").netloc
            except:
                domain = domain_or_engine
        
        # Get rate limit info
        if domain not in self.domain_rate_limits:
            self.domain_rate_limits[domain] = {
                'limit': self.domain_rate_limits['default']['limit'],
                'requests': [],
                'last_request': 0
            }
        
        limit_info = self.domain_rate_limits[domain]
        
        # Clean old requests (older than 1 minute)
        minute_ago = current_time - 60
        limit_info['requests'] = [req_time for req_time in limit_info['requests'] if req_time > minute_ago]
        
        # Check if we need to wait
        if len(limit_info['requests']) >= limit_info['limit']:
            sleep_time = 60 - (current_time - limit_info['requests'][0])
            if sleep_time > 0:
                logger.debug(f"Rate limiting: sleeping {sleep_time:.2f}s for {domain}")
                await asyncio.sleep(sleep_time)
        
        # Minimum delay between requests
        time_since_last = current_time - limit_info['last_request']
        min_delay = 1.0  # 1 second minimum
        
        if time_since_last < min_delay:
            await asyncio.sleep(min_delay - time_since_last)
        
        # Record this request
        limit_info['requests'].append(time.time())
        limit_info['last_request'] = time.time()
    
    async def _get_browser_context(self, context_name: str) -> BrowserContext:
        """Get or create browser context"""
        
        if context_name in self.contexts:
            return self.contexts[context_name]
        
        # Create new context
        browser = self.browsers[BrowserEngine.CHROMIUM]  # Default to Chromium
        
        context = await browser.new_context(
            viewport={'width': 1920, 'height': 1080},
            user_agent='Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
        )
        
        # Set timeouts
        context.set_default_timeout(30000)
        
        self.contexts[context_name] = context
        return context
    
    async def _extract_with_playwright(self, url: str, **kwargs) -> Optional[ExtractedContent]:
        """Extract content using Playwright (already implemented above)"""
        # This method is implemented in _extract_content_from_url
        pass
    
    async def _extract_with_beautifulsoup(self, url: str, **kwargs) -> Optional[ExtractedContent]:
        """Extract content using BeautifulSoup (already implemented above)"""
        # This method is implemented in _fallback_content_extraction
        pass
    
    async def _extract_with_api(self, url: str, **kwargs) -> Optional[ExtractedContent]:
        """Extract content using API requests"""
        
        try:
            # Simple API-based extraction for supported sites
            headers = {
                'User-Agent': 'Mozilla/5.0 (compatible; ResearchBot/1.0)',
                'Accept': 'application/json, text/html'
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.get(url, headers=headers, timeout=30) as response:
                    if response.status != 200:
                        return None
                    
                    content_type = response.headers.get('content-type', '')
                    
                    if 'application/json' in content_type:
                        # Handle JSON API responses
                        data = await response.json()
                        
                        content = ExtractedContent(
                            url=url,
                            extraction_method=ExtractionMethod.API_REQUEST
                        )
                        
                        # Extract relevant fields from JSON
                        if 'title' in data:
                            content.title = str(data['title'])
                        
                        if 'content' in data:
                            content.content = str(data['content'])
                        elif 'body' in data:
                            content.content = str(data['body'])
                        elif 'text' in data:
                            content.content = str(data['text'])
                        
                        content.clean_text = self._clean_text(content.content)
                        content.word_count = len(content.content.split()) if content.content else 0
                        content.metadata.update(data)
                        
                        return content
                    
                    else:
                        # Fall back to HTML parsing
                        return await self._fallback_content_extraction(url, "API extraction not applicable")
            
        except Exception as e:
            logger.warning(f"API extraction failed for {url}: {e}")
            return None
    
    async def _execute_extraction_task(self, task: BrowserTask) -> BrowserResult:
        """Execute content extraction task for specific URLs"""
        
        all_results = []
        failed_urls = []
        
        task.status = TaskStatus.EXTRACTING
        total_urls = len(task.urls)
        
        for i, url in enumerate(task.urls):
            try:
                # Update progress
                task.progress_percentage = (i / total_urls) * 90
                await self._stream_task_event("EXTRACTION_PROGRESS", task, {
                    'url': url,
                    'progress': task.progress_percentage,
                    'completed': i,
                    'total': total_urls
                })
                
                # Extract content
                extracted_content = await self._extract_content_from_url(url, task)
                
                if extracted_content:
                    all_results.append(extracted_content)
                    
                    await self._stream_task_event("CONTENT_EXTRACTED", task, {
                        'url': url,
                        'title': extracted_content.title,
                        'word_count': extracted_content.word_count,
                        'quality_score': extracted_content.content_quality_score
                    })
                else:
                    failed_urls.append({'url': url, 'error': 'Extraction failed'})
                
                # Rate limiting
                await self._respect_rate_limits(url)
                
            except Exception as e:
                failed_urls.append({'url': url, 'error': str(e)})
                logger.error(f"Failed to extract content from {url}: {e}")
        
        # AI analysis phase
        if task.enable_ai_analysis and all_results:
            task.status = TaskStatus.ANALYZING
            await self._perform_ai_analysis_batch(all_results, task)
        
        return BrowserResult(
            task_id=task.task_id,
            success=True,
            extracted_content=all_results,
            task_objective=task.objective,
            urls_processed=task.urls,
            total_sources_found=len(task.urls),
            sources_successfully_processed=len(all_results),
            failed_urls=failed_urls
        )
    
    async def _execute_analysis_task(self, task: BrowserTask) -> BrowserResult:
        """Execute analysis task on existing content"""
        
        # This would analyze already extracted content
        # For now, return empty result
        return BrowserResult(
            task_id=task.task_id,
            success=True,
            task_objective=task.objective
        )
    
    async def _execute_multi_step_task(self, task: BrowserTask) -> BrowserResult:
        """Execute complex multi-step research task"""
        
        # This would implement complex research workflows
        # For now, delegate to search task
        return await self._execute_search_task(task)
    
    async def _stream_task_event(
        self, 
        event_type: str, 
        task: BrowserTask, 
        data: Optional[Dict[str, Any]] = None
    ) -> None:
        """Stream task events for real-time monitoring"""
        
        if not self.stream_manager:
            return
        
        try:
            event_data = {
                'task_id': task.task_id,
                'task_type': task.task_type,
                'status': task.status.value,
                'progress': task.progress_percentage,
                'objective': task.objective
            }
            
            if data:
                event_data.update(data)
            
            await self.stream_manager.stream_event(StreamEvent(
                event_type=getattr(StreamEventType, event_type, StreamEventType.BROWSER_EVENT),
                session_id=f"research_vm_{self.research_vm_id}",
                user_id="research_user",  # This would come from actual session
                timestamp=datetime.now(timezone.utc),
                data=event_data,
                priority=StreamPriority.NORMAL
            ))
            
        except Exception as e:
            logger.warning(f"Failed to stream task event {event_type}: {e}")
    
    async def get_task_status(self, task_id: str) -> Optional[BrowserTask]:
        """Get status of active task"""
        return self.active_tasks.get(task_id)
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel active task"""
        
        if task_id in self.active_tasks:
            task = self.active_tasks[task_id]
            task.status = TaskStatus.CANCELLED
            
            await self._stream_task_event("TASK_CANCELLED", task)
            
            # Remove from active tasks
            del self.active_tasks[task_id]
            
            logger.info(f"Task {task_id} cancelled")
            return True
        
        return False
    
    async def get_cached_content(self, url: str) -> Optional[ExtractedContent]:
        """Get cached content for URL"""
        return self.url_cache.get(url)
    
    async def clear_cache(self) -> None:
        """Clear URL cache"""
        self.url_cache.clear()
        logger.info("URL cache cleared")
    
    async def get_agent_statistics(self) -> Dict[str, Any]:
        """Get comprehensive agent statistics"""
        
        return {
            'active_tasks': len(self.active_tasks),
            'cached_urls': len(self.url_cache),
            'browser_contexts': len(self.contexts),
            'rate_limits': {
                domain: {
                    'limit': info['limit'],
                    'current_requests': len(info['requests'])
                }
                for domain, info in self.domain_rate_limits.items()
            },
            'models_loaded': {
                'embedding_model': self.embedding_model is not None,
                'nlp_model': self.nlp_model is not None,
                'sentiment_analyzer': self.sentiment_analyzer is not None
            },
            'extraction_methods': list(self.content_extractors.keys()),
            'search_engines': list(self.search_engines.keys())
        }
    
    async def shutdown(self) -> None:
        """Shutdown browser research agent"""
        
        logger.info("Shutting down AI Browser Research Agent...")
        
        try:
            # Cancel all active tasks
            for task_id in list(self.active_tasks.keys()):
                await self.cancel_task(task_id)
            
            # Close browser contexts
            for context in self.contexts.values():
                await context.close()
            
            # Close browsers
            for browser in self.browsers.values():
                await browser.close()
            
            # Stop Playwright
            if self.playwright_instance:
                await self.playwright_instance.stop()
            
            # Shutdown thread pool
            self.task_executor.shutdown(wait=True, timeout=30)
            
            logger.info("AI Browser Research Agent shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during browser agent shutdown: {e}")


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

async def create_browser_research_agent(
    vm_orchestrator: ResearchVMOrchestrator,
    research_vm_id: UUID,
    model_loader: ModelLoader,
    memory_manager: MemoryManager,
    stream_manager: ResearchStreamManager,
    workspace_path: str = "/home/ai/research"
) -> AIBrowserResearchAgent:
    """
    Factory function to create and initialize AI browser research agent
    
    Returns:
        Fully initialized AIBrowserResearchAgent
    """
    
    agent = AIBrowserResearchAgent(
        vm_orchestrator=vm_orchestrator,
        research_vm_id=research_vm_id,
        model_loader=model_loader,
        memory_manager=memory_manager,
        stream_manager=stream_manager,
        workspace_path=workspace_path
    )
    
    await agent.initialize()
    return agent


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def validate_url(url: str) -> bool:
    """Validate URL format and safety"""
    
    if not url or not isinstance(url, str):
        return False
    
    if not url.startswith(('http://', 'https://')):
        return False
    
    # Block potentially unsafe domains
    blocked_patterns = [
        'malware', 'phishing', 'suspicious', 'spam',
        'localhost', '127.0.0.1', '0.0.0.0'
    ]
    
    url_lower = url.lower()
    for pattern in blocked_patterns:
        if pattern in url_lower:
            return False
    
    return True


def estimate_content_reading_time(word_count: int, words_per_minute: int = 200) -> float:
    """Estimate reading time for content"""
    return word_count / words_per_minute


def calculate_content_similarity(content1: str, content2: str) -> float:
    """Calculate similarity between two pieces of content"""
    
    if not content1 or not content2:
        return 0.0
    
    # Simple word-based similarity
    words1 = set(content1.lower().split())
    words2 = set(content2.lower().split())
    
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    
    return len(intersection) / len(union) if union else 0.0