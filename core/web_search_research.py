"""
MORPHEUS CHAT - Live Web Search & Deep Research Engine
Sovereign web search with privacy protection and comprehensive research capabilities.

Architecture Features:
- Privacy-first search via DuckDuckGo and SearxNG
- Multi-source aggregation and source credibility scoring
- Automatic fact-checking and bias detection
- Deep research with recursive query expansion
- Content extraction and summarization
- Citation tracking and evidence weighting
"""

import asyncio
import logging
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from urllib.parse import urljoin, urlparse
from uuid import UUID, uuid4
from dataclasses import dataclass, field

import aiohttp
import asyncio
from bs4 import BeautifulSoup
import feedparser
from newspaper import Article
import requests_html
from readability import Document
import nltk
from textstat import flesch_reading_ease

from pydantic import BaseModel, Field, validator
from schemas.session import SessionID, UserID

logger = logging.getLogger(__name__)


class SearchProvider(str, Enum):
    """Available search providers with privacy focus"""
    DUCKDUCKGO = "duckduckgo"
    SEARXNG = "searxng"
    BRAVE = "brave"
    STARTPAGE = "startpage"


class SourceType(str, Enum):
    """Content source classification"""
    NEWS = "news"
    ACADEMIC = "academic"
    TECHNICAL = "technical"
    BLOG = "blog"
    FORUM = "forum"
    SOCIAL = "social"
    GOVERNMENT = "government"
    COMMERCIAL = "commercial"
    UNKNOWN = "unknown"


class CredibilityScore(str, Enum):
    """Source credibility assessment"""
    VERY_HIGH = "very_high"    # 0.9-1.0
    HIGH = "high"              # 0.7-0.9
    MEDIUM = "medium"          # 0.5-0.7
    LOW = "low"               # 0.3-0.5
    VERY_LOW = "very_low"     # 0.0-0.3


@dataclass
class SearchResult:
    """Individual search result with enhanced metadata"""
    url: str
    title: str
    snippet: str
    source_domain: str
    search_rank: int
    
    # Content analysis
    content: Optional[str] = None
    extracted_text: Optional[str] = None
    word_count: int = 0
    reading_level: Optional[float] = None
    
    # Source credibility
    source_type: SourceType = SourceType.UNKNOWN
    credibility_score: float = 0.5
    credibility_factors: List[str] = field(default_factory=list)
    
    # Temporal data
    publish_date: Optional[datetime] = None
    last_updated: Optional[datetime] = None
    age_days: Optional[int] = None
    
    # Search metadata
    search_query: str = ""
    search_provider: SearchProvider = SearchProvider.DUCKDUCKGO
    retrieved_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    # Analysis results
    key_points: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    bias_indicators: List[str] = field(default_factory=list)
    
    @property
    def is_recent(self, days_threshold: int = 30) -> bool:
        """Check if content is recent"""
        if not self.publish_date:
            return False
        age = datetime.now(timezone.utc) - self.publish_date
        return age.days <= days_threshold
    
    @property
    def credibility_level(self) -> CredibilityScore:
        """Get credibility level based on score"""
        if self.credibility_score >= 0.9:
            return CredibilityScore.VERY_HIGH
        elif self.credibility_score >= 0.7:
            return CredibilityScore.HIGH
        elif self.credibility_score >= 0.5:
            return CredibilityScore.MEDIUM
        elif self.credibility_score >= 0.3:
            return CredibilityScore.LOW
        else:
            return CredibilityScore.VERY_LOW


class ResearchQuery(BaseModel):
    """Structured research query with depth and scope"""
    query_id: UUID = Field(default_factory=uuid4)
    original_query: str = Field(description="User's original research question")
    user_id: UserID = Field(description="User conducting research")
    session_id: SessionID = Field(description="Session context")
    
    # Research configuration
    max_depth: int = Field(default=3, ge=1, le=5, description="Research depth levels")
    max_sources: int = Field(default=20, ge=5, le=100, description="Maximum sources per level")
    min_credibility: float = Field(default=0.3, ge=0, le=1, description="Minimum source credibility")
    
    # Content preferences
    source_types: List[SourceType] = Field(default_factory=lambda: [
        SourceType.NEWS, SourceType.ACADEMIC, SourceType.TECHNICAL
    ])
    max_age_days: Optional[int] = Field(None, description="Maximum content age")
    languages: List[str] = Field(default_factory=lambda: ["en"])
    
    # Research focus
    focus_areas: List[str] = Field(default_factory=list, description="Specific topics to focus on")
    exclude_terms: List[str] = Field(default_factory=list, description="Terms to avoid")
    
    # Output preferences
    include_citations: bool = Field(default=True)
    include_summaries: bool = Field(default=True)
    bias_analysis: bool = Field(default=True)
    fact_checking: bool = Field(default=True)
    
    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None


class ResearchReport(BaseModel):
    """Comprehensive research report with analysis and citations"""
    report_id: UUID = Field(default_factory=uuid4)
    query: ResearchQuery = Field(description="Original research query")
    
    # Executive summary
    executive_summary: str = Field(description="Key findings summary")
    key_findings: List[str] = Field(description="Main discoveries")
    confidence_level: float = Field(ge=0, le=1, description="Overall confidence in findings")
    
    # Research results
    total_sources: int = Field(ge=0, description="Total sources analyzed")
    credible_sources: int = Field(ge=0, description="High-credibility sources")
    search_results: List[SearchResult] = Field(description="All search results")
    
    # Analysis
    consensus_points: List[str] = Field(default_factory=list, description="Points with broad agreement")
    conflicting_information: List[str] = Field(default_factory=list, description="Contradictory findings")
    knowledge_gaps: List[str] = Field(default_factory=list, description="Areas needing more research")
    
    # Quality metrics
    source_diversity: float = Field(ge=0, le=1, description="Diversity of source types")
    temporal_coverage: float = Field(ge=0, le=1, description="Time range coverage")
    geographic_coverage: List[str] = Field(default_factory=list, description="Geographic regions covered")
    
    # Bias and fact-checking
    bias_assessment: Dict[str, float] = Field(default_factory=dict, description="Bias scores by type")
    fact_check_results: List[Dict[str, Any]] = Field(default_factory=list, description="Fact-checking results")
    
    # Citations and references
    citations: List[Dict[str, str]] = Field(default_factory=list, description="Formatted citations")
    reference_urls: List[str] = Field(default_factory=list, description="Source URLs")
    
    # Metadata
    research_duration: float = Field(ge=0, description="Research time in seconds")
    generated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class WebSearchEngine:
    """
    Privacy-first web search engine with multiple provider support.
    
    Implements sovereign search capabilities without tracking or API dependencies.
    """
    
    def __init__(
        self, 
        primary_provider: SearchProvider = SearchProvider.DUCKDUCKGO,
        timeout: int = 30,
        max_concurrent: int = 5
    ):
        self.primary_provider = primary_provider
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.max_concurrent = max_concurrent
        
        # Session with privacy headers
        self.session = None
        self.semaphore = asyncio.Semaphore(max_concurrent)
        
        # Provider configurations
        self.providers = {
            SearchProvider.DUCKDUCKGO: {
                "search_url": "https://duckduckgo.com/html/",
                "params": {"q": "{query}", "s": "0", "dc": "1"},
                "result_selector": ".result__body",
                "title_selector": ".result__title a",
                "snippet_selector": ".result__snippet",
                "url_selector": ".result__title a"
            },
            SearchProvider.SEARXNG: {
                "search_url": "https://searx.org/search",
                "params": {"q": "{query}", "format": "json", "safesearch": "0"},
                "json_results": True
            }
        }
        
        logger.info(f"Web search engine initialized with provider: {primary_provider}")
    
    async def __aenter__(self):
        """Async context manager entry"""
        connector = aiohttp.TCPConnector(
            limit=self.max_concurrent,
            limit_per_host=2,
            ttl_dns_cache=300,
            use_dns_cache=True
        )
        
        self.session = aiohttp.ClientSession(
            connector=connector,
            timeout=self.timeout,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Accept-Encoding': 'gzip, deflate',
                'DNT': '1',  # Do Not Track
                'Connection': 'keep-alive',
                'Upgrade-Insecure-Requests': '1'
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self.session:
            await self.session.close()
    
    async def search(
        self, 
        query: str, 
        max_results: int = 10,
        provider: Optional[SearchProvider] = None
    ) -> List[SearchResult]:
        """
        Perform web search with specified provider.
        
        Returns list of SearchResult objects with metadata.
        """
        provider = provider or self.primary_provider
        
        try:
            async with self.semaphore:
                if provider == SearchProvider.DUCKDUCKGO:
                    return await self._search_duckduckgo(query, max_results)
                elif provider == SearchProvider.SEARXNG:
                    return await self._search_searxng(query, max_results)
                else:
                    logger.warning(f"Provider {provider} not implemented, falling back to DuckDuckGo")
                    return await self._search_duckduckgo(query, max_results)
        
        except Exception as e:
            logger.error(f"Search failed for query '{query}': {e}")
            return []
    
    async def _search_duckduckgo(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using DuckDuckGo HTML interface"""
        config = self.providers[SearchProvider.DUCKDUCKGO]
        
        # Build search URL
        params = {k: v.format(query=query) if isinstance(v, str) else v 
                 for k, v in config["params"].items()}
        
        try:
            async with self.session.get(config["search_url"], params=params) as response:
                if response.status != 200:
                    logger.error(f"DuckDuckGo search failed: HTTP {response.status}")
                    return []
                
                html = await response.text()
                soup = BeautifulSoup(html, 'html.parser')
                
                results = []
                result_elements = soup.select(config["result_selector"])[:max_results]
                
                for idx, element in enumerate(result_elements):
                    try:
                        # Extract title
                        title_elem = element.select_one(config["title_selector"])
                        title = title_elem.get_text(strip=True) if title_elem else "No title"
                        
                        # Extract URL
                        url_elem = element.select_one(config["url_selector"])
                        url = url_elem.get('href', '') if url_elem else ''
                        
                        # Clean DuckDuckGo redirect URL
                        if url.startswith('/l/?uddg='):
                            # Extract actual URL from DuckDuckGo redirect
                            import urllib.parse
                            url = urllib.parse.unquote(url.split('uddg=')[1].split('&')[0])
                        
                        # Extract snippet
                        snippet_elem = element.select_one(config["snippet_selector"])
                        snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                        
                        if url and title:
                            # Parse domain
                            domain = urlparse(url).netloc
                            
                            result = SearchResult(
                                url=url,
                                title=title,
                                snippet=snippet,
                                source_domain=domain,
                                search_rank=idx + 1,
                                search_query=query,
                                search_provider=SearchProvider.DUCKDUCKGO
                            )
                            
                            results.append(result)
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse search result {idx}: {e}")
                        continue
                
                logger.info(f"DuckDuckGo search returned {len(results)} results for: {query}")
                return results
        
        except Exception as e:
            logger.error(f"DuckDuckGo search request failed: {e}")
            return []
    
    async def _search_searxng(self, query: str, max_results: int) -> List[SearchResult]:
        """Search using SearxNG instance"""
        config = self.providers[SearchProvider.SEARXNG]
        
        params = {k: v.format(query=query) if isinstance(v, str) else v 
                 for k, v in config["params"].items()}
        
        try:
            async with self.session.get(config["search_url"], params=params) as response:
                if response.status != 200:
                    logger.error(f"SearxNG search failed: HTTP {response.status}")
                    return []
                
                data = await response.json()
                results = []
                
                for idx, item in enumerate(data.get('results', [])[:max_results]):
                    try:
                        result = SearchResult(
                            url=item.get('url', ''),
                            title=item.get('title', 'No title'),
                            snippet=item.get('content', ''),
                            source_domain=urlparse(item.get('url', '')).netloc,
                            search_rank=idx + 1,
                            search_query=query,
                            search_provider=SearchProvider.SEARXNG
                        )
                        
                        results.append(result)
                    
                    except Exception as e:
                        logger.warning(f"Failed to parse SearxNG result {idx}: {e}")
                        continue
                
                logger.info(f"SearxNG search returned {len(results)} results for: {query}")
                return results
        
        except Exception as e:
            logger.error(f"SearxNG search request failed: {e}")
            return []
    
    async def multi_provider_search(
        self, 
        query: str, 
        providers: List[SearchProvider] = None,
        max_results_per_provider: int = 5
    ) -> List[SearchResult]:
        """
        Search across multiple providers and merge results.
        
        Deduplicates and ranks results by relevance and credibility.
        """
        providers = providers or [SearchProvider.DUCKDUCKGO, SearchProvider.SEARXNG]
        
        # Run searches concurrently
        search_tasks = [
            self.search(query, max_results_per_provider, provider)
            for provider in providers
        ]
        
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # Merge and deduplicate results
        all_results = []
        seen_urls = set()
        
        for provider_results in search_results:
            if isinstance(provider_results, Exception):
                logger.error(f"Provider search failed: {provider_results}")
                continue
            
            for result in provider_results:
                if result.url not in seen_urls:
                    seen_urls.add(result.url)
                    all_results.append(result)
        
        # Sort by search rank and provider priority
        all_results.sort(key=lambda r: (r.search_rank, providers.index(r.search_provider)))
        
        logger.info(f"Multi-provider search returned {len(all_results)} unique results")
        return all_results


class ContentAnalyzer:
    """
    Advanced content analysis for research quality assessment.
    
    Provides credibility scoring, bias detection, and fact-checking capabilities.
    """
    
    def __init__(self):
        # Domain credibility mapping
        self.domain_credibility = {
            # Very High Credibility
            'nature.com': 0.95, 'science.org': 0.95, 'nejm.org': 0.95,
            'who.int': 0.9, 'cdc.gov': 0.9, 'nih.gov': 0.9,
            'reuters.com': 0.9, 'ap.org': 0.9, 'bbc.com': 0.85,
            
            # High Credibility
            'nytimes.com': 0.8, 'washingtonpost.com': 0.8, 'theguardian.com': 0.8,
            'economist.com': 0.85, 'wsj.com': 0.8, 'npr.org': 0.8,
            'arxiv.org': 0.8, 'pubmed.ncbi.nlm.nih.gov': 0.9,
            
            # Medium Credibility
            'wikipedia.org': 0.7, 'cnn.com': 0.6, 'forbes.com': 0.6,
            'techcrunch.com': 0.6, 'medium.com': 0.5,
            
            # Lower Credibility
            'reddit.com': 0.4, 'quora.com': 0.4, 'yahoo.com': 0.4,
            'buzzfeed.com': 0.3, 'dailymail.co.uk': 0.3
        }
        
        # Bias indicators
        self.bias_keywords = {
            'political_left': ['progressive', 'liberal', 'socialist', 'leftist'],
            'political_right': ['conservative', 'republican', 'right-wing', 'traditional'],
            'sensational': ['shocking', 'unbelievable', 'devastating', 'explosive'],
            'conspiracy': ['cover-up', 'hidden truth', 'they don\'t want you to know'],
            'commercial': ['buy now', 'limited time', 'special offer', 'discount']
        }
        
        logger.info("Content analyzer initialized")
    
    async def analyze_content(self, result: SearchResult) -> SearchResult:
        """
        Comprehensive content analysis with credibility and bias assessment.
        
        Updates the SearchResult with analysis results.
        """
        try:
            # Extract full content if not already done
            if not result.content:
                result.content = await self._extract_content(result.url)
            
            # Basic content metrics
            if result.content:
                result.word_count = len(result.content.split())
                result.reading_level = flesch_reading_ease(result.content)
            
            # Source type classification
            result.source_type = self._classify_source_type(result.source_domain, result.url)
            
            # Credibility assessment
            result.credibility_score = self._assess_credibility(result)
            
            # Bias detection
            result.bias_indicators = self._detect_bias(result.content or result.snippet)
            
            # Entity extraction (simplified)
            result.entities = self._extract_entities(result.content or result.snippet)
            
            # Key points extraction
            result.key_points = self._extract_key_points(result.content or result.snippet)
            
            # Publication date detection
            result.publish_date = await self._extract_publish_date(result.url, result.content)
            if result.publish_date:
                result.age_days = (datetime.now(timezone.utc) - result.publish_date).days
            
            logger.debug(f"Content analysis completed for: {result.url}")
            return result
        
        except Exception as e:
            logger.error(f"Content analysis failed for {result.url}: {e}")
            return result
    
    async def _extract_content(self, url: str) -> Optional[str]:
        """Extract clean text content from URL"""
        try:
            # Use newspaper3k for content extraction
            article = Article(url)
            await asyncio.to_thread(article.download)
            await asyncio.to_thread(article.parse)
            
            return article.text
        
        except Exception as e:
            logger.warning(f"Content extraction failed for {url}: {e}")
            return None
    
    def _classify_source_type(self, domain: str, url: str) -> SourceType:
        """Classify source type based on domain and URL patterns"""
        domain = domain.lower()
        url = url.lower()
        
        # Government sources
        if domain.endswith('.gov') or domain.endswith('.mil'):
            return SourceType.GOVERNMENT
        
        # Academic sources
        if (domain.endswith('.edu') or 'arxiv' in domain or 'pubmed' in domain or
            'scholar.google' in domain or 'researchgate' in domain):
            return SourceType.ACADEMIC
        
        # News sources
        news_indicators = ['news', 'times', 'post', 'herald', 'guardian', 'bbc', 'cnn', 'reuters', 'ap.org']
        if any(indicator in domain for indicator in news_indicators):
            return SourceType.NEWS
        
        # Technical sources
        tech_indicators = ['stackoverflow', 'github', 'techcrunch', 'wired', 'ars-technica']
        if any(indicator in domain for indicator in tech_indicators):
            return SourceType.TECHNICAL
        
        # Social media and forums
        social_indicators = ['reddit', 'twitter', 'facebook', 'linkedin', 'quora']
        if any(indicator in domain for indicator in social_indicators):
            return SourceType.SOCIAL
        
        # Blogs
        if 'blog' in domain or 'medium.com' in domain or 'wordpress' in domain:
            return SourceType.BLOG
        
        # Commercial
        if '.com' in domain and 'shop' in url or 'buy' in url or 'store' in url:
            return SourceType.COMMERCIAL
        
        return SourceType.UNKNOWN
    
    def _assess_credibility(self, result: SearchResult) -> float:
        """Assess source credibility based on multiple factors"""
        base_score = self.domain_credibility.get(result.source_domain, 0.5)
        
        # Adjust based on source type
        type_adjustments = {
            SourceType.ACADEMIC: 0.1,
            SourceType.GOVERNMENT: 0.1,
            SourceType.NEWS: 0.05,
            SourceType.TECHNICAL: 0.0,
            SourceType.BLOG: -0.1,
            SourceType.SOCIAL: -0.2,
            SourceType.FORUM: -0.15,
            SourceType.COMMERCIAL: -0.1
        }
        
        adjustment = type_adjustments.get(result.source_type, 0)
        credibility = base_score + adjustment
        
        # Additional factors
        if result.content:
            # Length bonus (longer articles often more credible)
            if result.word_count > 1000:
                credibility += 0.05
            elif result.word_count < 200:
                credibility -= 0.1
            
            # Reading level (too simple or too complex might indicate issues)
            if result.reading_level and 30 <= result.reading_level <= 70:
                credibility += 0.05
        
        # Recency bonus for news
        if result.source_type == SourceType.NEWS and result.is_recent:
            credibility += 0.05
        
        # Bias penalty
        if result.bias_indicators:
            credibility -= 0.05 * len(result.bias_indicators)
        
        return max(0.0, min(1.0, credibility))
    
    def _detect_bias(self, content: str) -> List[str]:
        """Detect potential bias indicators in content"""
        if not content:
            return []
        
        content_lower = content.lower()
        detected_bias = []
        
        for bias_type, keywords in self.bias_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                detected_bias.append(bias_type)
        
        return detected_bias
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract named entities (simplified implementation)"""
        if not content:
            return []
        
        # Simple regex-based entity extraction
        # In production, would use spaCy or similar NLP library
        entities = []
        
        # Find capitalized words (potential proper nouns)
        import re
        capitalized_words = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', content)
        
        # Filter common words and take unique entities
        common_words = {'The', 'This', 'That', 'These', 'Those', 'A', 'An', 'In', 'On', 'At', 'By', 'For'}
        entities = list(set([word for word in capitalized_words 
                           if word not in common_words and len(word) > 2]))
        
        return entities[:10]  # Limit to top 10
    
    def _extract_key_points(self, content: str) -> List[str]:
        """Extract key points from content (simplified)"""
        if not content:
            return []
        
        # Split into sentences and find important ones
        sentences = content.split('. ')
        key_points = []
        
        # Look for sentences with key indicators
        importance_indicators = [
            'study shows', 'research indicates', 'according to', 'data reveals',
            'experts say', 'scientists found', 'results show', 'analysis suggests'
        ]
        
        for sentence in sentences[:20]:  # Limit search
            sentence_lower = sentence.lower()
            if (any(indicator in sentence_lower for indicator in importance_indicators) or
                len(sentence.split()) > 10):  # Substantial sentences
                key_points.append(sentence.strip())
        
        return key_points[:5]  # Top 5 key points
    
    async def _extract_publish_date(self, url: str, content: Optional[str]) -> Optional[datetime]:
        """Extract publication date from content or metadata"""
        try:
            # Use newspaper3k's date extraction
            article = Article(url)
            if content:
                article.set_text(content)
            else:
                await asyncio.to_thread(article.download)
                await asyncio.to_thread(article.parse)
            
            return article.publish_date
        
        except Exception:
            return None


class DeepResearchEngine:
    """
    Advanced research engine with recursive query expansion and multi-source analysis.
    
    Implements comprehensive research workflows with quality assessment and synthesis.
    """
    
    def __init__(self, search_engine: WebSearchEngine, content_analyzer: ContentAnalyzer):
        self.search_engine = search_engine
        self.content_analyzer = content_analyzer
        
        # Research configuration
        self.max_concurrent_analysis = 5
        self.analysis_semaphore = asyncio.Semaphore(self.max_concurrent_analysis)
        
        logger.info("Deep research engine initialized")
    
    async def conduct_research(self, query: ResearchQuery) -> ResearchReport:
        """
        Conduct comprehensive research with multiple levels and analysis.
        
        Returns detailed research report with findings and citations.
        """
        start_time = time.time()
        logger.info(f"Starting deep research for: {query.original_query}")
        
        try:
            # Level 1: Initial search
            initial_results = await self._level_1_search(query)
            
            # Level 2: Expand based on initial findings
            expanded_results = await self._level_2_expansion(query, initial_results)
            
            # Level 3: Deep dive into specific areas
            deep_results = await self._level_3_deep_dive(query, expanded_results)
            
            # Combine all results
            all_results = initial_results + expanded_results + deep_results
            
            # Analyze all content
            analyzed_results = await self._analyze_all_results(all_results)
            
            # Generate research report
            report = await self._generate_report(query, analyzed_results, start_time)
            
            logger.info(f"Research completed: {len(analyzed_results)} sources analyzed in {report.research_duration:.1f}s")
            return report
        
        except Exception as e:
            logger.error(f"Research failed for query '{query.original_query}': {e}")
            
            # Return error report
            return ResearchReport(
                query=query,
                executive_summary=f"Research failed due to error: {str(e)}",
                key_findings=["Research could not be completed"],
                confidence_level=0.0,
                total_sources=0,
                credible_sources=0,
                search_results=[],
                research_duration=time.time() - start_time
            )
    
    async def _level_1_search(self, query: ResearchQuery) -> List[SearchResult]:
        """Level 1: Direct search on main query"""
        results = await self.search_engine.multi_provider_search(
            query.original_query,
            max_results_per_provider=query.max_sources // 2
        )
        
        logger.info(f"Level 1 search: {len(results)} results")
        return results
    
    async def _level_2_expansion(self, query: ResearchQuery, initial_results: List[SearchResult]) -> List[SearchResult]:
        """Level 2: Expand search based on initial findings"""
        if query.max_depth < 2:
            return []
        
        # Extract key terms from initial results
        expansion_queries = self._generate_expansion_queries(query, initial_results)
        
        # Search for expanded queries
        expanded_results = []
        for exp_query in expansion_queries[:3]:  # Limit expansion
            results = await self.search_engine.search(exp_query, max_results=5)
            expanded_results.extend(results)
        
        logger.info(f"Level 2 expansion: {len(expanded_results)} results from {len(expansion_queries)} queries")
        return expanded_results
    
    async def _level_3_deep_dive(self, query: ResearchQuery, previous_results: List[SearchResult]) -> List[SearchResult]:
        """Level 3: Deep dive into specific high-value sources"""
        if query.max_depth < 3:
            return []
        
        # Identify high-credibility sources for deep analysis
        high_value_sources = [
            result for result in previous_results
            if result.credibility_score >= 0.7
        ][:5]  # Top 5 high-value sources
        
        # Extract related topics and search for them
        deep_results = []
        for source in high_value_sources:
            if source.entities:
                # Search for entities found in high-value sources
                entity_query = f"{query.original_query} {' '.join(source.entities[:3])}"
                results = await self.search_engine.search(entity_query, max_results=3)
                deep_results.extend(results)
        
        logger.info(f"Level 3 deep dive: {len(deep_results)} results")
        return deep_results
    
    def _generate_expansion_queries(self, query: ResearchQuery, results: List[SearchResult]) -> List[str]:
        """Generate expansion queries based on initial results"""
        expansion_queries = []
        
        # Extract common entities across results
        all_entities = []
        for result in results[:5]:  # Top 5 results
            all_entities.extend(result.entities)
        
        # Find most common entities
        from collections import Counter
        entity_counts = Counter(all_entities)
        top_entities = [entity for entity, count in entity_counts.most_common(5)]
        
        # Generate queries combining original with entities
        for entity in top_entities:
            expansion_queries.append(f"{query.original_query} {entity}")
        
        # Add focus area queries
        for focus in query.focus_areas:
            expansion_queries.append(f"{query.original_query} {focus}")
        
        return expansion_queries
    
    async def _analyze_all_results(self, results: List[SearchResult]) -> List[SearchResult]:
        """Analyze all search results concurrently"""
        async def analyze_with_semaphore(result):
            async with self.analysis_semaphore:
                return await self.content_analyzer.analyze_content(result)
        
        # Remove duplicates
        unique_results = []
        seen_urls = set()
        for result in results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
        
        # Analyze concurrently
        analysis_tasks = [analyze_with_semaphore(result) for result in unique_results]
        analyzed_results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Filter out failed analyses
        valid_results = [
            result for result in analyzed_results
            if not isinstance(result, Exception)
        ]
        
        logger.info(f"Content analysis completed: {len(valid_results)}/{len(unique_results)} successful")
        return valid_results
    
    async def _generate_report(
        self, 
        query: ResearchQuery, 
        results: List[SearchResult], 
        start_time: float
    ) -> ResearchReport:
        """Generate comprehensive research report"""
        
        # Filter by credibility threshold
        credible_results = [r for r in results if r.credibility_score >= query.min_credibility]
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(query, credible_results)
        
        # Extract key findings
        key_findings = self._extract_key_findings(credible_results)
        
        # Assess consensus and conflicts
        consensus_points, conflicts = self._analyze_consensus(credible_results)
        
        # Calculate metrics
        source_diversity = self._calculate_source_diversity(credible_results)
        confidence_level = self._calculate_confidence(credible_results)
        
        # Generate citations
        citations = self._generate_citations(credible_results)
        
        # Bias assessment
        bias_assessment = self._assess_overall_bias(credible_results)
        
        return ResearchReport(
            query=query,
            executive_summary=executive_summary,
            key_findings=key_findings,
            confidence_level=confidence_level,
            total_sources=len(results),
            credible_sources=len(credible_results),
            search_results=credible_results,
            consensus_points=consensus_points,
            conflicting_information=conflicts,
            source_diversity=source_diversity,
            bias_assessment=bias_assessment,
            citations=citations,
            reference_urls=[r.url for r in credible_results],
            research_duration=time.time() - start_time
        )
    
    def _generate_executive_summary(self, query: ResearchQuery, results: List[SearchResult]) -> str:
        """Generate executive summary of research findings"""
        if not results:
            return "No credible sources found for the research query."
        
        # Analyze source types
        source_types = [r.source_type.value for r in results]
        type_counts = {t: source_types.count(t) for t in set(source_types)}
        
        # Analyze credibility
        avg_credibility = sum(r.credibility_score for r in results) / len(results)
        high_cred_count = sum(1 for r in results if r.credibility_score >= 0.7)
        
        summary_parts = [
            f"Research on '{query.original_query}' analyzed {len(results)} credible sources.",
            f"Source breakdown: {', '.join([f'{k}: {v}' for k, v in type_counts.items()])}.",
            f"Average source credibility: {avg_credibility:.2f} ({high_cred_count} high-credibility sources)."
        ]
        
        # Add key insights if available
        if results[0].key_points:
            summary_parts.append(f"Key insight: {results[0].key_points[0]}")
        
        return " ".join(summary_parts)
    
    def _extract_key_findings(self, results: List[SearchResult]) -> List[str]:
        """Extract key findings from research results"""
        all_key_points = []
        for result in results[:10]:  # Top 10 sources
            all_key_points.extend(result.key_points)
        
        # Remove duplicates and return top findings
        unique_findings = list(set(all_key_points))
        return unique_findings[:8]  # Top 8 findings
    
    def _analyze_consensus(self, results: List[SearchResult]) -> Tuple[List[str], List[str]]:
        """Analyze consensus and conflicts in sources"""
        # Simplified implementation - would use NLP for real consensus analysis
        consensus_points = ["Analysis of source consensus requires advanced NLP implementation"]
        conflicts = ["Conflict detection requires semantic analysis of source content"]
        
        return consensus_points, conflicts
    
    def _calculate_source_diversity(self, results: List[SearchResult]) -> float:
        """Calculate diversity of source types"""
        if not results:
            return 0.0
        
        source_types = set(r.source_type for r in results)
        max_diversity = len(SourceType)
        
        return len(source_types) / max_diversity
    
    def _calculate_confidence(self, results: List[SearchResult]) -> float:
        """Calculate overall confidence in research findings"""
        if not results:
            return 0.0
        
        # Base confidence on average credibility and source count
        avg_credibility = sum(r.credibility_score for r in results) / len(results)
        source_count_factor = min(1.0, len(results) / 10)  # Confidence increases with more sources
        
        return (avg_credibility * 0.7) + (source_count_factor * 0.3)
    
    def _generate_citations(self, results: List[SearchResult]) -> List[Dict[str, str]]:
        """Generate formatted citations for sources"""
        citations = []
        
        for i, result in enumerate(results, 1):
            citation = {
                'number': str(i),
                'title': result.title,
                'url': result.url,
                'domain': result.source_domain,
                'type': result.source_type.value,
                'credibility': result.credibility_level.value
            }
            
            if result.publish_date:
                citation['date'] = result.publish_date.strftime('%Y-%m-%d')
            
            citations.append(citation)
        
        return citations
    
    def _assess_overall_bias(self, results: List[SearchResult]) -> Dict[str, float]:
        """Assess overall bias across all sources"""
        bias_counts = {}
        total_sources = len(results)
        
        if total_sources == 0:
            return {}
        
        # Count bias indicators across sources
        for result in results:
            for bias in result.bias_indicators:
                bias_counts[bias] = bias_counts.get(bias, 0) + 1
        
        # Convert to percentages
        bias_assessment = {
            bias: count / total_sources 
            for bias, count in bias_counts.items()
        }
        
        return bias_assessment


# Factory function for easy integration
async def create_research_system() -> Tuple[WebSearchEngine, DeepResearchEngine]:
    """
    Factory function to create complete research system.
    
    Returns configured search engine and research engine.
    """
    # Create and initialize search engine
    search_engine = WebSearchEngine()
    await search_engine.__aenter__()
    
    # Create content analyzer
    content_analyzer = ContentAnalyzer()
    
    # Create research engine
    research_engine = DeepResearchEngine(search_engine, content_analyzer)
    
    logger.info("Research system created successfully")
    return search_engine, research_engine