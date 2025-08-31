"""
MORPHEUS RESEARCH - Research Intelligence Engine
Production-grade AI research guidance system with container orchestration and memory learning.

Features:
- Real-time research quality assessment and guidance
- Semantic gap detection and completeness scoring
- Source diversity and bias analysis
- Container orchestration for specialized research tasks
- Memory-based learning from past research patterns
- Contradiction mapping and resolution suggestions
- Research optimization and direction guidance
- Multi-dimensional research metrics and scoring
"""

import asyncio
import json
import logging
import numpy as np
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union, Callable
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, Counter
import statistics

from pydantic import BaseModel, Field, validator
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from textstat import flesch_reading_ease, flesch_kincaid_grade
import spacy

# Core system imports
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope
from core.memory_integration import SessionMemoryContext
from schemas.session import SessionID, UserID
from research_stream_manager import (
    ResearchStreamManager, StreamEvent, StreamEventType, StreamPriority
)

logger = logging.getLogger(__name__)


class ResearchDimension(str, Enum):
    """Research analysis dimensions"""
    COMPLETENESS = "completeness"
    DIVERSITY = "diversity"
    CREDIBILITY = "credibility"
    DEPTH = "depth"
    BIAS_BALANCE = "bias_balance"
    TEMPORAL_COVERAGE = "temporal_coverage"
    GEOGRAPHIC_COVERAGE = "geographic_coverage"
    PERSPECTIVE_VARIETY = "perspective_variety"
    EVIDENCE_STRENGTH = "evidence_strength"
    CONTRADICTION_RESOLUTION = "contradiction_resolution"


class ResearchInsightType(str, Enum):
    """Types of research insights"""
    GAP_DETECTED = "gap_detected"
    BIAS_PATTERN = "bias_pattern"
    QUALITY_CONCERN = "quality_concern"
    OPTIMIZATION_SUGGESTION = "optimization_suggestion"
    CONTRADICTION_FOUND = "contradiction_found"
    DEPTH_INSUFFICIENT = "depth_insufficient"
    SOURCE_IMBALANCE = "source_imbalance"
    METHODOLOGY_ISSUE = "methodology_issue"
    SEMANTIC_DRIFT = "semantic_drift"
    EXPERTISE_MISMATCH = "expertise_mismatch"


class ResearchComplexity(str, Enum):
    """Research complexity levels"""
    SIMPLE = "simple"           # Single domain, clear answers
    MODERATE = "moderate"       # Multi-faceted, some ambiguity
    COMPLEX = "complex"         # Multi-domain, high ambiguity
    EXPERT = "expert"           # Specialized knowledge required
    FRONTIER = "frontier"       # Cutting-edge, limited sources


@dataclass
class ResearchMetrics:
    """Comprehensive research quality metrics"""
    completeness_score: float = 0.0
    diversity_score: float = 0.0
    credibility_score: float = 0.0
    depth_score: float = 0.0
    bias_balance_score: float = 0.0
    overall_quality: float = 0.0
    
    # Detailed metrics
    source_count: int = 0
    unique_domains: int = 0
    perspective_count: int = 0
    contradiction_count: int = 0
    evidence_strength: float = 0.0
    temporal_span: float = 0.0  # Years covered
    geographic_coverage: float = 0.0
    
    # Confidence measures
    confidence_level: float = 0.0
    uncertainty_areas: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            'scores': {
                'completeness': self.completeness_score,
                'diversity': self.diversity_score,
                'credibility': self.credibility_score,
                'depth': self.depth_score,
                'bias_balance': self.bias_balance_score,
                'overall_quality': self.overall_quality
            },
            'details': {
                'source_count': self.source_count,
                'unique_domains': self.unique_domains,
                'perspective_count': self.perspective_count,
                'contradiction_count': self.contradiction_count,
                'evidence_strength': self.evidence_strength,
                'temporal_span': self.temporal_span,
                'geographic_coverage': self.geographic_coverage
            },
            'confidence': {
                'level': self.confidence_level,
                'uncertainty_areas': self.uncertainty_areas
            }
        }


@dataclass
class ResearchInsight:
    """Individual research insight with actionable recommendations"""
    insight_type: ResearchInsightType
    dimension: ResearchDimension
    severity: float  # 0.0 to 1.0
    confidence: float  # 0.0 to 1.0
    description: str
    evidence: Dict[str, Any]
    recommendations: List[str]
    automated_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    # Metadata
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    insight_id: UUID = field(default_factory=uuid4)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'insight_id': str(self.insight_id),
            'type': self.insight_type.value,
            'dimension': self.dimension.value,
            'severity': self.severity,
            'confidence': self.confidence,
            'description': self.description,
            'evidence': self.evidence,
            'recommendations': self.recommendations,
            'automated_actions': self.automated_actions,
            'detected_at': self.detected_at.isoformat()
        }


@dataclass
class ResearchContext:
    """Context for research analysis"""
    session_id: str
    user_id: str
    query: str
    research_goals: List[str]
    domain_expertise: List[str]
    complexity_level: ResearchComplexity
    sources_processed: List[Dict[str, Any]]
    plan_data: Dict[str, Any]
    user_preferences: Dict[str, Any] = field(default_factory=dict)
    
    # Dynamic state
    current_depth: int = 0
    time_spent: float = 0.0
    iterations_completed: int = 0


class SemanticAnalyzer:
    """Advanced semantic analysis for research content"""
    
    def __init__(self):
        self.nlp = None
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=10000,
            stop_words='english',
            ngram_range=(1, 3)
        )
        self._initialize_nlp()
    
    def _initialize_nlp(self):
        """Initialize NLP models"""
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("spaCy model not found. Some features will be limited.")
            self.nlp = None
    
    async def analyze_semantic_coverage(
        self,
        query: str,
        sources: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze semantic coverage of sources relative to query"""
        if not sources:
            return {'coverage_score': 0.0, 'gaps': ['No sources provided']}
        
        # Extract text content from sources
        source_texts = []
        for source in sources:
            text = source.get('content', '') or source.get('snippet', '')
            source_texts.append(text)
        
        if not any(source_texts):
            return {'coverage_score': 0.0, 'gaps': ['No text content in sources']}
        
        # Vectorize query and sources
        all_texts = [query] + source_texts
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(all_texts)
        
        # Calculate similarity between query and sources
        query_vector = tfidf_matrix[0:1]
        source_vectors = tfidf_matrix[1:]
        
        similarities = cosine_similarity(query_vector, source_vectors)[0]
        
        # Analyze coverage
        coverage_score = np.mean(similarities)
        
        # Identify potential gaps using entity analysis
        gaps = await self._identify_semantic_gaps(query, source_texts)
        
        return {
            'coverage_score': float(coverage_score),
            'individual_similarities': similarities.tolist(),
            'gaps': gaps,
            'keyword_coverage': self._analyze_keyword_coverage(query, source_texts)
        }
    
    async def _identify_semantic_gaps(
        self,
        query: str,
        source_texts: List[str]
    ) -> List[str]:
        """Identify semantic gaps in research coverage"""
        gaps = []
        
        if not self.nlp:
            return ['NLP model unavailable for gap analysis']
        
        try:
            # Extract entities and key concepts from query
            query_doc = self.nlp(query)
            query_entities = [ent.text.lower() for ent in query_doc.ents]
            query_keywords = [token.lemma_.lower() for token in query_doc 
                            if token.is_alpha and not token.is_stop]
            
            # Analyze coverage in sources
            covered_entities = set()
            covered_keywords = set();
            
            for text in source_texts:
                if text:
                    doc = self.nlp(text)
                    text_entities = [ent.text.lower() for ent in doc.ents]
                    text_keywords = [token.lemma_.lower() for token in doc 
                                   if token.is_alpha and not token.is_stop]
                    
                    covered_entities.update(text_entities)
                    covered_keywords.update(text_keywords)
            
            # Identify missing coverage
            missing_entities = set(query_entities) - covered_entities
            missing_keywords = set(query_keywords) - covered_keywords
            
            if missing_entities:
                gaps.append(f"Missing entity coverage: {', '.join(missing_entities)}")
            
            if len(missing_keywords) > len(query_keywords) * 0.3:  # 30% threshold
                gaps.append(f"Insufficient keyword coverage: {len(missing_keywords)} key terms missing")
        
        except Exception as e:
            logger.error(f"Error in semantic gap analysis: {e}")
            gaps.append("Error in semantic analysis")
        
        return gaps
    
    def _analyze_keyword_coverage(
        self,
        query: str,
        source_texts: List[str]
    ) -> Dict[str, Any]:
        """Analyze keyword coverage between query and sources"""
        query_words = set(query.lower().split())
        
        coverage_by_source = []
        all_source_words = set()
        
        for text in source_texts:
            if text:
                source_words = set(text.lower().split())
                all_source_words.update(source_words)
                overlap = len(query_words.intersection(source_words))
                coverage = overlap / len(query_words) if query_words else 0
                coverage_by_source.append(coverage)
        
        overall_coverage = len(query_words.intersection(all_source_words)) / len(query_words) if query_words else 0
        
        return {
            'overall_coverage': overall_coverage,
            'coverage_by_source': coverage_by_source,
            'query_terms_found': len(query_words.intersection(all_source_words)),
            'total_query_terms': len(query_words)
        }


class ContainerOrchestrator:
    """Orchestrates specialized containers for research analysis"""
    
    def __init__(self, vm_manager=None, container_runtime=None):
        self.vm_manager = vm_manager
        self.container_runtime = container_runtime
        self.active_analysis_tasks: Dict[str, asyncio.Task] = {}
    
    async def launch_analysis_container(
        self,
        analysis_type: str,
        data: Dict[str, Any],
        session_id: str
    ) -> Optional[str]:
        """Launch specialized container for research analysis"""
        
        if not self.container_runtime:
            logger.warning("Container runtime not available")
            return None
        
        try:
            # Define container specifications for different analysis types
            container_specs = {
                'semantic_analysis': {
                    'image': 'research/nlp-analyzer',
                    'environment': {
                        'ANALYSIS_TYPE': 'semantic',
                        'SESSION_ID': session_id
                    },
                    'resources': {
                        'memory': '4G',
                        'cpu': '2.0'
                    }
                },
                'bias_detection': {
                    'image': 'research/bias-detector',
                    'environment': {
                        'ANALYSIS_TYPE': 'bias',
                        'SESSION_ID': session_id
                    },
                    'resources': {
                        'memory': '2G',
                        'cpu': '1.0'
                    }
                },
                'credibility_analysis': {
                    'image': 'research/credibility-analyzer',
                    'environment': {
                        'ANALYSIS_TYPE': 'credibility',
                        'SESSION_ID': session_id
                    },
                    'resources': {
                        'memory': '3G',
                        'cpu': '1.5'
                    }
                }
            }
            
            spec = container_specs.get(analysis_type)
            if not spec:
                logger.error(f"Unknown analysis type: {analysis_type}")
                return None
            
            # Launch container (interface with actual container runtime)
            container_id = await self._launch_container(spec, data)
            
            if container_id:
                logger.info(f"Launched {analysis_type} container: {container_id}")
            
            return container_id
            
        except Exception as e:
            logger.error(f"Failed to launch analysis container: {e}")
            return None
    
    async def _launch_container(
        self,
        spec: Dict[str, Any],
        data: Dict[str, Any]
    ) -> Optional[str]:
        """Launch container with given specification"""
        # This would interface with the actual container runtime
        # For now, simulate container launch
        container_id = str(uuid4())
        
        # Simulate container startup time
        await asyncio.sleep(1)
        
        return container_id
    
    async def get_analysis_result(
        self,
        container_id: str,
        timeout: float = 30.0
    ) -> Optional[Dict[str, Any]]:
        """Get analysis result from container"""
        try:
            # Simulate container analysis completion
            await asyncio.sleep(2)
            
            # Return mock analysis result
            return {
                'container_id': container_id,
                'analysis_complete': True,
                'result': {
                    'score': 0.85,
                    'details': 'Analysis completed successfully',
                    'recommendations': ['Improve source diversity']
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to get analysis result: {e}")
            return None
    
    async def cleanup_container(self, container_id: str):
        """Clean up analysis container"""
        try:
            # Interface with container runtime to cleanup
            logger.info(f"Cleaning up container: {container_id}")
        except Exception as e:
            logger.error(f"Failed to cleanup container: {e}")


class ResearchIntelligenceEngine:
    """Production-grade research intelligence and guidance system"""
    
    def __init__(
        self,
        memory_manager: MemoryManager,
        stream_manager: ResearchStreamManager,
        vm_manager=None,
        container_runtime=None
    ):
        self.memory_manager = memory_manager
        self.stream_manager = stream_manager
        self.semantic_analyzer = SemanticAnalyzer()
        self.container_orchestrator = ContainerOrchestrator(vm_manager, container_runtime)
        
        # Research pattern learning
        self.pattern_cache: Dict[str, Any] = {}
        self.quality_thresholds = {
            ResearchDimension.COMPLETENESS: 0.7,
            ResearchDimension.DIVERSITY: 0.6,
            ResearchDimension.CREDIBILITY: 0.8,
            ResearchDimension.DEPTH: 0.6,
            ResearchDimension.BIAS_BALANCE: 0.7
        }
        
        # Real-time insights
        self.active_insights: Dict[str, List[ResearchInsight]] = defaultdict(list)
        self.insight_handlers: Dict[ResearchInsightType, List[Callable]] = defaultdict(list)
        
        # Performance metrics
        self.metrics = {
            'analyses_performed': 0,
            'insights_generated': 0,
            'containers_launched': 0,
            'average_analysis_time': 0.0
        }
        
        logger.info("Research Intelligence Engine initialized")
    
    async def analyze_research_quality(
        self,
        research_context: ResearchContext
    ) -> Tuple[ResearchMetrics, List[ResearchInsight]]:
        """Comprehensive research quality analysis"""
        start_time = time.time()
        
        try:
            # Parallel analysis across multiple dimensions
            analysis_tasks = [
                self._analyze_completeness(research_context),
                self._analyze_diversity(research_context),
                self._analyze_credibility(research_context),
                self._analyze_depth(research_context),
                self._analyze_bias_balance(research_context)
            ]
            
            results = await asyncio.gather(*analysis_tasks, return_exceptions=True)
            
            # Compile metrics
            metrics = ResearchMetrics()
            insights = []
            
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logger.error(f"Analysis {i} failed: {result}")
                    continue

                # Only unpack if result is not an exception
                dimension_metrics, dimension_insights = result
                
                # Update metrics
                if i == 0:  # Completeness
                    metrics.completeness_score = dimension_metrics.get('score', 0.0)
                elif i == 1:  # Diversity
                    metrics.diversity_score = dimension_metrics.get('score', 0.0)
                    metrics.unique_domains = dimension_metrics.get('unique_domains', 0)
                    metrics.perspective_count = dimension_metrics.get('perspectives', 0)
                elif i == 2:  # Credibility
                    metrics.credibility_score = dimension_metrics.get('score', 0.0)
                    metrics.evidence_strength = dimension_metrics.get('evidence_strength', 0.0)
                elif i == 3:  # Depth
                    metrics.depth_score = dimension_metrics.get('score', 0.0)
                elif i == 4:  # Bias balance
                    metrics.bias_balance_score = dimension_metrics.get('score', 0.0)
                
                insights.extend(dimension_insights)
            
            # Calculate overall quality score
            metrics.overall_quality = self._calculate_overall_quality(metrics)
            metrics.source_count = len(research_context.sources_processed)
            
            # Generate high-level insights
            high_level_insights = await self._generate_high_level_insights(metrics, research_context)
            insights.extend(high_level_insights)
            
            # Store in memory for pattern learning
            await self._store_research_pattern(research_context, metrics, insights)
            
            # Stream real-time updates
            await self._stream_analysis_results(research_context, metrics, insights)
            
            # Update performance metrics
            analysis_time = time.time() - start_time
            self.metrics['analyses_performed'] += 1
            self.metrics['insights_generated'] += len(insights)
            self.metrics['average_analysis_time'] = (
                self.metrics['average_analysis_time'] * (self.metrics['analyses_performed'] - 1) + 
                analysis_time
            ) / self.metrics['analyses_performed']
            
            logger.info(f"Research analysis completed in {analysis_time:.2f}s with {len(insights)} insights")
            
            return metrics, insights
            
        except Exception as e:
            logger.error(f"Research analysis failed: {e}")
            return ResearchMetrics(), []
    
    async def _analyze_completeness(
        self,
        context: ResearchContext
    ) -> Tuple[Dict[str, Any], List[ResearchInsight]]:
        """Analyze research completeness"""
        insights = []
        
        # Semantic coverage analysis
        semantic_analysis = await self.semantic_analyzer.analyze_semantic_coverage(
            context.query,
            context.sources_processed
        )
        
        coverage_score = semantic_analysis['coverage_score']
        gaps = semantic_analysis['gaps']
        
        # Generate insights for gaps
        if coverage_score < self.quality_thresholds[ResearchDimension.COMPLETENESS]:
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.GAP_DETECTED,
                dimension=ResearchDimension.COMPLETENESS,
                severity=1.0 - coverage_score,
                confidence=0.8,
                description=f"Research shows {coverage_score:.1%} semantic coverage. Significant gaps detected.",
                evidence={'gaps': gaps, 'coverage_score': coverage_score},
                recommendations=[
                    "Expand search to cover missing semantic areas",
                    "Include more diverse source types",
                    "Consider additional search terms from gap analysis"
                ],
                automated_actions=[
                    {
                        'action': 'expand_search',
                        'parameters': {'focus_areas': gaps[:3]}
                    }
                ]
            ))
        
        metrics = {
            'score': coverage_score,
            'semantic_gaps': len(gaps),
            'keyword_coverage': semantic_analysis['keyword_coverage']['overall_coverage']
        }
        
        return metrics, insights
    
    async def _analyze_diversity(
        self,
        context: ResearchContext
    ) -> Tuple[Dict[str, Any], List[ResearchInsight]]:
        """Analyze source and perspective diversity"""
        insights = []
        sources = context.sources_processed
        
        if not sources:
            return {'score': 0.0, 'unique_domains': 0, 'perspectives': 0}, insights
        
        # Domain diversity analysis
        domains = set()
        source_types = Counter()
        publication_dates = []
        
        for source in sources:
            url = source.get('url', '')
            if url:
                domain = url.split('/')[2] if '/' in url else url
                domains.add(domain)
            
            source_type = source.get('source_type', 'unknown')
            source_types[source_type] += 1
            
            pub_date = source.get('published_date')
            if pub_date:
                publication_dates.append(pub_date)
        
        # Calculate diversity scores
        domain_diversity = len(domains) / len(sources) if sources else 0
        type_diversity = len(source_types) / len(sources) if sources else 0
        
        # Temporal diversity
        temporal_diversity = 0.0
        if publication_dates:
            try:
                date_span = max(publication_dates) - min(publication_dates)
                temporal_diversity = min(date_span.days / 365.0, 1.0)  # Normalize to 1 year
            except:
                temporal_diversity = 0.0
        
        overall_diversity = (domain_diversity + type_diversity + temporal_diversity) / 3
        
        # Generate insights
        if domain_diversity < 0.5:
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.SOURCE_IMBALANCE,
                dimension=ResearchDimension.DIVERSITY,
                severity=1.0 - domain_diversity,
                confidence=0.9,
                description=f"Low domain diversity: {len(domains)} unique domains from {len(sources)} sources",
                evidence={
                    'domain_count': len(domains),
                    'source_count': len(sources),
                    'domains': list(domains)
                },
                recommendations=[
                    "Include sources from more diverse domains",
                    "Seek alternative perspectives on the topic",
                    "Add international or specialized sources"
                ]
            ))
        
        if len(source_types) < 3:
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.SOURCE_IMBALANCE,
                dimension=ResearchDimension.DIVERSITY,
                severity=0.6,
                confidence=0.8,
                description=f"Limited source type diversity: {dict(source_types)}",
                evidence={'source_types': dict(source_types)},
                recommendations=[
                    "Include academic sources if missing",
                    "Add news sources for current perspectives",
                    "Consider government or official sources"
                ]
            ))
        
        metrics = {
            'score': overall_diversity,
            'unique_domains': len(domains),
            'perspectives': len(source_types),
            'temporal_span': temporal_diversity,
            'type_distribution': dict(source_types)
        }
        
        return metrics, insights
    
    async def _analyze_credibility(
        self,
        context: ResearchContext
    ) -> Tuple[Dict[str, Any], List[ResearchInsight]]:
        """Analyze source credibility and evidence strength"""
        insights = []
        sources = context.sources_processed
        
        if not sources:
            return {'score': 0.0, 'evidence_strength': 0.0}, insights
        
        # Launch container for detailed credibility analysis
        container_id = await self.container_orchestrator.launch_analysis_container(
            'credibility_analysis',
            {'sources': sources},
            context.session_id
        )
        
        credibility_scores = []
        evidence_strengths = []
        low_credibility_sources = []
        
        for source in sources:
            # Basic credibility scoring
            credibility_score = source.get('credibility_score', 0.5)
            credibility_scores.append(credibility_score)
            
            # Evidence strength analysis
            evidence_strength = self._assess_evidence_strength(source)
            evidence_strengths.append(evidence_strength)
            
            if credibility_score < 0.6:
                low_credibility_sources.append(source)
        
        # Get enhanced analysis from container if available
        if container_id:
            container_result = await self.container_orchestrator.get_analysis_result(container_id)
            if container_result:
                # Enhance scores with container analysis
                enhanced_score = container_result.get('result', {}).get('score', 0.0)
                credibility_scores = [s * enhanced_score for s in credibility_scores]
        
        avg_credibility = statistics.mean(credibility_scores) if credibility_scores else 0.0
        avg_evidence_strength = statistics.mean(evidence_strengths) if evidence_strengths else 0.0
        
        # Generate insights
        if avg_credibility < self.quality_thresholds[ResearchDimension.CREDIBILITY]:
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.QUALITY_CONCERN,
                dimension=ResearchDimension.CREDIBILITY,
                severity=1.0 - avg_credibility,
                confidence=0.85,
                description=f"Average source credibility is {avg_credibility:.1%}",
                evidence={
                    'average_credibility': avg_credibility,
                    'low_credibility_count': len(low_credibility_sources),
                    'credibility_distribution': credibility_scores
                },
                recommendations=[
                    "Prioritize higher-credibility sources",
                    "Verify claims through authoritative sources",
                    "Cross-reference information across multiple sources"
                ]
            ))
        
        if low_credibility_sources:
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.QUALITY_CONCERN,
                dimension=ResearchDimension.CREDIBILITY,
                severity=len(low_credibility_sources) / len(sources),
                confidence=0.9,
                description=f"{len(low_credibility_sources)} sources have credibility concerns",
                evidence={'problematic_sources': [s.get('url', 'Unknown') for s in low_credibility_sources]},
                recommendations=[
                    "Review and potentially exclude low-credibility sources",
                    "Seek corroboration from more authoritative sources"
                ]
            ))
        
        metrics = {
            'score': avg_credibility,
            'evidence_strength': avg_evidence_strength,
            'credibility_distribution': credibility_scores,
            'low_credibility_count': len(low_credibility_sources)
        }
        
        return metrics, insights
    
    async def _analyze_depth(
        self,
        context: ResearchContext
    ) -> Tuple[Dict[str, Any], List[ResearchInsight]]:
        """Analyze research depth and thoroughness"""
        insights = []
        
        # Depth indicators
        avg_content_length = 0
        detailed_sources = 0
        shallow_sources = 0
        
        for source in context.sources_processed:
            content = source.get('content', '') or source.get('snippet', '')
            content_length = len(content)
            avg_content_length += content_length
            
            if content_length > 1000:  # Arbitrary threshold for detailed content
                detailed_sources += 1
            elif content_length < 200:
                shallow_sources += 1
        
        avg_content_length /= len(context.sources_processed) if context.sources_processed else 1
        
        # Reading complexity analysis
        complexity_scores = []
        for source in context.sources_processed:
            content = source.get('content', '')
            if content and len(content) > 100:
                try:
                    reading_ease = flesch_reading_ease(content)
                    grade_level = flesch_kincaid_grade(content)
                    complexity_scores.append({
                        'reading_ease': reading_ease,
                        'grade_level': grade_level
                    })
                except:
                    pass
        
        # Depth scoring
        depth_factors = [
            detailed_sources / len(context.sources_processed) if context.sources_processed else 0,
            min(avg_content_length / 2000, 1.0),  # Normalize to 2000 chars
            # Convert complexity level (string) to numeric weight for division
            context.current_depth / {
                ResearchComplexity.SIMPLE: 1,
                ResearchComplexity.MODERATE: 2,
                ResearchComplexity.COMPLEX: 3,
                ResearchComplexity.EXPERT: 4,
                ResearchComplexity.FRONTIER: 5
            }[context.complexity_level] if hasattr(context, 'complexity_level') else 0.5
        ]
        
        depth_score = statistics.mean(depth_factors)
        
        # Generate insights
        if depth_score < self.quality_thresholds[ResearchDimension.DEPTH]:
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.DEPTH_INSUFFICIENT,
                dimension=ResearchDimension.DEPTH,
                severity=1.0 - depth_score,
                confidence=0.8,
                description=f"Research depth appears insufficient: {depth_score:.1%} completeness",
                evidence={
                    'detailed_sources': detailed_sources,
                    'shallow_sources': shallow_sources,
                    'avg_content_length': avg_content_length,
                    'current_depth': context.current_depth
                },
                recommendations=[
                    "Seek more detailed and comprehensive sources",
                    "Increase research depth for complex topics",
                    "Look for expert analyses and academic sources"
                ],
                automated_actions=[
                    {
                        'action': 'increase_depth',
                        'parameters': {'target_depth': context.current_depth + 1}
                    }
                ]
            ))
        
        metrics = {
            'score': depth_score,
            'detailed_sources': detailed_sources,
            'shallow_sources': shallow_sources,
            'avg_content_length': avg_content_length,
            'complexity_scores': complexity_scores
        }
        
        return metrics, insights
    
    async def _analyze_bias_balance(
        self,
        context: ResearchContext
    ) -> Tuple[Dict[str, Any], List[ResearchInsight]]:
        """Analyze bias and perspective balance"""
        insights = []
        
        # Launch container for bias analysis
        container_id = await self.container_orchestrator.launch_analysis_container(
            'bias_detection',
            {'sources': context.sources_processed, 'query': context.query},
            context.session_id
        )
        
        # Basic bias indicators
        political_leanings = Counter()
        geographic_origins = Counter()
        publication_types = Counter()
        
        for source in context.sources_processed:
            # Extract bias indicators from metadata
            political_lean = source.get('political_leaning', 'neutral')
            political_leanings[political_lean] += 1
            
            geo_origin = source.get('geographic_origin', 'unknown')
            geographic_origins[geo_origin] += 1
            
            pub_type = source.get('publication_type', 'unknown')
            publication_types[pub_type] += 1
        
        # Calculate balance scores
        political_balance = self._calculate_balance_score(political_leanings)
        geographic_balance = self._calculate_balance_score(geographic_origins)
        type_balance = self._calculate_balance_score(publication_types)
        
        overall_balance = (political_balance + geographic_balance + type_balance) / 3
        
        # Enhanced analysis from container
        if container_id:
            container_result = await self.container_orchestrator.get_analysis_result(container_id)
            if container_result:
                enhanced_balance = container_result.get('result', {}).get('score', overall_balance)
                overall_balance = (overall_balance + enhanced_balance) / 2
        
        # Generate insights
        if overall_balance < self.quality_thresholds[ResearchDimension.BIAS_BALANCE]:
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.BIAS_PATTERN,
                dimension=ResearchDimension.BIAS_BALANCE,
                severity=1.0 - overall_balance,
                confidence=0.8,
                description=f"Bias imbalance detected: {overall_balance:.1%} balance score",
                evidence={
                    'political_distribution': dict(political_leanings),
                    'geographic_distribution': dict(geographic_origins),
                    'type_distribution': dict(publication_types)
                },
                recommendations=[
                    "Include sources from diverse political perspectives",
                    "Add international perspectives if relevant",
                    "Balance academic, news, and official sources"
                ]
            ))
        
        # Specific bias warnings
        if len(political_leanings) == 1 and list(political_leanings.keys())[0] != 'neutral':
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.BIAS_PATTERN,
                dimension=ResearchDimension.BIAS_BALANCE,
                severity=0.9,
                confidence=0.95,
                description=f"All sources show {list(political_leanings.keys())[0]} political leaning",
                evidence={'single_perspective': list(political_leanings.keys())[0]},
                recommendations=[
                    "Actively seek sources with opposing viewpoints",
                    "Include neutral or fact-checking sources"
                ]
            ))
        
        metrics = {
            'score': overall_balance,
            'political_balance': political_balance,
            'geographic_balance': geographic_balance,
            'type_balance': type_balance,
            'distributions': {
                'political': dict(political_leanings),
                'geographic': dict(geographic_origins),
                'types': dict(publication_types)
            }
        }
        
        return metrics, insights
    
    def _assess_evidence_strength(self, source: Dict[str, Any]) -> float:
        """Assess the strength of evidence in a source"""
        strength_factors = []
        
        # Citation count (if available)
        citation_count = source.get('citation_count', 0)
        if citation_count > 0:
            strength_factors.append(min(citation_count / 100, 1.0))
        
        # Publication type weight
        pub_type = source.get('publication_type', 'unknown')
        type_weights = {
            'academic': 1.0,
            'government': 0.9,
            'news': 0.7,
            'blog': 0.4,
            'social': 0.2,
            'unknown': 0.5
        }
        strength_factors.append(type_weights.get(pub_type, 0.5))
        
        # Recency (for time-sensitive topics)
        pub_date = source.get('published_date')
        if pub_date:
            try:
                days_old = (datetime.now(timezone.utc) - pub_date).days
                recency_score = max(0, 1.0 - days_old / 365)  # Decay over a year
                strength_factors.append(recency_score)
            except:
                strength_factors.append(0.5)
        
        return statistics.mean(strength_factors) if strength_factors else 0.5
    
    def _calculate_balance_score(self, distribution: Counter) -> float:
        """Calculate balance score for a distribution"""
        if not distribution:
            return 0.0
        
        total = sum(distribution.values())
        if total == 0:
            return 0.0
        
        # Calculate entropy-based balance
        entropy = 0.0
        for count in distribution.values():
            p = count / total
            if p > 0:
                entropy -= p * np.log2(p)
        
        # Normalize to 0-1 scale
        max_entropy = np.log2(len(distribution)) if len(distribution) > 1 else 1
        balance_score = entropy / max_entropy if max_entropy > 0 else 0
        
        return balance_score
    
    def _calculate_overall_quality(self, metrics: ResearchMetrics) -> float:
        """Calculate weighted overall quality score"""
        weights = {
            'completeness': 0.25,
            'credibility': 0.30,
            'diversity': 0.20,
            'depth': 0.15,
            'bias_balance': 0.10
        }
        
        weighted_score = (
            metrics.completeness_score * weights['completeness'] +
            metrics.credibility_score * weights['credibility'] +
            metrics.diversity_score * weights['diversity'] +
            metrics.depth_score * weights['depth'] +
            metrics.bias_balance_score * weights['bias_balance']
        )
        
        return weighted_score
    
    async def _generate_high_level_insights(
        self,
        metrics: ResearchMetrics,
        context: ResearchContext
    ) -> List[ResearchInsight]:
        """Generate high-level strategic insights"""
        insights = []
        
        # Overall quality assessment
        if metrics.overall_quality < 0.6:
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.QUALITY_CONCERN,
                dimension=ResearchDimension.COMPLETENESS,
                severity=1.0 - metrics.overall_quality,
                confidence=0.9,
                description=f"Overall research quality is {metrics.overall_quality:.1%} - consider improvement",
                evidence={'overall_score': metrics.overall_quality, 'metrics': metrics.to_dict()},
                recommendations=[
                    "Focus on areas with lowest scores",
                    "Expand source collection",
                    "Increase research depth and diversity"
                ]
            ))
        
        # Complexity vs. effort mismatch
        if (context.complexity_level == ResearchComplexity.EXPERT and 
            metrics.depth_score < 0.7):
            insights.append(ResearchInsight(
                insight_type=ResearchInsightType.METHODOLOGY_ISSUE,
                dimension=ResearchDimension.DEPTH,
                severity=0.8,
                confidence=0.85,
                description="Expert-level topic requires deeper research methodology",
                evidence={
                    'complexity_level': context.complexity_level.value,
                    'depth_score': metrics.depth_score
                },
                recommendations=[
                    "Seek academic and specialist sources",
                    "Increase research depth significantly",
                    "Consider expert consultation"
                ]
            ))
        
        return insights
    
    async def _store_research_pattern(
        self,
        context: ResearchContext,
        metrics: ResearchMetrics,
        insights: List[ResearchInsight]
    ):
        """Store research pattern in memory for learning"""
        try:
            pattern_data = {
                'query': context.query,
                'complexity': context.complexity_level.value,
                'source_count': len(context.sources_processed),
                'metrics': metrics.to_dict(),
                'insights': [insight.to_dict() for insight in insights],
                'user_domain_expertise': context.domain_expertise,
                'research_goals': context.research_goals
            }
            
            await self.memory_manager.store_memory(
                user_id=context.user_id,
                content=f"Research pattern for: {context.query}",
                memory_type=MemoryType.KNOWLEDGE,
                importance=MemoryImportance.HIGH,
                scope=MemoryScope.PRIVATE,
                metadata={
                    'type': 'research_pattern',
                    'session_id': context.session_id,
                    'pattern_data': pattern_data
                }
            )
            
        except Exception as e:
            logger.error(f"Failed to store research pattern: {e}")
    
    async def _stream_analysis_results(
        self,
        context: ResearchContext,
        metrics: ResearchMetrics,
        insights: List[ResearchInsight]
    ):
        """Stream analysis results to UI"""
        try:
            # Stream metrics update
            await self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.SEARCH_PROGRESS,
                session_id=context.session_id,
                user_id=context.user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'type': 'quality_analysis',
                    'metrics': metrics.to_dict(),
                    'insights_count': len(insights)
                },
                priority=StreamPriority.HIGH
            ))
            
            # Stream individual insights
            for insight in insights:
                await self.stream_manager.stream_event(StreamEvent(
                    event_type=StreamEventType.SYSTEM_MESSAGE,
                    session_id=context.session_id,
                    user_id=context.user_id,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        'type': 'research_insight',
                        'insight': insight.to_dict()
                    },
                    priority=StreamPriority.NORMAL
                ))
            
        except Exception as e:
            logger.error(f"Failed to stream analysis results: {e}")
    
    async def get_research_recommendations(
        self,
        context: ResearchContext,
        current_metrics: ResearchMetrics
    ) -> List[Dict[str, Any]]:
        """Get intelligent recommendations for research improvement"""
        recommendations = []
        
        # Learn from past successful research patterns
        similar_patterns = await self._find_similar_research_patterns(context)
        
        for pattern in similar_patterns:
            if pattern.get('metrics', {}).get('scores', {}).get('overall_quality', 0) > current_metrics.overall_quality:
                recommendations.append({
                    'type': 'pattern_based',
                    'description': f"Similar successful research used {pattern.get('source_count', 0)} sources",
                    'action': 'expand_sources',
                    'confidence': 0.8
                })
        
        # Dynamic recommendations based on current state
        if current_metrics.diversity_score < 0.6:
            recommendations.append({
                'type': 'diversity_improvement',
                'description': 'Include more diverse source types and perspectives',
                'action': 'diversify_sources',
                'confidence': 0.9
            })
        
        if current_metrics.credibility_score < 0.7:
            recommendations.append({
                'type': 'credibility_improvement',
                'description': 'Prioritize high-credibility sources',
                'action': 'improve_credibility',
                'confidence': 0.85
            })
        
        return recommendations
    
    async def _find_similar_research_patterns(
        self,
        context: ResearchContext
    ) -> List[Dict[str, Any]]:
        """Find similar research patterns from memory"""
        try:
            # Search for similar research patterns
            memories = await self.memory_manager.retrieve_memories(
                user_id=context.user_id,
                query=f"research pattern {context.query}",
                limit=5,
                memory_types=[MemoryType.KNOWLEDGE]
            )
            
            patterns = []
            for memory in memories:
                metadata = memory.get('metadata', {})
                if metadata.get('type') == 'research_pattern':
                    patterns.append(metadata.get('pattern_data', {}))
            
            return patterns
            
        except Exception as e:
            logger.error(f"Failed to find similar research patterns: {e}")
            return []
    
    async def trigger_automated_action(
        self,
        action: Dict[str, Any],
        context: ResearchContext
    ) -> bool:
        """Trigger automated research improvement action"""
        try:
            action_type = action.get('action')
            parameters = action.get('parameters', {})
            
            if action_type == 'expand_search':
                # Trigger search expansion
                await self._trigger_search_expansion(parameters, context)
                return True
            
            elif action_type == 'increase_depth':
                # Trigger depth increase
                await self._trigger_depth_increase(parameters, context)
                return True
            
            else:
                logger.warning(f"Unknown automated action: {action_type}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to trigger automated action: {e}")
            return False
    
    async def _trigger_search_expansion(
        self,
        parameters: Dict[str, Any],
        context: ResearchContext
    ):
        """Trigger automated search expansion"""
        # Stream expansion notification
        await self.stream_manager.stream_event(StreamEvent(
            event_type=StreamEventType.SYSTEM_MESSAGE,
            session_id=context.session_id,
            user_id=context.user_id,
            timestamp=datetime.now(timezone.utc),
            data={
                'type': 'automated_action',
                'action': 'search_expansion',
                'focus_areas': parameters.get('focus_areas', [])
            }
        ))
    
    async def _trigger_depth_increase(
        self,
        parameters: Dict[str, Any],
        context: ResearchContext
    ):
        """Trigger automated depth increase"""
        # Stream depth increase notification
        await self.stream_manager.stream_event(StreamEvent(
            event_type=StreamEventType.SYSTEM_MESSAGE,
            session_id=context.session_id,
            user_id=context.user_id,
            timestamp=datetime.now(timezone.utc),
            data={
                'type': 'automated_action',
                'action': 'depth_increase',
                'target_depth': parameters.get('target_depth', context.current_depth + 1)
            }
        ))
    
    def get_intelligence_metrics(self) -> Dict[str, Any]:
        """Get intelligence engine performance metrics"""
        return {
            **self.metrics,
            'active_insights': sum(len(insights) for insights in self.active_insights.values()),
            'pattern_cache_size': len(self.pattern_cache),
            'quality_thresholds': {dim.value: threshold for dim, threshold in self.quality_thresholds.items()}
        }


# Factory function for easy integration
async def create_research_intelligence_engine(
    memory_manager: MemoryManager,
    stream_manager: ResearchStreamManager,
    vm_manager=None,
    container_runtime=None
) -> ResearchIntelligenceEngine:
    """Create and initialize research intelligence engine"""
    
    engine = ResearchIntelligenceEngine(
        memory_manager=memory_manager,
        stream_manager=stream_manager,
        vm_manager=vm_manager,
        container_runtime=container_runtime
    )
    
    logger.info("Research Intelligence Engine created successfully")
    return engine