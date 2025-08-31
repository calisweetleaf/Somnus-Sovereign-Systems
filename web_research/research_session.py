"""
SOMNUS RESEARCH - Research Session
Core session object that tracks everything about a single research task across its entire lifecycle.

Following the Web Research FileTree and Deep Research Subsystem architecture.
Integrates cognitive concepts from Final Caching Strategy into the session tracking.

Tracks:
- Query string and session metadata
- Research plan object (from deep_research_planner.py)
- All search results (raw and processed with cognitive assessment)
- Contradictions, biases, logical gaps with metacognitive monitoring
- Citations and entity graph with semantic relationships
- Output format targets and export history
- Real-time collaboration and user interventions

Used by: planner, engine, memory index, and exporter
Can be serialized to .json for persistence or debugging
"""

import asyncio
import hashlib
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path

import numpy as np
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer

# Somnus system imports
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope
from backend.memory_integration import SessionMemoryContext
from settings.global.user_registry import UserRegistryManager
from research_stream_manager import ResearchStreamManager, StreamEvent, StreamEventType, StreamPriority

logger = logging.getLogger(__name__)


class ResearchStatus(str, Enum):
    """Research session status"""
    PLANNING = "planning"
    ACTIVE = "active"
    PAUSED = "paused"
    ANALYZING = "analyzing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    ERROR = "error"
    CANCELLED = "cancelled"


class ResearchDepthLevel(str, Enum):
    """Research depth levels for recursive expansion"""
    SURFACE = "surface"        # Basic search results
    MODERATE = "moderate"      # Some source analysis
    DEEP = "deep"             # Comprehensive analysis
    EXHAUSTIVE = "exhaustive"  # Full recursive reasoning


class ContradictionSeverity(str, Enum):
    """Severity levels for detected contradictions"""
    MINOR = "minor"           # Small discrepancies
    MODERATE = "moderate"     # Conflicting information
    MAJOR = "major"          # Direct contradictions
    CRITICAL = "critical"    # Fundamental disagreements


class EthicalConcern(str, Enum):
    """Types of ethical concerns detected in research"""
    BIAS = "bias"
    MISINFORMATION = "misinformation"
    HARMFUL_CONTENT = "harmful_content"
    PRIVACY_VIOLATION = "privacy_violation"
    MANIPULATION = "manipulation"


@dataclass
class ResearchEntity:
    """Individual research entity with cognitive assessment (simplified TriaxialKnowledgeEntity)"""
    # Core identification
    entity_id: str
    content_hash: str
    source_url: str
    content_text: str
    timestamp: float
    
    # Cognitive assessment (simplified from tensor operations)
    semantic_embedding: Optional[np.ndarray] = None
    credibility_score: float = 0.5
    reliability_score: float = 0.5
    bias_score: float = 0.5  # 0 = highly biased, 1 = unbiased
    completeness_score: float = 0.5
    
    # Metadata
    source_type: str = "web_page"  # web_page, pdf, video, etc.
    extraction_method: str = "browser"
    language: str = "en"
    
    # Relationships
    contradicts: List[str] = field(default_factory=list)  # Entity IDs that contradict this one
    supports: List[str] = field(default_factory=list)     # Entity IDs that support this one
    related: List[str] = field(default_factory=list)      # Semantically related entities
    
    # Processing history
    processed_by: List[str] = field(default_factory=list)  # Which engines processed this
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def update_cognitive_scores(self, credibility: float, bias: float, completeness: float):
        """Update cognitive assessment scores"""
        self.credibility_score = max(0.0, min(1.0, credibility))
        self.bias_score = max(0.0, min(1.0, bias))
        self.completeness_score = max(0.0, min(1.0, completeness))
        self.last_updated = datetime.now(timezone.utc)
    
    def add_relationship(self, entity_id: str, relationship_type: str):
        """Add relationship to another entity"""
        if relationship_type == "contradicts" and entity_id not in self.contradicts:
            self.contradicts.append(entity_id)
        elif relationship_type == "supports" and entity_id not in self.supports:
            self.supports.append(entity_id)
        elif relationship_type == "related" and entity_id not in self.related:
            self.related.append(entity_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'entity_id': self.entity_id,
            'content_hash': self.content_hash,
            'source_url': self.source_url,
            'content_text': self.content_text,
            'timestamp': self.timestamp,
            'semantic_embedding': self.semantic_embedding.tolist() if self.semantic_embedding is not None else None,
            'credibility_score': self.credibility_score,
            'reliability_score': self.reliability_score,
            'bias_score': self.bias_score,
            'completeness_score': self.completeness_score,
            'source_type': self.source_type,
            'extraction_method': self.extraction_method,
            'language': self.language,
            'contradicts': self.contradicts,
            'supports': self.supports,
            'related': self.related,
            'processed_by': self.processed_by,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class ResearchContradiction:
    """Detected contradiction between research entities"""
    contradiction_id: str
    entity1_id: str
    entity2_id: str
    severity: ContradictionSeverity
    confidence: float
    description: str
    detected_at: datetime
    resolution_status: str = "unresolved"  # unresolved, investigating, resolved
    resolution_notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'contradiction_id': self.contradiction_id,
            'entity1_id': self.entity1_id,
            'entity2_id': self.entity2_id,
            'severity': self.severity.value,
            'confidence': self.confidence,
            'description': self.description,
            'detected_at': self.detected_at.isoformat(),
            'resolution_status': self.resolution_status,
            'resolution_notes': self.resolution_notes
        }


@dataclass
class ResearchPlan:
    """Research plan object (created by deep_research_planner.py)"""
    plan_id: str
    primary_question: str
    sub_questions: List[str]
    research_goals: List[str]
    depth_level: ResearchDepthLevel
    max_sources: int
    trusted_domains: List[str] = field(default_factory=list)
    blocked_domains: List[str] = field(default_factory=list)
    source_types: List[str] = field(default_factory=lambda: ["web", "academic", "news"])
    ethical_guidelines: Dict[str, Any] = field(default_factory=dict)
    time_constraints: Dict[str, Any] = field(default_factory=dict)
    
    # Plan evolution tracking
    version: int = 1
    modification_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def modify_plan(self, modifications: Dict[str, Any], reason: str):
        """Modify the research plan and track changes"""
        self.modification_history.append({
            'version': self.version,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'reason': reason,
            'changes': modifications.copy()
        })
        
        # Apply modifications
        for key, value in modifications.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        self.version += 1
        self.last_modified = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'plan_id': self.plan_id,
            'primary_question': self.primary_question,
            'sub_questions': self.sub_questions,
            'research_goals': self.research_goals,
            'depth_level': self.depth_level.value,
            'max_sources': self.max_sources,
            'trusted_domains': self.trusted_domains,
            'blocked_domains': self.blocked_domains,
            'source_types': self.source_types,
            'ethical_guidelines': self.ethical_guidelines,
            'time_constraints': self.time_constraints,
            'version': self.version,
            'modification_history': self.modification_history,
            'created_at': self.created_at.isoformat(),
            'last_modified': self.last_modified.isoformat()
        }


@dataclass
class ResearchMetrics:
    """Research session performance and quality metrics"""
    # Performance metrics
    total_sources_found: int = 0
    sources_processed: int = 0
    processing_time_seconds: float = 0.0
    
    # Quality metrics
    average_credibility: float = 0.0
    average_bias_score: float = 0.0
    contradiction_count: int = 0
    resolution_rate: float = 0.0
    
    # Coverage metrics
    depth_achieved: str = "surface"
    goals_addressed: int = 0
    questions_answered: int = 0
    
    # Efficiency metrics
    sources_per_minute: float = 0.0
    insights_generated: int = 0
    synthesis_attempts: int = 0
    
    def update_metrics(self, entities: List[ResearchEntity], contradictions: List[ResearchContradiction]):
        """Update metrics based on current research state"""
        if entities:
            self.sources_processed = len(entities)
            self.average_credibility = sum(e.credibility_score for e in entities) / len(entities)
            self.average_bias_score = sum(e.bias_score for e in entities) / len(entities)
        
        self.contradiction_count = len(contradictions)
        if contradictions:
            resolved_count = len([c for c in contradictions if c.resolution_status == "resolved"])
            self.resolution_rate = resolved_count / len(contradictions)
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResearchSession:
    """
    Core research session object that tracks everything about a research task.
    
    This is the central hub that coordinates between:
    - deep_research_planner.py (plan generation)
    - research_engine.py (execution) 
    - ai_browser_research_agent.py (data collection)
    - report_exporter.py (output generation)
    - research_memory_index.py (persistence)
    """
    
    def __init__(self,
                 session_id: Optional[str] = None,
                 user_id: Optional[str] = None,
                 query: str = "",
                 memory_manager: Optional[MemoryManager] = None,
                 stream_manager: Optional[ResearchStreamManager] = None):
        
        # Core identification
        self.session_id = session_id or str(uuid4())
        self.user_id = user_id or "unknown"
        self.query = query
        
        # Research state
        self.status = ResearchStatus.PLANNING
        self.plan: Optional[ResearchPlan] = None
        self.current_depth = 0
        self.max_depth = 3
        
        # Data storage
        self.entities: Dict[str, ResearchEntity] = {}
        self.contradictions: Dict[str, ResearchContradiction] = {}
        self.citations: List[Dict[str, Any]] = []
        self.synthesis_results: List[Dict[str, Any]] = []
        
        # Quality tracking
        self.ethical_concerns: List[Dict[str, Any]] = []
        self.bias_detections: List[Dict[str, Any]] = []
        self.quality_flags: List[str] = []
        
        # Progress tracking
        self.metrics = ResearchMetrics()
        self.processing_history: List[Dict[str, Any]] = []
        
        # Integration systems
        self.memory_manager = memory_manager
        self.stream_manager = stream_manager
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Session metadata
        self.created_at = datetime.now(timezone.utc)
        self.last_updated = datetime.now(timezone.utc)
        self.completion_time: Optional[datetime] = None
        
        # Output tracking
        self.export_formats: List[str] = []
        self.artifact_exports: List[str] = []  # Links to generated artifacts
        
        # Collaboration tracking
        self.collaborators: List[str] = []
        self.user_interventions: List[Dict[str, Any]] = []
        
        logger.info(f"ResearchSession created: {self.session_id}")
    
    def set_plan(self, plan: ResearchPlan):
        """Set the research plan (from deep_research_planner.py)"""
        self.plan = plan
        self.max_depth = {
            ResearchDepthLevel.SURFACE: 1,
            ResearchDepthLevel.MODERATE: 2,
            ResearchDepthLevel.DEEP: 3,
            ResearchDepthLevel.EXHAUSTIVE: 5
        }.get(plan.depth_level, 3)
        
        # Stream plan update
        if self.stream_manager:
            asyncio.create_task(self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.PLAN_GENERATED,
                session_id=self.session_id,
                user_id=self.user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'plan_id': plan.plan_id,
                    'primary_question': plan.primary_question,
                    'depth_level': plan.depth_level.value,
                    'max_sources': plan.max_sources
                },
                priority=StreamPriority.HIGH
            )))
    
    def add_entity(self, 
                   content: str, 
                   source_url: str, 
                   source_type: str = "web_page",
                   extraction_method: str = "browser") -> ResearchEntity:
        """Add a new research entity to the session"""
        
        # Generate content hash and entity ID
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        entity_id = f"entity_{len(self.entities)}_{content_hash[:8]}"
        
        # Generate semantic embedding
        embedding = self.embedding_model.encode([content])[0]
        
        # Create entity
        entity = ResearchEntity(
            entity_id=entity_id,
            content_hash=content_hash,
            source_url=source_url,
            content_text=content,
            timestamp=time.time(),
            semantic_embedding=embedding,
            source_type=source_type,
            extraction_method=extraction_method
        )
        
        # Store entity
        self.entities[entity_id] = entity
        
        # Check for contradictions with existing entities
        self._detect_contradictions(entity)
        
        # Update metrics
        self.metrics.update_metrics(list(self.entities.values()), list(self.contradictions.values()))
        self.last_updated = datetime.now(timezone.utc)
        
        # Stream entity addition
        if self.stream_manager:
            asyncio.create_task(self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.SOURCE_FOUND,
                session_id=self.session_id,
                user_id=self.user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'entity_id': entity_id,
                    'source_url': source_url,
                    'source_type': source_type,
                    'content_length': len(content)
                },
                priority=StreamPriority.NORMAL
            )))
        
        logger.info(f"Added entity {entity_id} to session {self.session_id}")
        return entity
    
    def update_entity_scores(self, entity_id: str, credibility: float, bias: float, completeness: float):
        """Update cognitive assessment scores for an entity"""
        if entity_id in self.entities:
            self.entities[entity_id].update_cognitive_scores(credibility, bias, completeness)
            
            # Update overall metrics
            self.metrics.update_metrics(list(self.entities.values()), list(self.contradictions.values()))
            
            # Stream score update
            if self.stream_manager:
                asyncio.create_task(self.stream_manager.stream_event(StreamEvent(
                    event_type=StreamEventType.SOURCE_ANALYZED,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        'entity_id': entity_id,
                        'credibility_score': credibility,
                        'bias_score': bias,
                        'completeness_score': completeness
                    },
                    priority=StreamPriority.NORMAL
                )))
    
    def add_contradiction(self, 
                         entity1_id: str, 
                         entity2_id: str, 
                         severity: ContradictionSeverity,
                         confidence: float,
                         description: str) -> ResearchContradiction:
        """Add a detected contradiction between entities"""
        
        contradiction_id = f"contradiction_{len(self.contradictions)}_{int(time.time())}"
        
        contradiction = ResearchContradiction(
            contradiction_id=contradiction_id,
            entity1_id=entity1_id,
            entity2_id=entity2_id,
            severity=severity,
            confidence=confidence,
            description=description,
            detected_at=datetime.now(timezone.utc)
        )
        
        self.contradictions[contradiction_id] = contradiction
        
        # Update entity relationships
        if entity1_id in self.entities:
            self.entities[entity1_id].add_relationship(entity2_id, "contradicts")
        if entity2_id in self.entities:
            self.entities[entity2_id].add_relationship(entity1_id, "contradicts")
        
        # Stream contradiction detection
        if self.stream_manager:
            asyncio.create_task(self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.CONTRADICTION_DETECTED,
                session_id=self.session_id,
                user_id=self.user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'contradiction_id': contradiction_id,
                    'entity1_id': entity1_id,
                    'entity2_id': entity2_id,
                    'severity': severity.value,
                    'confidence': confidence,
                    'description': description
                },
                priority=StreamPriority.HIGH
            )))
        
        logger.warning(f"Contradiction detected: {entity1_id} vs {entity2_id} (severity: {severity.value})")
        return contradiction
    
    def add_ethical_concern(self, concern_type: EthicalConcern, description: str, entity_id: Optional[str] = None):
        """Add an ethical concern detected during research"""
        concern = {
            'concern_id': str(uuid4()),
            'type': concern_type.value,
            'description': description,
            'entity_id': entity_id,
            'detected_at': datetime.now(timezone.utc).isoformat(),
            'severity': 'medium'  # Could be determined by analysis
        }
        
        self.ethical_concerns.append(concern)
        self.quality_flags.append(f"ethical_concern_{concern_type.value}")
        
        logger.warning(f"Ethical concern detected: {concern_type.value} - {description}")
    
    def record_user_intervention(self, intervention_type: str, data: Dict[str, Any]):
        """Record a user intervention in the research process"""
        intervention = {
            'intervention_id': str(uuid4()),
            'type': intervention_type,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_state_before': self.get_session_summary()
        }
        
        self.user_interventions.append(intervention)
        
        # Stream intervention
        if self.stream_manager:
            asyncio.create_task(self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.USER_INTERVENTION,
                session_id=self.session_id,
                user_id=self.user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'intervention_type': intervention_type,
                    'intervention_data': data
                },
                priority=StreamPriority.HIGH
            )))
    
    def modify_plan(self, modifications: Dict[str, Any], reason: str):
        """Modify the research plan mid-session"""
        if self.plan:
            self.plan.modify_plan(modifications, reason)
            
            # Stream plan modification
            if self.stream_manager:
                asyncio.create_task(self.stream_manager.stream_event(StreamEvent(
                    event_type=StreamEventType.PLAN_MODIFIED,
                    session_id=self.session_id,
                    user_id=self.user_id,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        'plan_version': self.plan.version,
                        'modifications': modifications,
                        'reason': reason
                    },
                    priority=StreamPriority.HIGH
                )))
    
    def update_status(self, new_status: ResearchStatus):
        """Update research session status"""
        old_status = self.status
        self.status = new_status
        self.last_updated = datetime.now(timezone.utc)
        
        if new_status == ResearchStatus.COMPLETED:
            self.completion_time = datetime.now(timezone.utc)
        
        # Stream status update
        if self.stream_manager:
            asyncio.create_task(self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.SEARCH_PROGRESS,
                session_id=self.session_id,
                user_id=self.user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'old_status': old_status.value,
                    'new_status': new_status.value,
                    'progress_summary': self.get_progress_summary()
                },
                priority=StreamPriority.NORMAL
            )))
        
        logger.info(f"Session {self.session_id} status: {old_status.value} -> {new_status.value}")
    
    def add_synthesis_result(self, synthesis_data: Dict[str, Any]):
        """Add a knowledge synthesis result"""
        synthesis = {
            'synthesis_id': str(uuid4()),
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'input_entities': synthesis_data.get('input_entities', []),
            'result': synthesis_data.get('result', ''),
            'confidence': synthesis_data.get('confidence', 0.0),
            'method': synthesis_data.get('method', 'unknown')
        }
        
        self.synthesis_results.append(synthesis)
        self.metrics.synthesis_attempts += 1
        
        if synthesis_data.get('confidence', 0) > 0.7:
            self.metrics.insights_generated += 1
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get a comprehensive summary of the research session"""
        return {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'query': self.query,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'completion_time': self.completion_time.isoformat() if self.completion_time else None,
            
            # Content summary
            'entity_count': len(self.entities),
            'contradiction_count': len(self.contradictions),
            'ethical_concerns': len(self.ethical_concerns),
            'synthesis_results': len(self.synthesis_results),
            
            # Quality metrics
            'metrics': self.metrics.to_dict(),
            
            # Plan summary
            'plan_summary': self.plan.to_dict() if self.plan else None,
            
            # Progress indicators
            'current_depth': self.current_depth,
            'max_depth': self.max_depth,
            'user_interventions': len(self.user_interventions)
        }
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get current progress summary"""
        total_goals = len(self.plan.research_goals) if self.plan else 1
        total_questions = len(self.plan.sub_questions) if self.plan else 1
        
        return {
            'completion_percentage': min(100, (self.current_depth / self.max_depth) * 100),
            'entities_processed': len(self.entities),
            'contradictions_found': len(self.contradictions),
            'goals_progress': f"{self.metrics.goals_addressed}/{total_goals}",
            'questions_progress': f"{self.metrics.questions_answered}/{total_questions}",
            'quality_score': self.metrics.average_credibility,
            'time_elapsed': (datetime.now(timezone.utc) - self.created_at).total_seconds()
        }
    
    def _detect_contradictions(self, new_entity: ResearchEntity):
        """Detect contradictions between new entity and existing entities"""
        for existing_id, existing_entity in self.entities.items():
            if existing_id == new_entity.entity_id:
                continue
            
            # Use semantic similarity to detect potential contradictions
            if (new_entity.semantic_embedding is not None and 
                existing_entity.semantic_embedding is not None):
                
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(
                    new_entity.semantic_embedding.reshape(1, -1),
                    existing_entity.semantic_embedding.reshape(1, -1)
                )[0, 0]
                
                # High semantic similarity but different credibility scores might indicate contradiction
                credibility_diff = abs(new_entity.credibility_score - existing_entity.credibility_score)
                
                if similarity > 0.8 and credibility_diff > 0.4:
                    # Potential contradiction detected
                    severity = ContradictionSeverity.MODERATE
                    if credibility_diff > 0.7:
                        severity = ContradictionSeverity.MAJOR
                    
                    confidence = min(1.0, similarity * credibility_diff)
                    description = f"High semantic similarity ({similarity:.2f}) but significant credibility difference ({credibility_diff:.2f})"
                    
                    self.add_contradiction(
                        new_entity.entity_id,
                        existing_id,
                        severity,
                        confidence,
                        description
                    )
    
    async def store_in_memory(self):
        """Store research session in persistent memory"""
        if not self.memory_manager:
            return
        
        try:
            # Store session summary
            await self.memory_manager.store_memory(
                user_id=self.user_id,
                content=f"Research Session: {self.query}",
                memory_type=MemoryType.DEV_SESSION,
                importance=MemoryImportance.HIGH,
                scope=MemoryScope.PRIVATE,
                metadata={
                    'type': 'research_session',
                    'session_id': self.session_id,
                    'entity_count': len(self.entities),
                    'contradiction_count': len(self.contradictions),
                    'status': self.status.value,
                    'quality_score': self.metrics.average_credibility
                }
            )
            
            # Store key entities
            for entity in list(self.entities.values())[:10]:  # Store top 10 entities
                if entity.credibility_score > 0.7:  # Only store high-quality entities
                    await self.memory_manager.store_memory(
                        user_id=self.user_id,
                        content=entity.content_text[:500],  # Truncate for memory
                        memory_type=MemoryType.DOCUMENT,
                        importance=MemoryImportance.MEDIUM,
                        scope=MemoryScope.PRIVATE,
                        metadata={
                            'type': 'research_entity',
                            'session_id': self.session_id,
                            'entity_id': entity.entity_id,
                            'source_url': entity.source_url,
                            'credibility_score': entity.credibility_score
                        }
                    )
            
            logger.info(f"Stored research session {self.session_id} in memory")
            
        except Exception as e:
            logger.error(f"Failed to store research session in memory: {e}")
    
    def to_json(self) -> str:
        """Serialize session to JSON for persistence or debugging"""
        session_data = {
            'session_id': self.session_id,
            'user_id': self.user_id,
            'query': self.query,
            'status': self.status.value,
            'created_at': self.created_at.isoformat(),
            'last_updated': self.last_updated.isoformat(),
            'completion_time': self.completion_time.isoformat() if self.completion_time else None,
            
            # Core data
            'plan': self.plan.to_dict() if self.plan else None,
            'entities': {k: v.to_dict() for k, v in self.entities.items()},
            'contradictions': {k: v.to_dict() for k, v in self.contradictions.items()},
            'citations': self.citations,
            'synthesis_results': self.synthesis_results,
            
            # Quality tracking
            'ethical_concerns': self.ethical_concerns,
            'bias_detections': self.bias_detections,
            'quality_flags': self.quality_flags,
            
            # Metadata
            'metrics': self.metrics.to_dict(),
            'processing_history': self.processing_history,
            'export_formats': self.export_formats,
            'artifact_exports': self.artifact_exports,
            'collaborators': self.collaborators,
            'user_interventions': self.user_interventions,
            'current_depth': self.current_depth,
            'max_depth': self.max_depth
        }
        
        return json.dumps(session_data, indent=2, default=str)
    
    @classmethod
    def from_json(cls, json_data: str, memory_manager=None, stream_manager=None) -> 'ResearchSession':
        """Restore session from JSON data"""
        data = json.loads(json_data)
        
        # Create session
        session = cls(
            session_id=data['session_id'],
            user_id=data['user_id'],
            query=data['query'],
            memory_manager=memory_manager,
            stream_manager=stream_manager
        )
        
        # Restore basic properties
        session.status = ResearchStatus(data['status'])
        session.created_at = datetime.fromisoformat(data['created_at'])
        session.last_updated = datetime.fromisoformat(data['last_updated'])
        if data['completion_time']:
            session.completion_time = datetime.fromisoformat(data['completion_time'])
        
        # Restore plan
        if data['plan']:
            plan_data = data['plan']
            session.plan = ResearchPlan(
                plan_id=plan_data['plan_id'],
                primary_question=plan_data['primary_question'],
                sub_questions=plan_data['sub_questions'],
                research_goals=plan_data['research_goals'],
                depth_level=ResearchDepthLevel(plan_data['depth_level']),
                max_sources=plan_data['max_sources'],
                trusted_domains=plan_data['trusted_domains'],
                blocked_domains=plan_data['blocked_domains'],
                source_types=plan_data['source_types'],
                ethical_guidelines=plan_data['ethical_guidelines'],
                time_constraints=plan_data['time_constraints']
            )
            session.plan.version = plan_data['version']
            session.plan.modification_history = plan_data['modification_history']
            session.plan.created_at = datetime.fromisoformat(plan_data['created_at'])
            session.plan.last_modified = datetime.fromisoformat(plan_data['last_modified'])
        
        # Restore entities (without embeddings - would need to regenerate)
        for entity_id, entity_data in data['entities'].items():
            entity = ResearchEntity(
                entity_id=entity_data['entity_id'],
                content_hash=entity_data['content_hash'],
                source_url=entity_data['source_url'],
                content_text=entity_data['content_text'],
                timestamp=entity_data['timestamp'],
                credibility_score=entity_data['credibility_score'],
                reliability_score=entity_data['reliability_score'],
                bias_score=entity_data['bias_score'],
                completeness_score=entity_data['completeness_score'],
                source_type=entity_data['source_type'],
                extraction_method=entity_data['extraction_method'],
                language=entity_data['language'],
                contradicts=entity_data['contradicts'],
                supports=entity_data['supports'],
                related=entity_data['related'],
                processed_by=entity_data['processed_by']
            )
            entity.last_updated = datetime.fromisoformat(entity_data['last_updated'])
            session.entities[entity_id] = entity
        
        # Restore contradictions
        for contradiction_id, contradiction_data in data['contradictions'].items():
            contradiction = ResearchContradiction(
                contradiction_id=contradiction_data['contradiction_id'],
                entity1_id=contradiction_data['entity1_id'],
                entity2_id=contradiction_data['entity2_id'],
                severity=ContradictionSeverity(contradiction_data['severity']),
                confidence=contradiction_data['confidence'],
                description=contradiction_data['description'],
                detected_at=datetime.fromisoformat(contradiction_data['detected_at']),
                resolution_status=contradiction_data['resolution_status'],
                resolution_notes=contradiction_data['resolution_notes']
            )
            session.contradictions[contradiction_id] = contradiction
        
        # Restore other data
        session.citations = data['citations']
        session.synthesis_results = data['synthesis_results']
        session.ethical_concerns = data['ethical_concerns']
        session.bias_detections = data['bias_detections']
        session.quality_flags = data['quality_flags']
        session.processing_history = data['processing_history']
        session.export_formats = data['export_formats']
        session.artifact_exports = data['artifact_exports']
        session.collaborators = data['collaborators']
        session.user_interventions = data['user_interventions']
        session.current_depth = data['current_depth']
        session.max_depth = data['max_depth']
        
        # Restore metrics
        metrics_data = data['metrics']
        session.metrics = ResearchMetrics(**metrics_data)
        
        logger.info(f"Restored research session {session.session_id} from JSON")
        return session


class ContradictionDetector:
    """
    Advanced contradiction detection with symbolic and temporal reasoning.
    
    Implements detection beyond simple embeddings using:
    - Symbolic reasoning patterns
    - Temporal consistency analysis  
    - Statistical contradiction patterns
    - Multi-modal analysis support
    """
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.contradiction_patterns = self._load_contradiction_patterns()
        self.temporal_window_hours = 24
        
    def _load_contradiction_patterns(self) -> Dict[str, List[str]]:
        """Load linguistic patterns that indicate contradictions"""
        return {
            'negation': ['not', 'never', 'no', 'none', 'neither', 'nor'],
            'opposition': ['but', 'however', 'although', 'despite', 'contrary to', 'opposite'],
            'temporal_conflict': ['before', 'after', 'during', 'while', 'when', 'then'],
            'quantity_conflict': ['more', 'less', 'higher', 'lower', 'increase', 'decrease'],
            'certainty_conflict': ['definitely', 'maybe', 'possibly', 'certainly', 'unlikely']
        }
    
    async def detect_contradictions(self, entities: Dict[str, ResearchEntity]) -> List[ResearchContradiction]:
        """Detect contradictions using multiple analysis methods"""
        contradictions = []
        entity_list = list(entities.values())
        
        for i, entity1 in enumerate(entity_list):
            for j, entity2 in enumerate(entity_list[i+1:], i+1):
                # Semantic similarity check
                semantic_contradiction = self._check_semantic_contradiction(entity1, entity2)
                if semantic_contradiction:
                    contradictions.append(semantic_contradiction)
                
                # Symbolic pattern check
                symbolic_contradiction = self._check_symbolic_contradiction(entity1, entity2)
                if symbolic_contradiction:
                    contradictions.append(symbolic_contradiction)
                
                # Temporal consistency check
                temporal_contradiction = self._check_temporal_contradiction(entity1, entity2)
                if temporal_contradiction:
                    contradictions.append(temporal_contradiction)
        
        return contradictions
    
    def _check_semantic_contradiction(self, entity1: ResearchEntity, entity2: ResearchEntity) -> Optional[ResearchContradiction]:
        """Check for semantic contradictions using embeddings"""
        if entity1.semantic_embedding is None or entity2.semantic_embedding is None:
            return None
        
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            similarity = cosine_similarity(
                entity1.semantic_embedding.reshape(1, -1),
                entity2.semantic_embedding.reshape(1, -1)
            )[0, 0]
            
            # High semantic similarity but different conclusions might indicate contradiction
            credibility_diff = abs(entity1.credibility_score - entity2.credibility_score)
            bias_diff = abs(entity1.bias_score - entity2.bias_score)
            
            if similarity > 0.75 and (credibility_diff > 0.4 or bias_diff > 0.4):
                severity = ContradictionSeverity.MODERATE
                if credibility_diff > 0.7 or bias_diff > 0.7:
                    severity = ContradictionSeverity.MAJOR
                
                confidence = min(1.0, similarity * max(credibility_diff, bias_diff))
                description = f"Semantic similarity {similarity:.2f} with credibility diff {credibility_diff:.2f}, bias diff {bias_diff:.2f}"
                
                return ResearchContradiction(
                    contradiction_id=f"semantic_{int(time.time())}_{entity1.entity_id[:8]}_{entity2.entity_id[:8]}",
                    entity1_id=entity1.entity_id,
                    entity2_id=entity2.entity_id,
                    severity=severity,
                    confidence=confidence,
                    description=description,
                    detected_at=datetime.now(timezone.utc)
                )
        except Exception as e:
            logger.error(f"Error in semantic contradiction detection: {e}")
        
        return None
    
    def _check_symbolic_contradiction(self, entity1: ResearchEntity, entity2: ResearchEntity) -> Optional[ResearchContradiction]:
        """Check for contradictions using symbolic pattern matching"""
        text1 = entity1.content_text.lower()
        text2 = entity2.content_text.lower()
        
        contradiction_score = 0.0
        detected_patterns = []
        
        for pattern_type, patterns in self.contradiction_patterns.items():
            count1 = sum(1 for pattern in patterns if pattern in text1)
            count2 = sum(1 for pattern in patterns if pattern in text2)
            
            # Different pattern usage might indicate contradiction
            if count1 > 0 and count2 > 0:
                pattern_diff = abs(count1 - count2) / max(count1, count2)
                if pattern_diff > 0.5:
                    contradiction_score += 0.2
                    detected_patterns.append(pattern_type)
        
        # Check for direct negation patterns
        for word in ['increase', 'decrease', 'positive', 'negative', 'true', 'false']:
            if word in text1 and f"not {word}" in text2:
                contradiction_score += 0.4
                detected_patterns.append('direct_negation')
        
        if contradiction_score > 0.3:
            severity = ContradictionSeverity.MINOR
            if contradiction_score > 0.6:
                severity = ContradictionSeverity.MODERATE
            if contradiction_score > 0.8:
                severity = ContradictionSeverity.MAJOR
            
            return ResearchContradiction(
                contradiction_id=f"symbolic_{int(time.time())}_{entity1.entity_id[:8]}_{entity2.entity_id[:8]}",
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                severity=severity,
                confidence=min(1.0, contradiction_score),
                description=f"Symbolic patterns: {', '.join(detected_patterns)} (score: {contradiction_score:.2f})",
                detected_at=datetime.now(timezone.utc)
            )
        
        return None
    
    def _check_temporal_contradiction(self, entity1: ResearchEntity, entity2: ResearchEntity) -> Optional[ResearchContradiction]:
        """Check for temporal inconsistencies between entities"""
        # Check if entities are from similar time periods
        time_diff = abs(entity1.timestamp - entity2.timestamp)
        
        # Only check temporal consistency for recent entities
        if time_diff > (self.temporal_window_hours * 3600):
            return None
        
        text1 = entity1.content_text.lower()
        text2 = entity2.content_text.lower()
        
        # Look for temporal indicators
        temporal_indicators = {
            'past': ['was', 'were', 'had', 'did', 'happened', 'occurred'],
            'present': ['is', 'are', 'have', 'do', 'happening', 'occurring'],
            'future': ['will', 'shall', 'going to', 'planned', 'expected']
        }
        
        entity1_tense = None
        entity2_tense = None
        
        for tense, indicators in temporal_indicators.items():
            if any(indicator in text1 for indicator in indicators):
                entity1_tense = tense
            if any(indicator in text2 for indicator in indicators):
                entity2_tense = tense
        
        # Check for temporal contradictions
        if (entity1_tense and entity2_tense and 
            entity1_tense != entity2_tense and 
            time_diff < 3600):  # Within 1 hour
            
            return ResearchContradiction(
                contradiction_id=f"temporal_{int(time.time())}_{entity1.entity_id[:8]}_{entity2.entity_id[:8]}",
                entity1_id=entity1.entity_id,
                entity2_id=entity2.entity_id,
                severity=ContradictionSeverity.MINOR,
                confidence=0.6,
                description=f"Temporal inconsistency: {entity1_tense} vs {entity2_tense} in similar timeframe",
                detected_at=datetime.now(timezone.utc)
            )
        
        return None


class BiasDetector:
    """
    Enhanced bias detection using statistical and language model classifiers.
    
    Detects multiple types of bias:
    - Confirmation bias
    - Selection bias  
    - Language bias
    - Source bias
    - Temporal bias
    """
    
    def __init__(self):
        self.bias_indicators = self._load_bias_indicators()
        self.source_reliability_scores = {}
        
    def _load_bias_indicators(self) -> Dict[str, Dict[str, List[str]]]:
        """Load linguistic indicators for different bias types"""
        return {
            'confirmation_bias': {
                'strong_language': ['definitely', 'absolutely', 'certainly', 'obviously', 'clearly'],
                'weak_qualifiers': ['might', 'possibly', 'perhaps', 'could be', 'seems']
            },
            'selection_bias': {
                'cherry_picking': ['only', 'just', 'merely', 'simply', 'ignore'],
                'overgeneralization': ['all', 'every', 'always', 'never', 'none']
            },
            'emotional_bias': {
                'positive_emotion': ['amazing', 'fantastic', 'incredible', 'outstanding'],
                'negative_emotion': ['terrible', 'awful', 'horrible', 'disaster']
            },
            'authority_bias': {
                'appeal_to_authority': ['expert says', 'study shows', 'research proves'],
                'credentials': ['professor', 'doctor', 'phd', 'university']
            }
        }
    
    def detect_bias(self, entity: ResearchEntity) -> Dict[str, float]:
        """Detect various types of bias in a research entity"""
        bias_scores = {
            'confirmation_bias': self._detect_confirmation_bias(entity),
            'selection_bias': self._detect_selection_bias(entity),
            'emotional_bias': self._detect_emotional_bias(entity),
            'authority_bias': self._detect_authority_bias(entity),
            'source_bias': self._detect_source_bias(entity),
            'language_bias': self._detect_language_bias(entity)
        }
        
        # Calculate overall bias score
        overall_bias = sum(bias_scores.values()) / len(bias_scores)
        bias_scores['overall'] = min(1.0, overall_bias)
        
        return bias_scores
    
    def _detect_confirmation_bias(self, entity: ResearchEntity) -> float:
        """Detect confirmation bias patterns"""
        text = entity.content_text.lower()
        
        strong_count = sum(1 for indicator in self.bias_indicators['confirmation_bias']['strong_language'] 
                          if indicator in text)
        weak_count = sum(1 for indicator in self.bias_indicators['confirmation_bias']['weak_qualifiers']
                        if indicator in text)
        
        # High strong language with low weak qualifiers suggests confirmation bias
        text_length = len(text.split())
        strong_ratio = strong_count / max(1, text_length / 100)
        weak_ratio = weak_count / max(1, text_length / 100)
        
        if strong_ratio > weak_ratio and strong_ratio > 0.5:
            return min(1.0, strong_ratio * 0.3)
        
        return 0.1
    
    def _detect_selection_bias(self, entity: ResearchEntity) -> float:
        """Detect selection bias patterns"""
        text = entity.content_text.lower()
        
        cherry_pick_count = sum(1 for indicator in self.bias_indicators['selection_bias']['cherry_picking']
                               if indicator in text)
        overgeneralization_count = sum(1 for indicator in self.bias_indicators['selection_bias']['overgeneralization']
                                      if indicator in text)
        
        total_indicators = cherry_pick_count + overgeneralization_count
        text_length = len(text.split())
        
        bias_ratio = total_indicators / max(1, text_length / 100)
        return min(1.0, bias_ratio * 0.2)
    
    def _detect_emotional_bias(self, entity: ResearchEntity) -> float:
        """Detect emotional bias patterns"""
        text = entity.content_text.lower()
        
        positive_count = sum(1 for indicator in self.bias_indicators['emotional_bias']['positive_emotion']
                            if indicator in text)
        negative_count = sum(1 for indicator in self.bias_indicators['emotional_bias']['negative_emotion']
                            if indicator in text)
        
        total_emotional = positive_count + negative_count
        text_length = len(text.split())
        
        emotional_ratio = total_emotional / max(1, text_length / 50)
        return min(1.0, emotional_ratio * 0.25)
    
    def _detect_authority_bias(self, entity: ResearchEntity) -> float:
        """Detect authority bias patterns"""
        text = entity.content_text.lower()
        
        authority_appeals = sum(1 for indicator in self.bias_indicators['authority_bias']['appeal_to_authority']
                               if indicator in text)
        credential_mentions = sum(1 for indicator in self.bias_indicators['authority_bias']['credentials']
                                 if indicator in text)
        
        # Some authority reference is good, too much suggests bias
        total_authority = authority_appeals + credential_mentions
        text_length = len(text.split())
        
        authority_ratio = total_authority / max(1, text_length / 100)
        
        if authority_ratio > 2.0:  # Too many authority appeals
            return min(1.0, authority_ratio * 0.15)
        
        return 0.05  # Minimal bias for normal authority references
    
    def _detect_source_bias(self, entity: ResearchEntity) -> float:
        """Detect bias based on source characteristics"""
        source_url = entity.source_url.lower()
        
        # Check for biased domain patterns
        biased_domains = ['.blog', 'opinion', 'editorial', 'personal']
        reliable_domains = ['.edu', '.gov', '.org']
        
        bias_score = 0.0
        
        for biased in biased_domains:
            if biased in source_url:
                bias_score += 0.3
        
        for reliable in reliable_domains:
            if reliable in source_url:
                bias_score = max(0.0, bias_score - 0.2)
        
        # Check source credibility if available
        if entity.credibility_score < 0.5:
            bias_score += (0.5 - entity.credibility_score)
        
        return min(1.0, bias_score)
    
    def _detect_language_bias(self, entity: ResearchEntity) -> float:
        """Detect language bias patterns"""
        text = entity.content_text
        
        # Simple language bias indicators
        caps_count = sum(1 for char in text if char.isupper())
        exclamation_count = text.count('!')
        question_count = text.count('?')
        
        text_length = len(text)
        
        caps_ratio = caps_count / max(1, text_length)
        punctuation_ratio = (exclamation_count + question_count) / max(1, text_length / 100)
        
        language_bias = (caps_ratio * 0.5) + (punctuation_ratio * 0.3)
        return min(1.0, language_bias)


class EntityManager:
    """
    Entity prioritization by credibility, uniqueness, and coverage.
    
    Manages research entities with intelligent prioritization and deduplication.
    """
    
    def __init__(self, embedding_model=None):
        self.embedding_model = embedding_model or SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.similarity_threshold = 0.85  # Threshold for considering entities similar
        
    def prioritize_entities(self, entities: Dict[str, ResearchEntity]) -> List[Tuple[str, float]]:
        """Prioritize entities by credibility, uniqueness, and coverage"""
        entity_scores = []
        
        for entity_id, entity in entities.items():
            score = self._calculate_priority_score(entity, entities)
            entity_scores.append((entity_id, score))
        
        # Sort by priority score (descending)
        entity_scores.sort(key=lambda x: x[1], reverse=True)
        return entity_scores
    
    def _calculate_priority_score(self, entity: ResearchEntity, all_entities: Dict[str, ResearchEntity]) -> float:
        """Calculate priority score for an entity"""
        credibility_weight = 0.4
        uniqueness_weight = 0.3
        coverage_weight = 0.2
        recency_weight = 0.1
        
        # Credibility score (higher is better)
        credibility_score = entity.credibility_score
        
        # Uniqueness score (less similar to others is better)
        uniqueness_score = self._calculate_uniqueness(entity, all_entities)
        
        # Coverage score (more comprehensive content is better)
        coverage_score = self._calculate_coverage(entity)
        
        # Recency score (more recent is better)
        recency_score = self._calculate_recency(entity)
        
        total_score = (
            credibility_score * credibility_weight +
            uniqueness_score * uniqueness_weight +
            coverage_score * coverage_weight +
            recency_score * recency_weight
        )
        
        return total_score
    
    def _calculate_uniqueness(self, entity: ResearchEntity, all_entities: Dict[str, ResearchEntity]) -> float:
        """Calculate how unique an entity is compared to others"""
        if entity.semantic_embedding is None:
            return 0.5  # Neutral score if no embedding
        
        similarities = []
        
        for other_id, other_entity in all_entities.items():
            if other_id == entity.entity_id or other_entity.semantic_embedding is None:
                continue
            
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(
                    entity.semantic_embedding.reshape(1, -1),
                    other_entity.semantic_embedding.reshape(1, -1)
                )[0, 0]
                similarities.append(similarity)
            except Exception:
                continue
        
        if not similarities:
            return 1.0  # Completely unique if no comparisons possible
        
        # Uniqueness is inverse of maximum similarity
        max_similarity = max(similarities)
        uniqueness = 1.0 - max_similarity
        return max(0.0, uniqueness)
    
    def _calculate_coverage(self, entity: ResearchEntity) -> float:
        """Calculate coverage score based on content comprehensiveness"""
        content = entity.content_text
        
        # Simple coverage indicators
        word_count = len(content.split())
        sentence_count = len([s for s in content.split('.') if s.strip()])
        paragraph_count = len([p for p in content.split('\n\n') if p.strip()])
        
        # Normalize scores
        word_score = min(1.0, word_count / 500)  # 500 words = full score
        structure_score = min(1.0, (sentence_count + paragraph_count) / 20)
        completeness_score = entity.completeness_score
        
        coverage = (word_score * 0.4 + structure_score * 0.3 + completeness_score * 0.3)
        return coverage
    
    def _calculate_recency(self, entity: ResearchEntity) -> float:
        """Calculate recency score based on timestamp"""
        now = time.time()
        age_hours = (now - entity.timestamp) / 3600
        
        # Recency score decreases over time
        if age_hours < 1:
            return 1.0
        elif age_hours < 24:
            return 0.8
        elif age_hours < 168:  # 1 week
            return 0.6
        elif age_hours < 720:  # 1 month
            return 0.4
        else:
            return 0.2
    
    def deduplicate_entities(self, entities: Dict[str, ResearchEntity]) -> List[str]:
        """Identify duplicate entities for removal"""
        duplicates = set()
        entity_list = list(entities.values())
        
        for i, entity1 in enumerate(entity_list):
            if entity1.entity_id in duplicates:
                continue
                
            for j, entity2 in enumerate(entity_list[i+1:], i+1):
                if entity2.entity_id in duplicates:
                    continue
                
                if self._are_entities_similar(entity1, entity2):
                    # Keep the one with higher credibility
                    if entity1.credibility_score >= entity2.credibility_score:
                        duplicates.add(entity2.entity_id)
                    else:
                        duplicates.add(entity1.entity_id)
        
        return list(duplicates)
    
    def _are_entities_similar(self, entity1: ResearchEntity, entity2: ResearchEntity) -> bool:
        """Check if two entities are similar enough to be considered duplicates"""
        # Content hash check
        if entity1.content_hash == entity2.content_hash:
            return True
        
        # URL similarity check
        if entity1.source_url == entity2.source_url:
            return True
        
        # Semantic similarity check
        if (entity1.semantic_embedding is not None and 
            entity2.semantic_embedding is not None):
            try:
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity(
                    entity1.semantic_embedding.reshape(1, -1),
                    entity2.semantic_embedding.reshape(1, -1)
                )[0, 0]
                
                if similarity > self.similarity_threshold:
                    return True
            except Exception:
                pass
        
        return False


class CollaborationManager:
    """
    Multi-user/agent collaboration manager with session locks and merge capabilities.
    
    Handles concurrent access to research sessions and merges contributions.
    """
    
    def __init__(self):
        self.session_locks: Dict[str, Dict[str, Any]] = {}
        self.collaboration_history: Dict[str, List[Dict[str, Any]]] = {}
        
    async def acquire_session_lock(self, session_id: str, user_id: str, operation: str) -> bool:
        """Acquire a lock on a research session for collaborative editing"""
        current_time = datetime.now(timezone.utc)
        
        if session_id in self.session_locks:
            lock_info = self.session_locks[session_id]
            
            # Check if lock is expired (5 minutes)
            if (current_time - lock_info['acquired_at']).total_seconds() > 300:
                # Lock expired, remove it
                del self.session_locks[session_id]
            else:
                # Lock still active
                if lock_info['user_id'] != user_id:
                    return False  # Someone else has the lock
        
        # Acquire the lock
        self.session_locks[session_id] = {
            'user_id': user_id,
            'operation': operation,
            'acquired_at': current_time
        }
        
        logger.info(f"Session lock acquired by {user_id} for {session_id} ({operation})")
        return True
    
    async def release_session_lock(self, session_id: str, user_id: str) -> bool:
        """Release a session lock"""
        if session_id not in self.session_locks:
            return True  # No lock to release
        
        lock_info = self.session_locks[session_id]
        if lock_info['user_id'] != user_id:
            return False  # Not the lock owner
        
        del self.session_locks[session_id]
        logger.info(f"Session lock released by {user_id} for {session_id}")
        return True
    
    def record_collaboration(self, session_id: str, user_id: str, action: str, data: Dict[str, Any]):
        """Record a collaboration action"""
        if session_id not in self.collaboration_history:
            self.collaboration_history[session_id] = []
        
        collaboration_record = {
            'user_id': user_id,
            'action': action,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        self.collaboration_history[session_id].append(collaboration_record)
        
        # Keep only last 100 collaboration records per session
        if len(self.collaboration_history[session_id]) > 100:
            self.collaboration_history[session_id] = self.collaboration_history[session_id][-100:]
    
    def merge_entity_contributions(self, base_entity: ResearchEntity, contributions: List[Dict[str, Any]]) -> ResearchEntity:
        """Merge multiple contributions to a research entity"""
        merged_entity = base_entity
        
        for contribution in contributions:
            contributor = contribution['user_id']
            changes = contribution['changes']
            
            # Merge credibility scores (weighted average)
            if 'credibility_score' in changes:
                new_score = changes['credibility_score']
                weight = contribution.get('confidence', 0.5)
                merged_entity.credibility_score = (
                    merged_entity.credibility_score * (1 - weight) + new_score * weight
                )
            
            # Merge bias scores
            if 'bias_score' in changes:
                new_score = changes['bias_score']
                weight = contribution.get('confidence', 0.5)
                merged_entity.bias_score = (
                    merged_entity.bias_score * (1 - weight) + new_score * weight
                )
            
            # Add to processing history
            if contributor not in merged_entity.processed_by:
                merged_entity.processed_by.append(contributor)
        
        merged_entity.last_updated = datetime.now(timezone.utc)
        return merged_entity
    
    def get_collaboration_summary(self, session_id: str) -> Dict[str, Any]:
        """Get collaboration summary for a session"""
        if session_id not in self.collaboration_history:
            return {'collaborators': [], 'actions': 0, 'last_activity': None}
        
        history = self.collaboration_history[session_id]
        collaborators = list(set(record['user_id'] for record in history))
        
        return {
            'collaborators': collaborators,
            'actions': len(history),
            'last_activity': history[-1]['timestamp'] if history else None,
            'current_lock': self.session_locks.get(session_id)
        }


class EnhancedResearchSession(ResearchSession):
    """
    Enhanced research session with advanced contradiction detection,
    bias analysis, entity management, and collaboration support.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Initialize advanced components
        self.contradiction_detector = ContradictionDetector(self.embedding_model)
        self.bias_detector = BiasDetector()
        self.entity_manager = EntityManager(self.embedding_model)
        self.collaboration_manager = CollaborationManager()
        
        # Memory integration
        self.memory_context: Optional[SessionMemoryContext] = None
        
        logger.info(f"Enhanced research session created: {self.session_id}")
    
    async def initialize_memory_context(self, memory_manager: MemoryManager, user_registry: Optional[UserRegistryManager] = None):
        """Initialize memory context for persistent storage"""
        if memory_manager:
            self.memory_context = SessionMemoryContext(
                session_id=self.session_id,
                user_id=self.user_id,
                memory_manager=memory_manager
            )
            await self.memory_context.initialize_context()
            logger.info(f"Memory context initialized for session {self.session_id}")
    
    async def add_entity_with_analysis(self, 
                                      content: str, 
                                      source_url: str, 
                                      source_type: str = "web_page",
                                      extraction_method: str = "browser",
                                      user_id: Optional[str] = None) -> ResearchEntity:
        """Add entity with comprehensive analysis"""
        # Add basic entity
        entity = self.add_entity(content, source_url, source_type, extraction_method)
        
        # Perform bias analysis
        bias_scores = self.bias_detector.detect_bias(entity)
        entity.bias_score = 1.0 - bias_scores['overall']  # Invert for consistency
        
        # Store bias analysis results
        self.bias_detections.append({
            'entity_id': entity.entity_id,
            'bias_scores': bias_scores,
            'detected_at': datetime.now(timezone.utc).isoformat()
        })
        
        # Update entity scores based on analysis
        self.update_entity_scores(
            entity.entity_id, 
            entity.credibility_score,
            entity.bias_score,
            entity.completeness_score
        )
        
        # Record collaboration if user specified
        if user_id and user_id != self.user_id:
            self.collaboration_manager.record_collaboration(
                self.session_id, user_id, 'add_entity', 
                {'entity_id': entity.entity_id, 'source_url': source_url}
            )
        
        # Store in memory if available
        if self.memory_context:
            await self.memory_context.store_document_memory(
                content, f"research_entity_{entity.entity_id}", "research_extraction"
            )
        
        return entity
    
    async def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """Run comprehensive analysis on all entities"""
        analysis_results = {
            'contradictions': [],
            'bias_analysis': {},
            'entity_priorities': [],
            'duplicates_removed': [],
            'collaboration_summary': {},
            'analysis_timestamp': datetime.now(timezone.utc).isoformat()
        }
        
        # Detect contradictions
        contradictions = await self.contradiction_detector.detect_contradictions(self.entities)
        for contradiction in contradictions:
            self.contradictions[contradiction.contradiction_id] = contradiction
            analysis_results['contradictions'].append(contradiction.to_dict())
        
        # Prioritize entities
        priorities = self.entity_manager.prioritize_entities(self.entities)
        analysis_results['entity_priorities'] = priorities
        
        # Remove duplicates
        duplicates = self.entity_manager.deduplicate_entities(self.entities)
        for duplicate_id in duplicates:
            if duplicate_id in self.entities:
                del self.entities[duplicate_id]
        analysis_results['duplicates_removed'] = duplicates
        
        # Collaboration summary
        analysis_results['collaboration_summary'] = self.collaboration_manager.get_collaboration_summary(self.session_id)
        
        # Update metrics
        self.metrics.update_metrics(list(self.entities.values()), list(self.contradictions.values()))
        
        # Store analysis in memory
        if self.memory_context:
            await self.memory_context.store_conversation_turn(
                f"Research analysis for query: {self.query}",
                f"Found {len(self.entities)} entities, {len(contradictions)} contradictions, removed {len(duplicates)} duplicates",
                {'analysis_results': analysis_results}
            )
        
        logger.info(f"Comprehensive analysis completed for session {self.session_id}")
        return analysis_results
    
    async def collaborate_on_entity(self, entity_id: str, user_id: str, changes: Dict[str, Any], confidence: float = 0.7) -> bool:
        """Allow collaborative editing of entity scores"""
        # Acquire lock
        lock_acquired = await self.collaboration_manager.acquire_session_lock(
            self.session_id, user_id, f"edit_entity_{entity_id}"
        )
        
        if not lock_acquired:
            return False
        
        try:
            if entity_id in self.entities:
                # Apply changes with confidence weighting
                contribution = {
                    'user_id': user_id,
                    'changes': changes,
                    'confidence': confidence,
                    'timestamp': datetime.now(timezone.utc).isoformat()
                }
                
                # Merge the contribution
                self.entities[entity_id] = self.collaboration_manager.merge_entity_contributions(
                    self.entities[entity_id], [contribution]
                )
                
                # Record collaboration
                self.collaboration_manager.record_collaboration(
                    self.session_id, user_id, 'edit_entity', 
                    {'entity_id': entity_id, 'changes': changes, 'confidence': confidence}
                )
                
                logger.info(f"Entity {entity_id} updated by {user_id} in session {self.session_id}")
                return True
        
        finally:
            # Release lock
            await self.collaboration_manager.release_session_lock(self.session_id, user_id)
        
        return False
    
    def get_enhanced_summary(self) -> Dict[str, Any]:
        """Get enhanced session summary with analysis results"""
        base_summary = self.get_session_summary()
        
        enhanced_summary = {
            **base_summary,
            'bias_detections': len(self.bias_detections),
            'collaboration_summary': self.collaboration_manager.get_collaboration_summary(self.session_id),
            'top_entities': self.entity_manager.prioritize_entities(self.entities)[:5],
            'contradiction_severity_distribution': self._get_contradiction_severity_distribution(),
            'bias_analysis_summary': self._get_bias_analysis_summary(),
            'memory_integration': self.memory_context is not None
        }
        
        return enhanced_summary
    
    def _get_contradiction_severity_distribution(self) -> Dict[str, int]:
        """Get distribution of contradiction severities"""
        distribution = {
            'minor': 0,
            'moderate': 0, 
            'major': 0,
            'critical': 0
        }
        
        for contradiction in self.contradictions.values():
            distribution[contradiction.severity.value] += 1
        
        return distribution
    
    def _get_bias_analysis_summary(self) -> Dict[str, float]:
        """Get summary of bias analysis across all entities"""
        if not self.bias_detections:
            return {}
        
        bias_types = ['confirmation_bias', 'selection_bias', 'emotional_bias', 'authority_bias', 'source_bias', 'language_bias']
        summary = {bias_type: 0.0 for bias_type in bias_types}
        
        for detection in self.bias_detections:
            bias_scores = detection['bias_scores']
            for bias_type in bias_types:
                if bias_type in bias_scores:
                    summary[bias_type] += bias_scores[bias_type]
        
        # Calculate averages
        num_detections = len(self.bias_detections)
        for bias_type in bias_types:
            summary[bias_type] = summary[bias_type] / num_detections if num_detections > 0 else 0.0
        
        return summary