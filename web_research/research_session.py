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
from core.memory_integration import SessionMemoryContext
from schemas.session import SessionID, UserID
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
                memory_type=MemoryType.KNOWLEDGE,
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
                        memory_type=MemoryType.KNOWLEDGE,
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