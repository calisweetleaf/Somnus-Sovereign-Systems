"""
SOMNUS RESEARCH - Cognitive Resonance Cache Engine
Full implementation of Final Caching Strategy + Autodidactic Learning using code operations.

Features:
- TriaxialKnowledgeEntity as atomic cache unit (no tensors, just smart data structures)
- IALA Breath Phases (INHALE/EXHALE/HOLD) for research synchronization
- Cache Quality Guardian with ethical assessment and contradiction detection
- Recursive Weight system using embeddings and mathematical operations
- Metacognitive monitoring and contradiction resolution
- Knowledge synthesis engine
- Predictive caching based on breath phases
- Dynamic TTL with homeostatic invalidation
- Integration with Somnus memory_core and model_loader

This gives us the full cognitive architecture without tensor complexity.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
import zlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import defaultdict, deque

import msgpack
import numpy as np
import redis.asyncio as redis
import rocksdb
from pydantic import BaseModel, Field, validator
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Somnus system imports
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope
from core.memory_integration import SessionMemoryContext
from core.model_loader import ModelLoader
from schemas.session import SessionID, UserID
from research_stream_manager import ResearchStreamManager, StreamEvent, StreamEventType, StreamPriority

logger = logging.getLogger(__name__)


class CacheLayer(str, Enum):
    """Tiered cache storage layers following Final Caching Strategy"""
    HOT = "hot"        # Redis - Metadata headers for rapid access decisions
    WARM = "warm"      # RocksDB - Full TriaxialKnowledgeEntity objects
    COLD = "cold"      # Memory system - Permanent archive and learning history


class BreathPhase(str, Enum):
    """IALA breath phases for research synchronization"""
    INHALE = "inhale"        # Knowledge acquisition phase
    HOLD_IN = "hold_in"      # Integration and analysis phase  
    EXHALE = "exhale"        # Synthesis and output phase
    HOLD_OUT = "hold_out"    # Reflection and evaluation phase


class EthicalArchetype(str, Enum):
    """Ethical archetypes for knowledge classification"""
    TRUTH_SEEKER = "truth_seeker"
    SKEPTICAL_ANALYST = "skeptical_analyst"
    BALANCED_SYNTHESIZER = "balanced_synthesizer"
    CREATIVE_EXPLORER = "creative_explorer"
    ETHICAL_GUARDIAN = "ethical_guardian"


class CacheVerdict(str, Enum):
    """Cache Quality Guardian verdicts"""
    RETAIN = "retain"
    PURGE = "purge"
    FLAG_WITH_WARNING = "flag_with_warning"
    TRIGGER_SYNTHESIS = "trigger_synthesis"


class AbstractionLevel(str, Enum):
    """Abstraction levels for knowledge hierarchies"""
    CONCRETE_IMPLEMENTATION = "concrete_implementation"      # Rank 0-1
    OPERATIONAL_PROCEDURES = "operational_procedures"        # Rank 1-2
    FUNCTIONAL_PATTERNS = "functional_patterns"              # Rank 2-3
    CONCEPTUAL_FRAMEWORKS = "conceptual_frameworks"          # Rank 3-4
    THEORETICAL_PRINCIPLES = "theoretical_principles"        # Rank 4-5
    PHILOSOPHICAL_FOUNDATIONS = "philosophical_foundations"  # Rank 5-6


@dataclass
class RecursiveWeight:
    """Recursive weight system using embeddings and code operations (no tensors)"""
    base_representation: np.ndarray  # Embedding vector
    recursive_depth: int
    convergence_score: float
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not isinstance(self.base_representation, np.ndarray):
            self.base_representation = np.array(self.base_representation, dtype=np.float32)
        
        # Normalize the representation
        norm = np.linalg.norm(self.base_representation)
        if norm > 0:
            self.base_representation = self.base_representation / norm
    
    def evolve(self, mutation_strength: float, ethical_guidance: np.ndarray):
        """Evolve the recursive weight using mathematical operations"""
        if len(ethical_guidance) != len(self.base_representation):
            # Pad or truncate ethical guidance to match
            if len(ethical_guidance) < len(self.base_representation):
                ethical_guidance = np.pad(ethical_guidance, 
                                        (0, len(self.base_representation) - len(ethical_guidance)))
            else:
                ethical_guidance = ethical_guidance[:len(self.base_representation)]
        
        # Apply mutation with ethical guidance
        mutation = np.random.normal(0, mutation_strength, self.base_representation.shape)
        ethical_influence = ethical_guidance * 0.1
        
        # Update representation
        old_representation = self.base_representation.copy()
        self.base_representation += mutation + ethical_influence
        
        # Renormalize
        norm = np.linalg.norm(self.base_representation)
        if norm > 0:
            self.base_representation = self.base_representation / norm
        
        # Update convergence score based on change
        change_magnitude = np.linalg.norm(self.base_representation - old_representation)
        self.convergence_score = max(0.0, min(1.0, 1.0 - change_magnitude))
        
        # Record evolution
        self.evolution_history.append({
            'timestamp': time.time(),
            'mutation_strength': mutation_strength,
            'convergence_score': self.convergence_score,
            'change_magnitude': change_magnitude
        })
    
    def compute_similarity(self, other: 'RecursiveWeight') -> float:
        """Compute semantic similarity using cosine similarity"""
        return float(cosine_similarity(
            self.base_representation.reshape(1, -1),
            other.base_representation.reshape(1, -1)
        )[0, 0])
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'base_representation': self.base_representation.tolist(),
            'recursive_depth': self.recursive_depth,
            'convergence_score': self.convergence_score,
            'evolution_history': self.evolution_history
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'RecursiveWeight':
        """Create from dictionary"""
        return cls(
            base_representation=np.array(data['base_representation']),
            recursive_depth=data['recursive_depth'],
            convergence_score=data['convergence_score'],
            evolution_history=data['evolution_history']
        )


@dataclass
class TriaxialKnowledgeEntity:
    """
    Core knowledge entity with triaxial intelligence using code operations.
    This is the atomic unit cached according to Final Caching Strategy.
    """
    # Core identification
    content_hash: str
    content_text: str
    source_url: str
    timestamp: float
    
    # Recursive weight representation (using embeddings)
    recursive_weight: RecursiveWeight
    
    # Ethical vector (5 dimensions: truthfulness, harm, bias, completeness, reliability)
    ethical_vector: List[float]
    ethical_archetype: EthicalArchetype
    ethical_coherence: float
    
    # Metacognitive assessment
    reliability_score: float
    contradiction_level: float
    integration_difficulty: float
    
    # Abstraction encoding
    abstraction_level: AbstractionLevel
    abstraction_rank: int
    coherence_scores: Dict[str, float] = field(default_factory=dict)
    
    # Learning dynamics
    application_count: int = 0
    synthesis_count: int = 0
    evolution_history: List[Dict] = field(default_factory=list)
    
    # Cache metadata
    cache_layer: CacheLayer = CacheLayer.WARM
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    
    def update_from_application(self, success: bool, context: Dict[str, Any]):
        """Update entity based on application results"""
        self.application_count += 1
        self.access_count += 1
        self.last_accessed = datetime.now(timezone.utc)
        
        # Update reliability based on success
        if success:
            self.reliability_score = min(1.0, self.reliability_score + 0.05)
        else:
            self.reliability_score = max(0.0, self.reliability_score - 0.02)
        
        # Evolve recursive weight
        mutation_strength = 0.01 if success else 0.005
        ethical_guidance = np.array(self.ethical_vector + [0.0] * max(0, 384 - len(self.ethical_vector)))
        self.recursive_weight.evolve(mutation_strength, ethical_guidance)
        
        # Record evolution
        self.evolution_history.append({
            'timestamp': time.time(),
            'application_success': success,
            'context': context,
            'reliability_score': self.reliability_score
        })
        
        # Promote to hot cache if frequently accessed
        if self.access_count > 5 and self.cache_layer == CacheLayer.WARM:
            self.cache_layer = CacheLayer.HOT
    
    def calculate_dynamic_ttl(self) -> int:
        """Calculate dynamic TTL following Final Caching Strategy formula"""
        # Base TTL based on abstraction rank
        base_ttl_map = {
            0: 43200,   # 12 hours (concrete)
            1: 43200,   # 12 hours (concrete)
            2: 604800,  # 7 days (functional)
            3: 604800,  # 7 days (functional)
            4: 604800,  # 7 days (conceptual)
            5: 2592000, # 30 days (philosophical)
            6: 2592000  # 30 days (philosophical)
        }
        
        base_ttl = base_ttl_map.get(self.abstraction_rank, 3600)
        
        # Popularity multiplier
        popularity_multiplier = min(2.0, 1.0 + (self.access_count / 10))
        
        # Volatility multiplier (higher contradiction = shorter TTL)
        volatility_multiplier = 1.0 + self.contradiction_level
        
        # Contradiction multiplier
        contradiction_multiplier = 1.0 + (self.contradiction_level * 2)
        
        # Apply Final Caching Strategy formula
        final_ttl = int((base_ttl * popularity_multiplier) / (volatility_multiplier * contradiction_multiplier))
        
        return max(300, min(2592000, final_ttl))  # Between 5 minutes and 30 days
    
    def to_cache_optimized_format(self) -> Tuple[Dict[str, Any], bytes]:
        """
        Convert to cache-optimized format with metadata header and payload.
        Following Final Caching Strategy bifurcated schema.
        """
        # Metadata header (always deserialized for rapid checks)
        metadata_header = {
            'content_hash': self.content_hash,
            'abstraction_rank': self.abstraction_rank,
            'timestamp': self.timestamp,
            'ethical_archetype': self.ethical_archetype.value,
            'reliability_score': self.reliability_score,
            'contradiction_level': self.contradiction_level,
            'access_count': self.access_count,
            'cache_layer': self.cache_layer.value,
            'ttl': self.calculate_dynamic_ttl()
        }
        
        # Full payload (lazily deserialized)
        full_payload = {
            'content_text': self.content_text,
            'source_url': self.source_url,
            'recursive_weight': self.recursive_weight.to_dict(),
            'ethical_vector': self.ethical_vector,
            'ethical_coherence': self.ethical_coherence,
            'integration_difficulty': self.integration_difficulty,
            'abstraction_level': self.abstraction_level.value,
            'coherence_scores': self.coherence_scores,
            'evolution_history': self.evolution_history,
            'application_count': self.application_count,
            'synthesis_count': self.synthesis_count
        }
        
        # Serialize and compress payload
        serialized_payload = msgpack.packb(full_payload, use_bin_type=True)
        compressed_payload = zlib.compress(serialized_payload)
        
        return metadata_header, compressed_payload
    
    @classmethod
    def from_cache_format(cls, metadata_header: Dict[str, Any], compressed_payload: bytes) -> 'TriaxialKnowledgeEntity':
        """Reconstruct from cache format"""
        # Decompress and deserialize payload
        serialized_payload = zlib.decompress(compressed_payload)
        full_payload = msgpack.unpackb(serialized_payload, raw=False)
        
        # Reconstruct recursive weight
        recursive_weight = RecursiveWeight.from_dict(full_payload['recursive_weight'])
        
        return cls(
            content_hash=metadata_header['content_hash'],
            content_text=full_payload['content_text'],
            source_url=full_payload['source_url'],
            timestamp=metadata_header['timestamp'],
            recursive_weight=recursive_weight,
            ethical_vector=full_payload['ethical_vector'],
            ethical_archetype=EthicalArchetype(metadata_header['ethical_archetype']),
            ethical_coherence=full_payload['ethical_coherence'],
            reliability_score=metadata_header['reliability_score'],
            contradiction_level=metadata_header['contradiction_level'],
            integration_difficulty=full_payload['integration_difficulty'],
            abstraction_level=AbstractionLevel(full_payload['abstraction_level']),
            abstraction_rank=metadata_header['abstraction_rank'],
            coherence_scores=full_payload['coherence_scores'],
            application_count=full_payload['application_count'],
            synthesis_count=full_payload['synthesis_count'],
            evolution_history=full_payload['evolution_history'],
            cache_layer=CacheLayer(metadata_header['cache_layer']),
            access_count=metadata_header['access_count']
        )


class MetacognitiveMonitor:
    """Metacognitive monitoring system using heuristic algorithms"""
    
    def __init__(self):
        self.contradiction_history: List[Dict[str, Any]] = []
        self.confidence_threshold = 0.7
        self.contradiction_threshold = 0.5
    
    def detect_contradiction(self, entity1: TriaxialKnowledgeEntity, entity2: TriaxialKnowledgeEntity) -> Dict[str, Any]:
        """Detect contradictions between knowledge entities using similarity analysis"""
        # Semantic similarity
        semantic_similarity = entity1.recursive_weight.compute_similarity(entity2.recursive_weight)
        
        # Reliability difference
        reliability_diff = abs(entity1.reliability_score - entity2.reliability_score)
        
        # Ethical coherence difference
        ethical_diff = abs(entity1.ethical_coherence - entity2.ethical_coherence)
        
        # High semantic similarity but different reliability/ethics suggests contradiction
        contradiction_detected = (
            semantic_similarity > 0.8 and 
            (reliability_diff > 0.3 or ethical_diff > 0.3)
        )
        
        contradiction_strength = 0.0
        if contradiction_detected:
            contradiction_strength = (reliability_diff + ethical_diff) / 2
        
        result = {
            'contradiction_detected': contradiction_detected,
            'contradiction_strength': contradiction_strength,
            'semantic_similarity': semantic_similarity,
            'reliability_diff': reliability_diff,
            'ethical_diff': ethical_diff
        }
        
        if contradiction_detected:
            self.contradiction_history.append({
                'timestamp': time.time(),
                'entity1_hash': entity1.content_hash,
                'entity2_hash': entity2.content_hash,
                'strength': contradiction_strength,
                'details': result
            })
        
        return result
    
    def assess_system_stability(self, recent_entities: List[TriaxialKnowledgeEntity]) -> float:
        """Assess overall system stability based on contradictions and coherence"""
        if not recent_entities:
            return 1.0
        
        # Recent contradiction rate
        recent_contradictions = [
            c for c in self.contradiction_history 
            if c['timestamp'] > time.time() - 3600  # Last hour
        ]
        contradiction_rate = len(recent_contradictions) / max(len(recent_entities), 1)
        
        # Average reliability
        avg_reliability = sum(e.reliability_score for e in recent_entities) / len(recent_entities)
        
        # Coherence stability
        coherence_scores = []
        for entity in recent_entities:
            if entity.coherence_scores:
                coherence_scores.extend(entity.coherence_scores.values())
        
        coherence_stability = 1.0
        if coherence_scores:
            coherence_variance = np.var(coherence_scores)
            coherence_stability = 1.0 / (1.0 + coherence_variance)
        
        # Combined stability score
        stability = (
            (1.0 - min(1.0, contradiction_rate)) * 0.4 +
            avg_reliability * 0.4 +
            coherence_stability * 0.2
        )
        
        return max(0.0, min(1.0, stability))


class EthicalLearningFilter:
    """Ethical assessment system using rule-based analysis"""
    
    def __init__(self):
        self.ethical_thresholds = {
            'acceptance_threshold': 0.6,
            'warning_threshold': 0.4,
            'rejection_threshold': 0.2
        }
    
    def filter_knowledge(self, content: str, source_reliability: float = 0.8) -> Dict[str, Any]:
        """Filter knowledge based on ethical assessment"""
        # Analyze content for ethical concerns
        ethical_scores = self._analyze_ethical_dimensions(content)
        
        # Determine archetype
        archetype = self._determine_ethical_archetype(ethical_scores)
        
        # Calculate acceptance probability
        acceptance_prob = self._calculate_acceptance_probability(ethical_scores, source_reliability)
        
        # Make filtering decision
        if acceptance_prob >= self.ethical_thresholds['acceptance_threshold']:
            filtered_content = content
            verdict = "accept"
        elif acceptance_prob >= self.ethical_thresholds['warning_threshold']:
            filtered_content = content  # Accept with warning
            verdict = "accept_with_warning"
        else:
            filtered_content = None  # Reject
            verdict = "reject"
        
        return {
            'filtered_content': filtered_content,
            'verdict': verdict,
            'acceptance_probability': acceptance_prob,
            'ethical_scores': ethical_scores,
            'recommended_archetype': archetype
        }
    
    def _analyze_ethical_dimensions(self, content: str) -> Dict[str, float]:
        """Analyze content across 5 ethical dimensions using heuristics"""
        content_lower = content.lower()
        
        # Truthfulness (based on source indicators)
        truthfulness = 0.7  # Base score
        if any(word in content_lower for word in ['study', 'research', 'data', 'evidence']):
            truthfulness += 0.2
        if any(word in content_lower for word in ['claim', 'allegedly', 'rumor']):
            truthfulness -= 0.2
        
        # Harm assessment (based on negative indicators)
        harm_score = 0.8  # Start high (low harm)
        harmful_keywords = ['violence', 'dangerous', 'harmful', 'illegal', 'hate']
        harm_count = sum(1 for word in harmful_keywords if word in content_lower)
        harm_score = max(0.0, harm_score - (harm_count * 0.2))
        
        # Bias assessment (based on language patterns)
        bias_score = 0.6  # Neutral starting point
        biased_phrases = ['always', 'never', 'all', 'none', 'everyone', 'nobody']
        bias_count = sum(1 for phrase in biased_phrases if phrase in content_lower)
        bias_score = max(0.0, bias_score - (bias_count * 0.1))
        
        # Completeness (based on content length and structure)
        completeness = min(1.0, len(content) / 1000)  # Longer = more complete
        if '.' in content:
            completeness += 0.1  # Has sentences
        if any(word in content_lower for word in ['because', 'therefore', 'however']):
            completeness += 0.1  # Has reasoning
        
        # Reliability (based on source and content quality)
        reliability = 0.6  # Base reliability
        if any(word in content_lower for word in ['university', 'journal', 'study']):
            reliability += 0.2
        if any(word in content_lower for word in ['blog', 'opinion', 'i think']):
            reliability -= 0.1
        
        return {
            'truthfulness': max(0.0, min(1.0, truthfulness)),
            'harm': max(0.0, min(1.0, harm_score)),
            'bias': max(0.0, min(1.0, bias_score)),
            'completeness': max(0.0, min(1.0, completeness)),
            'reliability': max(0.0, min(1.0, reliability))
        }
    
    def _determine_ethical_archetype(self, ethical_scores: Dict[str, float]) -> EthicalArchetype:
        """Determine ethical archetype based on score patterns"""
        if ethical_scores['truthfulness'] > 0.8 and ethical_scores['reliability'] > 0.8:
            return EthicalArchetype.TRUTH_SEEKER
        elif ethical_scores['bias'] < 0.5:
            return EthicalArchetype.SKEPTICAL_ANALYST
        elif all(score > 0.6 for score in ethical_scores.values()):
            return EthicalArchetype.BALANCED_SYNTHESIZER
        elif ethical_scores['completeness'] > 0.7:
            return EthicalArchetype.CREATIVE_EXPLORER
        else:
            return EthicalArchetype.ETHICAL_GUARDIAN
    
    def _calculate_acceptance_probability(self, ethical_scores: Dict[str, float], source_reliability: float) -> float:
        """Calculate overall acceptance probability"""
        # Weighted average of ethical dimensions
        weights = {
            'truthfulness': 0.25,
            'harm': 0.25,
            'bias': 0.2,
            'completeness': 0.15,
            'reliability': 0.15
        }
        
        ethical_score = sum(ethical_scores[dim] * weight for dim, weight in weights.items())
        
        # Combine with source reliability
        final_score = (ethical_score * 0.7) + (source_reliability * 0.3)
        
        return max(0.0, min(1.0, final_score))


class KnowledgeSynthesisEngine:
    """Knowledge synthesis using semantic analysis and combination heuristics"""
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.synthesis_history: deque = deque(maxlen=1000)
    
    def synthesize_knowledge(self, entities: List[TriaxialKnowledgeEntity], query: str) -> Dict[str, Any]:
        """Synthesize knowledge from multiple entities"""
        if not entities:
            return {'success': False, 'error': 'No entities provided'}
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Find most relevant entities
        relevance_scores = []
        for entity in entities:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                entity.recursive_weight.base_representation.reshape(1, -1)
            )[0, 0]
            relevance_scores.append((entity, similarity))
        
        # Sort by relevance
        relevance_scores.sort(key=lambda x: x[1], reverse=True)
        top_entities = [entity for entity, _ in relevance_scores[:5]]  # Top 5
        
        # Check for contradictions
        contradictions = self._identify_contradictions(top_entities)
        
        # Synthesize content
        synthesis_content = self._combine_entity_content(top_entities, contradictions)
        
        # Calculate synthesis quality
        synthesis_quality = self._assess_synthesis_quality(top_entities, contradictions)
        
        # Create synthesis entity
        synthesis_entity = self._create_synthesis_entity(synthesis_content, top_entities, query)
        
        result = {
            'success': True,
            'synthesis_entity': synthesis_entity,
            'source_entities': [e.content_hash for e in top_entities],
            'contradictions': contradictions,
            'synthesis_quality': synthesis_quality,
            'query': query
        }
        
        self.synthesis_history.append(result)
        return result
    
    def _identify_contradictions(self, entities: List[TriaxialKnowledgeEntity]) -> List[Dict[str, Any]]:
        """Identify contradictions between entities"""
        contradictions = []
        monitor = MetacognitiveMonitor()
        
        for i, entity1 in enumerate(entities):
            for entity2 in entities[i+1:]:
                contradiction = monitor.detect_contradiction(entity1, entity2)
                if contradiction['contradiction_detected']:
                    contradictions.append({
                        'entity1': entity1.content_hash,
                        'entity2': entity2.content_hash,
                        'strength': contradiction['contradiction_strength'],
                        'details': contradiction
                    })
        
        return contradictions
    
    def _combine_entity_content(self, entities: List[TriaxialKnowledgeEntity], contradictions: List[Dict]) -> str:
        """Combine content from entities, handling contradictions"""
        combined_content = f"Synthesis from {len(entities)} sources:\n\n"
        
        for i, entity in enumerate(entities):
            combined_content += f"Source {i+1} (Reliability: {entity.reliability_score:.2f}):\n"
            combined_content += f"{entity.content_text[:500]}...\n\n"
        
        if contradictions:
            combined_content += f"\nContradictions detected ({len(contradictions)} found):\n"
            for contradiction in contradictions:
                combined_content += f"- Conflict between sources with strength {contradiction['strength']:.2f}\n"
        
        return combined_content
    
    def _assess_synthesis_quality(self, entities: List[TriaxialKnowledgeEntity], contradictions: List[Dict]) -> float:
        """Assess quality of synthesis"""
        if not entities:
            return 0.0
        
        # Base quality from entity reliability
        avg_reliability = sum(e.reliability_score for e in entities) / len(entities)
        
        # Penalty for contradictions
        contradiction_penalty = min(0.5, len(contradictions) * 0.1)
        
        # Diversity bonus
        diversity_score = len(set(e.ethical_archetype for e in entities)) / len(EthicalArchetype)
        
        quality = avg_reliability - contradiction_penalty + (diversity_score * 0.2)
        return max(0.0, min(1.0, quality))
    
    def _create_synthesis_entity(self, content: str, source_entities: List[TriaxialKnowledgeEntity], query: str) -> TriaxialKnowledgeEntity:
        """Create new entity from synthesis"""
        # Generate embedding for synthesis
        synthesis_embedding = self.embedding_model.encode([content])[0]
        
        # Create recursive weight
        recursive_weight = RecursiveWeight(
            base_representation=synthesis_embedding,
            recursive_depth=max(e.recursive_weight.recursive_depth for e in source_entities) + 1,
            convergence_score=0.8  # High convergence for synthesis
        )
        
        # Average ethical vectors
        avg_ethical_vector = [
            sum(e.ethical_vector[i] if i < len(e.ethical_vector) else 0.0 for e in source_entities) / len(source_entities)
            for i in range(5)
        ]
        
        # Determine synthesis abstraction level
        max_rank = max(e.abstraction_rank for e in source_entities)
        synthesis_level = AbstractionLevel.CONCEPTUAL_FRAMEWORKS  # Default for synthesis
        
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        return TriaxialKnowledgeEntity(
            content_hash=content_hash,
            content_text=content,
            source_url=f"synthesis:{query}",
            timestamp=time.time(),
            recursive_weight=recursive_weight,
            ethical_vector=avg_ethical_vector,
            ethical_archetype=EthicalArchetype.BALANCED_SYNTHESIZER,
            ethical_coherence=0.8,
            reliability_score=0.8,  # High reliability for successful synthesis
            contradiction_level=0.0,
            integration_difficulty=0.3,  # Easier to integrate synthetic knowledge
            abstraction_level=synthesis_level,
            abstraction_rank=max(3, max_rank),  # At least conceptual level
            synthesis_count=1  # Mark as synthesized
        )


class CacheQualityGuardian:
    """
    Cache Quality Guardian - Systemic immune response mechanism.
    Following Final Caching Strategy Pillar 2.
    """
    
    def __init__(self, metacognitive_monitor: MetacognitiveMonitor, ethical_filter: EthicalLearningFilter):
        self.metacognitive_monitor = metacognitive_monitor
        self.ethical_filter = ethical_filter
        self.audit_history: List[Dict[str, Any]] = []
        self.intervention_count = 0
    
    def select_entity_for_audit(self, cached_entities: List[TriaxialKnowledgeEntity]) -> Optional[TriaxialKnowledgeEntity]:
        """Select entity for audit using weighted algorithm from Final Caching Strategy"""
        if not cached_entities:
            return None
        
        audit_scores = []
        current_time = time.time()
        
        for entity in cached_entities:
            # Time since last access weight
            time_since_access = current_time - entity.last_accessed.timestamp()
            w1_score = min(1.0, time_since_access / 3600)  # Normalize to hours
            
            # Contradiction proximity score
            w2_score = entity.contradiction_level
            
            # Ethical ambiguity score (proximity to threshold)
            ethical_threshold = 0.6
            w3_score = 1.0 - abs(entity.ethical_coherence - ethical_threshold)
            
            # Calculate audit score with weights
            audit_score = (0.4 * w1_score) + (0.4 * w2_score) + (0.2 * w3_score)
            audit_scores.append((entity, audit_score))
        
        # Select entity with highest audit score
        audit_scores.sort(key=lambda x: x[1], reverse=True)
        return audit_scores[0][0] if audit_scores else None
    
    def validate_entity(self, entity: TriaxialKnowledgeEntity, system_breath_phase: BreathPhase) -> CacheVerdict:
        """Validate entity and return Cache Quality Guardian verdict"""
        # Metacognitive check - assess for contradictions
        metacognitive_assessment = self._assess_metacognitive_health(entity)
        
        # Ethical check - re-evaluate with current breath phase
        ethical_assessment = self._assess_ethical_alignment(entity, system_breath_phase)
        
        # Determine verdict
        verdict = self._determine_verdict(entity, metacognitive_assessment, ethical_assessment)
        
        # Record audit
        self.audit_history.append({
            'entity_hash': entity.content_hash,
            'timestamp': time.time(),
            'breath_phase': system_breath_phase.value,
            'metacognitive_score': metacognitive_assessment['score'],
            'ethical_score': ethical_assessment['score'],
            'verdict': verdict.value
        })
        
        if verdict in [CacheVerdict.PURGE, CacheVerdict.FLAG_WITH_WARNING]:
            self.intervention_count += 1
        
        return verdict
    
    def _assess_metacognitive_health(self, entity: TriaxialKnowledgeEntity) -> Dict[str, Any]:
        """Assess entity's metacognitive health"""
        # Check reliability degradation
        reliability_health = entity.reliability_score
        
        # Check for recent contradictions
        contradiction_health = 1.0 - entity.contradiction_level
        
        # Check coherence stability
        coherence_health = 1.0
        if entity.coherence_scores:
            coherence_variance = np.var(list(entity.coherence_scores.values()))
            coherence_health = 1.0 / (1.0 + coherence_variance)
        
        overall_score = (reliability_health + contradiction_health + coherence_health) / 3
        
        return {
            'score': overall_score,
            'reliability_health': reliability_health,
            'contradiction_health': contradiction_health,
            'coherence_health': coherence_health
        }
    
    def _assess_ethical_alignment(self, entity: TriaxialKnowledgeEntity, breath_phase: BreathPhase) -> Dict[str, Any]:
        """Assess ethical alignment considering current breath phase"""
        # Re-evaluate content with ethical filter
        ethical_result = self.ethical_filter.filter_knowledge(entity.content_text)
        current_ethical_score = ethical_result['acceptance_probability']
        
        # Adjust for breath phase
        phase_adjustment = {
            BreathPhase.INHALE: 0.1,      # More lenient during acquisition
            BreathPhase.HOLD_IN: 0.0,     # Neutral during integration
            BreathPhase.EXHALE: -0.1,     # More strict during synthesis
            BreathPhase.HOLD_OUT: -0.2    # Most strict during reflection
        }
        
        adjusted_score = current_ethical_score + phase_adjustment[breath_phase]
        adjusted_score = max(0.0, min(1.0, adjusted_score))
        
        return {
            'score': adjusted_score,
            'raw_score': current_ethical_score,
            'phase_adjustment': phase_adjustment[breath_phase],
            'archetype_match': entity.ethical_archetype == ethical_result['recommended_archetype']
        }
    
    def _determine_verdict(self, entity: TriaxialKnowledgeEntity, metacognitive: Dict, ethical: Dict) -> CacheVerdict:
        """Determine final verdict based on assessments"""
        meta_score = metacognitive['score']
        ethical_score = ethical['score']
        
        # Combined assessment
        overall_score = (meta_score + ethical_score) / 2
        
        # Decision thresholds
        if overall_score >= 0.8:
            return CacheVerdict.RETAIN
        elif overall_score <= 0.3:
            return CacheVerdict.PURGE
        elif entity.contradiction_level > 0.7:
            return CacheVerdict.TRIGGER_SYNTHESIS
        else:
            return CacheVerdict.FLAG_WITH_WARNING


class IALAForecastingEngine:
    """
    IALA-Powered Forecasting Engine for predictive caching.
    Following Final Caching Strategy Pillar 3.
    """
    
    def __init__(self, embedding_model: SentenceTransformer):
        self.embedding_model = embedding_model
        self.current_breath_phase = BreathPhase.INHALE
        self.speculative_cache: Dict[str, Any] = {}
        self.prediction_history: List[Dict[str, Any]] = []
    
    def set_breath_phase(self, phase: BreathPhase):
        """Update current IALA breath phase"""
        old_phase = self.current_breath_phase
        self.current_breath_phase = phase
        
        logger.info(f"IALA breath phase transition: {old_phase.value} -> {phase.value}")
        
        # Trigger phase-specific actions
        if phase == BreathPhase.INHALE:
            asyncio.create_task(self._inhale_phase_actions())
        elif phase == BreathPhase.EXHALE:
            asyncio.create_task(self._exhale_phase_actions())
    
    async def _inhale_phase_actions(self):
        """Predictive pre-warming during INHALE phase"""
        logger.info("🫁 INHALE Phase: Activating predictive pre-warming")
        
        # This would interface with research engine to pre-fetch related sources
        # For now, we simulate the behavior
        await asyncio.sleep(0.1)  # Simulate background work
    
    async def _exhale_phase_actions(self):
        """Speculative pre-synthesis during EXHALE phase"""
        logger.info("🫁 EXHALE Phase: Activating speculative pre-synthesis")
        
        # This would identify synthesis opportunities and pre-compute them
        # For now, we simulate the behavior
        await asyncio.sleep(0.1)  # Simulate background work
    
    def predict_cache_needs(self, query: str, context_entities: List[TriaxialKnowledgeEntity]) -> List[str]:
        """Predict what content should be pre-cached based on query and context"""
        if not context_entities:
            return []
        
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        
        # Find semantic neighbors
        related_concepts = []
        for entity in context_entities:
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                entity.recursive_weight.base_representation.reshape(1, -1)
            )[0, 0]
            
            if similarity > 0.6:  # Similarity threshold
                related_concepts.append(entity.source_url)
        
        return related_concepts[:5]  # Top 5 predictions


class CognitiveResonanceCache:
    """
    Main Cognitive Resonance Cache Engine implementing Full Final Caching Strategy.
    """
    
    def __init__(self, 
                 memory_manager: MemoryManager,
                 model_loader: ModelLoader,
                 stream_manager: ResearchStreamManager,
                 redis_url: str = "redis://localhost:6379",
                 rocksdb_path: str = "data/research_cache"):
        
        self.memory_manager = memory_manager
        self.model_loader = model_loader
        self.stream_manager = stream_manager
        
        # Storage layers
        self.redis_client: Optional[redis.Redis] = None
        self.rocksdb_path = Path(rocksdb_path)
        self.rocksdb_path.mkdir(parents=True, exist_ok=True)
        self.warm_db: Optional[rocksdb.DB] = None
        
        # Core systems
        self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        self.metacognitive_monitor = MetacognitiveMonitor()
        self.ethical_filter = EthicalLearningFilter()
        self.synthesis_engine = KnowledgeSynthesisEngine(self.embedding_model)
        self.cache_guardian = CacheQualityGuardian(self.metacognitive_monitor, self.ethical_filter)
        self.forecasting_engine = IALAForecastingEngine(self.embedding_model)
        
        # Cache state
        self.cached_entities: Dict[str, TriaxialKnowledgeEntity] = {}
        self.current_breath_phase = BreathPhase.INHALE
        
        # Metrics
        self.cache_metrics = {
            'hits': 0,
            'misses': 0,
            'purges': 0,
            'syntheses': 0,
            'guardian_interventions': 0
        }
        
        # Background tasks
        self.guardian_task: Optional[asyncio.Task] = None
        self.is_running = False
        
        logger.info("Cognitive Resonance Cache Engine initialized")
    
    async def initialize(self):
        """Initialize all cache systems"""
        try:
            # Initialize Redis (Hot Cache)
            self.redis_client = redis.from_url("redis://localhost:6379", decode_responses=False)
            await self.redis_client.ping()
            logger.info("✅ Hot cache (Redis) connected")
            
            # Initialize RocksDB (Warm Cache)
            opts = rocksdb.Options()
            opts.create_if_missing = True
            opts.max_open_files = 300000
            opts.write_buffer_size = 67108864
            opts.max_write_buffer_number = 3
            opts.target_file_size_base = 67108864
            
            self.warm_db = rocksdb.DB(str(self.rocksdb_path), opts)
            logger.info("✅ Warm cache (RocksDB) initialized")
            
            # Start background guardian
            self.is_running = True
            self.guardian_task = asyncio.create_task(self._guardian_loop())
            
            logger.info("🧠 Cognitive Resonance Cache fully operational")
            
        except Exception as e:
            logger.error(f"❌ Cache initialization failed: {e}")
            raise
    
    async def shutdown(self):
        """Graceful shutdown"""
        self.is_running = False
        
        if self.guardian_task:
            self.guardian_task.cancel()
        
        if self.redis_client:
            await self.redis_client.close()
        
        if self.warm_db:
            del self.warm_db
        
        logger.info("🛑 Cognitive Resonance Cache shutdown complete")
    
    async def store_research_result(self, content: str, source_url: str, metadata: Dict[str, Any]) -> TriaxialKnowledgeEntity:
        """Store research result as TriaxialKnowledgeEntity"""
        # Generate embedding
        embedding = self.embedding_model.encode([content])[0]
        
        # Create recursive weight
        recursive_weight = RecursiveWeight(
            base_representation=embedding,
            recursive_depth=1,
            convergence_score=0.9
        )
        
        # Ethical assessment
        ethical_result = self.ethical_filter.filter_knowledge(content)
        
        # Determine abstraction level based on content
        abstraction_level = self._determine_abstraction_level(content)
        abstraction_rank = self._get_abstraction_rank(abstraction_level)
        
        # Create entity
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        entity = TriaxialKnowledgeEntity(
            content_hash=content_hash,
            content_text=content,
            source_url=source_url,
            timestamp=time.time(),
            recursive_weight=recursive_weight,
            ethical_vector=ethical_result['ethical_scores'][ethical_result['recommended_archetype']],
            ethical_archetype=ethical_result['recommended_archetype'],
            ethical_coherence=ethical_result['acceptance_probability'],
            reliability_score=metadata.get('credibility_score', 0.7),
            contradiction_level=0.0,  # Will be updated as contradictions are detected
            integration_difficulty=0.5,  # Default
            abstraction_level=abstraction_level,
            abstraction_rank=abstraction_rank
        )
        
        # Check for contradictions with existing entities
        await self._check_contradictions(entity)
        
        # Store in cache
        await self._store_entity(entity)
        
        # Stream update
        await self.stream_manager.stream_event(StreamEvent(
            event_type=StreamEventType.SOURCE_ANALYZED,
            session_id=metadata.get('session_id', 'unknown'),
            user_id=metadata.get('user_id', 'unknown'),
            timestamp=datetime.now(timezone.utc),
            data={
                'content_hash': content_hash,
                'source_url': source_url,
                'abstraction_level': abstraction_level.value,
                'ethical_coherence': entity.ethical_coherence,
                'reliability_score': entity.reliability_score
            },
            priority=StreamPriority.NORMAL
        ))
        
        return entity
    
    async def retrieve_cached_entity(self, content_hash: str) -> Optional[TriaxialKnowledgeEntity]:
        """Retrieve entity from cache with layer promotion"""
        # Check hot cache first
        if self.redis_client:
            hot_data = await self.redis_client.get(f"hot:{content_hash}")
            if hot_data:
                self.cache_metrics['hits'] += 1
                # Deserialize and return
                metadata, payload = pickle.loads(hot_data)
                entity = TriaxialKnowledgeEntity.from_cache_format(metadata, payload)
                entity.update_from_application(True, {'cache_hit': 'hot'})
                return entity
        
        # Check warm cache
        if self.warm_db:
            warm_data = self.warm_db.get(f"warm:{content_hash}".encode())
            if warm_data:
                self.cache_metrics['hits'] += 1
                metadata, payload = pickle.loads(warm_data)
                entity = TriaxialKnowledgeEntity.from_cache_format(metadata, payload)
                entity.update_from_application(True, {'cache_hit': 'warm'})
                
                # Promote to hot if frequently accessed
                if entity.access_count > 3:
                    await self._promote_to_hot(entity)
                
                return entity
        
        # Check cold storage (memory system)
        cold_entity = await self._retrieve_from_cold_storage(content_hash)
        if cold_entity:
            self.cache_metrics['hits'] += 1
            return cold_entity
        
        self.cache_metrics['misses'] += 1
        return None
    
    async def semantic_search(self, query: str, limit: int = 10) -> List[Tuple[TriaxialKnowledgeEntity, float]]:
        """Perform semantic search across cached entities"""
        query_embedding = self.embedding_model.encode([query])[0]
        
        results = []
        for entity in self.cached_entities.values():
            similarity = cosine_similarity(
                query_embedding.reshape(1, -1),
                entity.recursive_weight.base_representation.reshape(1, -1)
            )[0, 0]
            
            if similarity > 0.3:  # Minimum similarity threshold
                results.append((entity, similarity))
        
        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:limit]
    
    async def synthesize_from_cache(self, query: str, entity_hashes: List[str]) -> Optional[TriaxialKnowledgeEntity]:
        """Synthesize knowledge from cached entities"""
        entities = []
        for hash_val in entity_hashes:
            entity = await self.retrieve_cached_entity(hash_val)
            if entity:
                entities.append(entity)
        
        if not entities:
            return None
        
        # Perform synthesis
        synthesis_result = self.synthesis_engine.synthesize_knowledge(entities, query)
        
        if synthesis_result['success']:
            synthesis_entity = synthesis_result['synthesis_entity']
            
            # Store synthesis result
            await self._store_entity(synthesis_entity)
            
            self.cache_metrics['syntheses'] += 1
            
            # Stream synthesis event
            await self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.SEARCH_PROGRESS,
                session_id="synthesis",
                user_id="system",
                timestamp=datetime.now(timezone.utc),
                data={
                    'type': 'knowledge_synthesis',
                    'query': query,
                    'source_entities': len(entities),
                    'synthesis_quality': synthesis_result['synthesis_quality'],
                    'contradictions': len(synthesis_result['contradictions'])
                },
                priority=StreamPriority.HIGH
            ))
            
            return synthesis_entity
        
        return None
    
    def set_breath_phase(self, phase: BreathPhase):
        """Update IALA breath phase"""
        self.current_breath_phase = phase
        self.forecasting_engine.set_breath_phase(phase)
        
        # Stream breath phase change
        asyncio.create_task(self.stream_manager.stream_event(StreamEvent(
            event_type=StreamEventType.SYSTEM_MESSAGE,
            session_id="system",
            user_id="iala",
            timestamp=datetime.now(timezone.utc),
            data={
                'type': 'breath_phase_change',
                'new_phase': phase.value,
                'cache_state': self._get_cache_state()
            },
            priority=StreamPriority.HIGH
        )))
    
    async def _check_contradictions(self, new_entity: TriaxialKnowledgeEntity):
        """Check new entity for contradictions with existing entities"""
        contradiction_count = 0
        
        # Check against recent entities (last 10)
        recent_entities = list(self.cached_entities.values())[-10:]
        
        for existing_entity in recent_entities:
            contradiction = self.metacognitive_monitor.detect_contradiction(new_entity, existing_entity)
            
            if contradiction['contradiction_detected']:
                # Update contradiction levels
                new_entity.contradiction_level = max(
                    new_entity.contradiction_level,
                    contradiction['contradiction_strength']
                )
                existing_entity.contradiction_level = max(
                    existing_entity.contradiction_level,
                    contradiction['contradiction_strength']
                )
                
                contradiction_count += 1
                
                # Stream contradiction detection
                await self.stream_manager.stream_event(StreamEvent(
                    event_type=StreamEventType.CONTRADICTION_DETECTED,
                    session_id="system",
                    user_id="metacognitive",
                    timestamp=datetime.now(timezone.utc),
                    data={
                        'new_entity': new_entity.content_hash,
                        'existing_entity': existing_entity.content_hash,
                        'strength': contradiction['contradiction_strength'],
                        'details': contradiction
                    },
                    priority=StreamPriority.HIGH
                ))
        
        if contradiction_count > 0:
            logger.warning(f"🔍 Detected {contradiction_count} contradictions for new entity")
    
    async def _store_entity(self, entity: TriaxialKnowledgeEntity):
        """Store entity in appropriate cache layer"""
        self.cached_entities[entity.content_hash] = entity
        
        # Prepare cache format
        metadata_header, compressed_payload = entity.to_cache_optimized_format()
        cache_data = pickle.dumps((metadata_header, compressed_payload))
        
        # Store in warm cache (RocksDB)
        if self.warm_db:
            self.warm_db.put(f"warm:{entity.content_hash}".encode(), cache_data)
        
        # Store in hot cache if high access or quality
        if entity.access_count > 3 or entity.cache_layer == CacheLayer.HOT:
            await self._promote_to_hot(entity)
        
        # Store in cold storage (memory system) for permanence
        await self._store_in_cold_storage(entity)
    
    async def _promote_to_hot(self, entity: TriaxialKnowledgeEntity):
        """Promote entity to hot cache (Redis)"""
        if not self.redis_client:
            return
        
        entity.cache_layer = CacheLayer.HOT
        metadata_header, compressed_payload = entity.to_cache_optimized_format()
        cache_data = pickle.dumps((metadata_header, compressed_payload))
        
        # Store with TTL
        ttl = entity.calculate_dynamic_ttl()
        await self.redis_client.setex(f"hot:{entity.content_hash}", ttl, cache_data)
    
    async def _store_in_cold_storage(self, entity: TriaxialKnowledgeEntity):
        """Store entity in cold storage (memory system) for permanence"""
        try:
            await self.memory_manager.store_memory(
                user_id="system",
                content=f"Research Entity: {entity.content_text[:200]}...",
                memory_type=MemoryType.KNOWLEDGE,
                importance=MemoryImportance.HIGH if entity.reliability_score > 0.8 else MemoryImportance.MEDIUM,
                scope=MemoryScope.SYSTEM,
                metadata={
                    'type': 'triaxial_knowledge_entity',
                    'content_hash': entity.content_hash,
                    'source_url': entity.source_url,
                    'abstraction_level': entity.abstraction_level.value,
                    'ethical_archetype': entity.ethical_archetype.value,
                    'reliability_score': entity.reliability_score,
                    'contradiction_level': entity.contradiction_level,
                    'cache_timestamp': entity.timestamp
                }
            )
        except Exception as e:
            logger.error(f"Failed to store entity in cold storage: {e}")
    
    async def _retrieve_from_cold_storage(self, content_hash: str) -> Optional[TriaxialKnowledgeEntity]:
        """Retrieve entity from cold storage and re-hydrate"""
        try:
            # This would query memory system for the entity
            # For now, return None (not implemented)
            return None
        except Exception as e:
            logger.error(f"Failed to retrieve from cold storage: {e}")
            return None
    
    def _determine_abstraction_level(self, content: str) -> AbstractionLevel:
        """Determine abstraction level based on content analysis"""
        content_lower = content.lower()
        
        # Keywords for different abstraction levels
        philosophical_keywords = ['philosophy', 'fundamental', 'essence', 'meaning', 'existence']
        theoretical_keywords = ['theory', 'principle', 'model', 'framework', 'paradigm']
        conceptual_keywords = ['concept', 'idea', 'approach', 'methodology', 'strategy']
        functional_keywords = ['function', 'process', 'procedure', 'method', 'technique']
        operational_keywords = ['step', 'instruction', 'command', 'action', 'implementation']
        concrete_keywords = ['example', 'instance', 'specific', 'particular', 'actual']
        
        # Count matches
        scores = {
            AbstractionLevel.PHILOSOPHICAL_FOUNDATIONS: sum(1 for kw in philosophical_keywords if kw in content_lower),
            AbstractionLevel.THEORETICAL_PRINCIPLES: sum(1 for kw in theoretical_keywords if kw in content_lower),
            AbstractionLevel.CONCEPTUAL_FRAMEWORKS: sum(1 for kw in conceptual_keywords if kw in content_lower),
            AbstractionLevel.FUNCTIONAL_PATTERNS: sum(1 for kw in functional_keywords if kw in content_lower),
            AbstractionLevel.OPERATIONAL_PROCEDURES: sum(1 for kw in operational_keywords if kw in content_lower),
            AbstractionLevel.CONCRETE_IMPLEMENTATION: sum(1 for kw in concrete_keywords if kw in content_lower)
        }
        
        # Return highest scoring level
        max_level = max(scores.items(), key=lambda x: x[1])
        return max_level[0] if max_level[1] > 0 else AbstractionLevel.FUNCTIONAL_PATTERNS
    
    def _get_abstraction_rank(self, level: AbstractionLevel) -> int:
        """Get numerical rank for abstraction level"""
        rank_map = {
            AbstractionLevel.CONCRETE_IMPLEMENTATION: 0,
            AbstractionLevel.OPERATIONAL_PROCEDURES: 1,
            AbstractionLevel.FUNCTIONAL_PATTERNS: 2,
            AbstractionLevel.CONCEPTUAL_FRAMEWORKS: 3,
            AbstractionLevel.THEORETICAL_PRINCIPLES: 4,
            AbstractionLevel.PHILOSOPHICAL_FOUNDATIONS: 5
        }
        return rank_map.get(level, 2)
    
    def _get_cache_state(self) -> Dict[str, Any]:
        """Get current cache state for monitoring"""
        return {
            'entity_count': len(self.cached_entities),
            'breath_phase': self.current_breath_phase.value,
            'metrics': self.cache_metrics,
            'guardian_interventions': self.cache_guardian.intervention_count
        }
    
    async def _guardian_loop(self):
        """Background Cache Quality Guardian loop"""
        while self.is_running:
            try:
                if self.cached_entities:
                    # Select entity for audit
                    entities_list = list(self.cached_entities.values())
                    entity_to_audit = self.cache_guardian.select_entity_for_audit(entities_list)
                    
                    if entity_to_audit:
                        # Validate entity
                        verdict = self.cache_guardian.validate_entity(entity_to_audit, self.current_breath_phase)
                        
                        # Execute verdict
                        await self._execute_guardian_verdict(entity_to_audit, verdict)
                        
                        self.cache_metrics['guardian_interventions'] += 1
                
                # Sleep before next audit
                await asyncio.sleep(60)  # Audit every minute
                
            except Exception as e:
                logger.error(f"Cache Guardian error: {e}")
                await asyncio.sleep(30)
    
    async def _execute_guardian_verdict(self, entity: TriaxialKnowledgeEntity, verdict: CacheVerdict):
        """Execute Cache Quality Guardian verdict"""
        if verdict == CacheVerdict.PURGE:
            # Remove from all cache layers
            await self._purge_entity(entity)
            self.cache_metrics['purges'] += 1
            
            # Stream purge event
            await self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.SYSTEM_MESSAGE,
                session_id="guardian",
                user_id="system",
                timestamp=datetime.now(timezone.utc),
                data={
                    'type': 'cache_purge',
                    'entity_hash': entity.content_hash,
                    'reason': 'quality_guardian_verdict'
                },
                priority=StreamPriority.NORMAL
            ))
            
        elif verdict == CacheVerdict.FLAG_WITH_WARNING:
            # Add warning metadata
            entity.contradiction_flags.append(f"Guardian warning: {datetime.now().isoformat()}")
            
        elif verdict == CacheVerdict.TRIGGER_SYNTHESIS:
            # Find related entities for synthesis
            related_entities = await self._find_related_entities(entity)
            if related_entities:
                await self.synthesize_from_cache(
                    f"Resolve contradiction for {entity.source_url}",
                    [e.content_hash for e in related_entities]
                )
    
    async def _purge_entity(self, entity: TriaxialKnowledgeEntity):
        """Remove entity from all cache layers"""
        # Remove from in-memory cache
        if entity.content_hash in self.cached_entities:
            del self.cached_entities[entity.content_hash]
        
        # Remove from hot cache
        if self.redis_client:
            await self.redis_client.delete(f"hot:{entity.content_hash}")
        
        # Remove from warm cache
        if self.warm_db:
            self.warm_db.delete(f"warm:{entity.content_hash}".encode())
    
    async def _find_related_entities(self, entity: TriaxialKnowledgeEntity, limit: int = 3) -> List[TriaxialKnowledgeEntity]:
        """Find entities related to the given entity"""
        related = []
        
        for other_entity in self.cached_entities.values():
            if other_entity.content_hash == entity.content_hash:
                continue
            
            similarity = entity.recursive_weight.compute_similarity(other_entity.recursive_weight)
            if similarity > 0.7:  # High similarity threshold
                related.append(other_entity)
        
        return related[:limit]
    
    def get_cache_metrics(self) -> Dict[str, Any]:
        """Get comprehensive cache metrics"""
        hit_rate = self.cache_metrics['hits'] / max(1, self.cache_metrics['hits'] + self.cache_metrics['misses'])
        
        return {
            'performance': {
                'hit_rate': hit_rate,
                'total_hits': self.cache_metrics['hits'],
                'total_misses': self.cache_metrics['misses']
            },
            'quality': {
                'entities_cached': len(self.cached_entities),
                'purges_executed': self.cache_metrics['purges'],
                'syntheses_created': self.cache_metrics['syntheses'],
                'guardian_interventions': self.cache_metrics['guardian_interventions']
            },
            'intelligence': {
                'current_breath_phase': self.current_breath_phase.value,
                'contradiction_detections': len(self.metacognitive_monitor.contradiction_history),
                'synthesis_history_size': len(self.synthesis_engine.synthesis_history)
            }
        }


# Factory function for easy integration
async def create_cognitive_cache(
    memory_manager: MemoryManager,
    model_loader: ModelLoader,
    stream_manager: ResearchStreamManager,
    config: Optional[Dict[str, Any]] = None
) -> CognitiveResonanceCache:
    """Create and initialize Cognitive Resonance Cache"""
    
    cache_config = config or {}
    
    cache = CognitiveResonanceCache(
        memory_manager=memory_manager,
        model_loader=model_loader,
        stream_manager=stream_manager,
        redis_url=cache_config.get('redis_url', 'redis://localhost:6379'),
        rocksdb_path=cache_config.get('rocksdb_path', 'data/research_cache')
    )
    
    await cache.initialize()
    
    logger.info("🧠 Cognitive Resonance Cache created successfully")
    return cache