"""
SOMNUS RESEARCH - Research Execution Engine
The cognitive research orchestrator that turns strategy into insight.

This is the central intelligence that:
- Executes ResearchPlan steps from DeepResearchPlanner
- Coordinates between memory, browser, collaboration, and intelligence subsystems
- Manages real-time streaming and user interventions
- Performs adaptive planning based on discovered knowledge
- Integrates with Somnus memory system for persistent research knowledge

Architecture:
- Dependency injection for clean modularity and testing
- State machine for different research step types
- Asynchronous execution with graceful error handling
- Real-time UI streaming via ResearchStreamManager
- Dynamic plan adaptation based on intelligence assessment
"""

import asyncio
import hashlib
import json
import logging
import time
import traceback
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union, AsyncGenerator
from uuid import UUID, uuid4
from dataclasses import dataclass
from enum import Enum
from contextlib import asynccontextmanager

import numpy as np
from pydantic import BaseModel, Field

# Somnus system imports
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope
from backend.memory_integration import SessionMemoryContext
from core.session_manager import SessionID, UserID

# Research subsystem imports
from research_session import (
    ResearchSession, ResearchStatus, ResearchDepthLevel, 
    ResearchEntity, ResearchContradiction, EthicalConcern
)
from deep_research_planner import (
    DeepResearchPlanner, ResearchPlan, ResearchComplexity, 
    CollaborationMode, ResearchPhase
)
from research_cache_engine import (
    CognitiveResonanceCache, TriaxialKnowledgeEntity,
    BreathPhase, CacheQualityGuardian
)
from research_stream_manager import (
    ResearchStreamManager, StreamEvent, StreamEventType, StreamPriority
)

# Core system integrations
from core.agent_collaboration_core import AgentCollaborationHub

logger = logging.getLogger(__name__)

# Placeholder classes for missing components that will be implemented later
@dataclass
class ResearchStep:
    step_id: str
    step_type: str
    goal: str
    parameters: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'step_type': self.step_type,
            'goal': self.goal,
            'parameters': self.parameters
        }

@dataclass
class BrowserTask:
    task_id: str
    task_type: str
    objective: str
    queries: List[str]
    max_sources: int
    depth_level: str
    data_extraction_requirements: List[str]
    quality_filters: Dict[str, Any]

@dataclass
class BrowserResult:
    sources: List[Dict[str, Any]]
    success: bool = True

@dataclass
class AgentTask:
    task_id: str
    task_type: str
    objective: str
    input_data: Dict[str, Any]
    required_agents: List[str]
    quality_requirements: Dict[str, Any]

@dataclass
class CollaborationResult:
    success: bool
    result: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    error: str = ""

@dataclass
class QualityAssessment:
    credibility_score: float
    bias_score: float
    completeness_score: float
    ethical_flags: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'credibility_score': self.credibility_score,
            'bias_score': self.bias_score,
            'completeness_score': self.completeness_score,
            'ethical_flags': self.ethical_flags
        }

class StepType(str, Enum):
    PRELIMINARY_MEMORY_SEARCH = "preliminary_memory_search"
    EXPLORATORY_SEARCH = "exploratory_search"
    DEEP_DIVE = "deep_dive"
    SYNTHESIS = "synthesis"
    VALIDATION_AND_CONTRADICTION_RESOLUTION = "validation_and_contradiction_resolution"
    REPORT_GENERATION = "report_generation"

class BrowserResearchAgent:
    async def execute_task(self, task: BrowserTask) -> BrowserResult:
        # Placeholder implementation - returns mock data
        return BrowserResult(sources=[])

class ResearchIntelligenceEngine:
    async def assess_source_quality(self, content: str, url: str, metadata: Dict[str, Any]) -> QualityAssessment:
        # Placeholder implementation
        return QualityAssessment(
            credibility_score=0.7,
            bias_score=0.6,
            completeness_score=0.8
        )
    
    async def detect_contradictions(self, entities: List[Any]) -> Any:
        # Placeholder - returns empty contradictions
        class ContradictionAnalysis:
            def __init__(self):
                self.contradictions = []
        return ContradictionAnalysis()
    
    async def assess_session_state(self, session: Any) -> Any:
        # Placeholder intelligence metrics
        class IntelligenceMetrics:
            def to_dict(self):
                return {'knowledge_coverage': 0.7, 'contradiction_level': 0.2}
        return IntelligenceMetrics()
    
    def is_plan_adaptation_needed(self, metrics: Any) -> bool:
        return False

class ReportExporter:
    async def generate_artifact(self, research_data: Dict[str, Any], export_format: Any, session_id: str, user_id: str) -> Any:
        # Placeholder artifact reference
        class ArtifactReference:
            def __init__(self):
                self.artifact_id = str(uuid4())
        return ArtifactReference()

class ExportFormat:
    def __init__(self, format_name: str):
        self.format_name = format_name

# Add missing method to AgentCollaborationHub for compatibility
def _add_execute_task_method():
    async def execute_task(self, task: AgentTask) -> CollaborationResult:
        # Placeholder implementation
        return CollaborationResult(
            success=True,
            result={'synthesis_text': f'Synthesized result for {task.objective}'},
            confidence=0.8
        )
    
    # Monkey patch the method
    AgentCollaborationHub.execute_task = execute_task

_add_execute_task_method()


class ExecutionState(str, Enum):
    """Research execution engine states"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    PAUSED = "paused"
    WAITING_USER_INPUT = "waiting_user_input"
    ADAPTING_PLAN = "adapting_plan"
    ERROR = "error"
    COMPLETED = "completed"


class InterventionType(str, Enum):
    """Types of user interventions during research"""
    PAUSE = "pause"
    RESUME = "resume"
    SKIP_STEP = "skip_step"
    MODIFY_PLAN = "modify_plan"
    ADD_SOURCE = "add_source"
    BLOCK_SOURCE = "block_source"
    CHANGE_PRIORITY = "change_priority"
    MANUAL_SYNTHESIS = "manual_synthesis"
    STOP = "stop"


@dataclass
class ExecutionMetrics:
    """Real-time execution metrics for monitoring and optimization"""
    steps_completed: int = 0
    steps_skipped: int = 0
    steps_failed: int = 0
    total_execution_time: float = 0.0
    memory_queries_executed: int = 0
    browser_tasks_executed: int = 0
    synthesis_operations: int = 0
    contradictions_detected: int = 0
    contradictions_resolved: int = 0
    plan_adaptations: int = 0
    user_interventions: int = 0
    knowledge_entities_processed: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'steps_completed': self.steps_completed,
            'steps_skipped': self.steps_skipped,
            'steps_failed': self.steps_failed,
            'total_execution_time': self.total_execution_time,
            'memory_queries_executed': self.memory_queries_executed,
            'browser_tasks_executed': self.browser_tasks_executed,
            'synthesis_operations': self.synthesis_operations,
            'contradictions_detected': self.contradictions_detected,
            'contradictions_resolved': self.contradictions_resolved,
            'plan_adaptations': self.plan_adaptations,
            'user_interventions': self.user_interventions,
            'knowledge_entities_processed': self.knowledge_entities_processed,
            'efficiency_score': self._calculate_efficiency_score()
        }
    
    def _calculate_efficiency_score(self) -> float:
        """Calculate overall execution efficiency score (0.0 - 1.0)"""
        if self.steps_completed == 0:
            return 0.0
        
        completion_rate = self.steps_completed / max(1, self.steps_completed + self.steps_failed)
        resolution_rate = self.contradictions_resolved / max(1, self.contradictions_detected)
        adaptation_efficiency = 1.0 - min(1.0, self.plan_adaptations / max(1, self.steps_completed))
        
        return (completion_rate * 0.5 + resolution_rate * 0.3 + adaptation_efficiency * 0.2)


class ResearchExecutionEngine:
    """
    The cognitive research orchestrator that executes research plans.
    
    This is the central intelligence that coordinates all research subsystems
    to transform strategic research plans into actionable insights.
    """
    
    def __init__(
        self,
        session: ResearchSession,
        planner: DeepResearchPlanner,
        intelligence_engine: ResearchIntelligenceEngine,
        cache_engine: CognitiveResonanceCache,
        stream_manager: ResearchStreamManager,
        memory_manager: MemoryManager,
        memory_context: SessionMemoryContext,
        browser_agent: Optional[BrowserResearchAgent] = None,
        collaboration_hub: Optional[AgentCollaborationHub] = None,
        report_exporter: Optional[ReportExporter] = None
    ):
        """Initialize the research execution engine with all dependencies"""
        
        # Core session and state
        self.session = session
        self.state = ExecutionState.INITIALIZING
        self.current_step: Optional[ResearchStep] = None
        self.execution_start_time: Optional[datetime] = None
        
        # Injected service dependencies
        self.planner = planner
        self.intelligence = intelligence_engine
        self.cache = cache_engine
        self.streamer = stream_manager
        self.memory_manager = memory_manager
        self.memory_context = memory_context
        self.browser = browser_agent or BrowserResearchAgent()
        self.collab_hub = collaboration_hub or AgentCollaborationHub(None, None)
        self.exporter = report_exporter or ReportExporter()
        
        # Execution control
        self.is_running = False
        self.pause_requested = False
        self.stop_requested = False
        self.user_intervention_queue: asyncio.Queue = asyncio.Queue()
        
        # Metrics and monitoring
        self.metrics = ExecutionMetrics()
        self.step_history: List[Dict[str, Any]] = []
        
        # Background tasks
        self.intervention_handler_task: Optional[asyncio.Task] = None
        self.metrics_streaming_task: Optional[asyncio.Task] = None
        
        logger.info(f"ResearchExecutionEngine initialized for session {self.session.session_id}")
    
    async def run(self) -> Dict[str, Any]:
        """
        Execute the complete research plan with real-time monitoring and adaptation.
        
        Returns:
            Dict containing final execution results and summary
        """
        try:
            await self._initialize_execution()
            
            # Create a simple step sequence for demonstration
            steps = [
                ResearchStep("step_1", StepType.PRELIMINARY_MEMORY_SEARCH.value, "Search existing knowledge"),
                ResearchStep("step_2", StepType.EXPLORATORY_SEARCH.value, "Conduct web research"),
                ResearchStep("step_3", StepType.SYNTHESIS.value, "Synthesize findings"),
                ResearchStep("step_4", StepType.REPORT_GENERATION.value, "Generate final report")
            ]
            
            step_index = 0
            while self.is_running and step_index < len(steps):
                # Check for pause/stop requests
                if self.pause_requested:
                    await self._handle_pause()
                    continue
                    
                if self.stop_requested:
                    logger.info("Stop requested, terminating research execution")
                    break
                
                # Get next step
                self.current_step = steps[step_index]
                step_index += 1
                
                # Execute the step
                step_start_time = time.time()
                step_result = await self._execute_step(self.current_step)
                step_duration = time.time() - step_start_time
                
                # Record step execution
                self._record_step_execution(self.current_step, step_result, step_duration)
                
                # Post-step processing
                await self._process_step_results(step_result)
                
                # Check for plan adaptation needs (simplified)
                # await self._check_plan_adaptation()
                
                # Update metrics
                self.metrics.total_execution_time += step_duration
                self.metrics.steps_completed += 1
            
            # Finalize execution
            return await self._finalize_execution()
            
        except Exception as e:
            logger.error(f"Unhandled exception in research engine: {e}", exc_info=True)
            await self._handle_execution_error(e)
            raise
        
        finally:
            await self._cleanup_execution()
    
    async def pause(self):
        """Request execution pause"""
        logger.info("Pause requested for research execution")
        self.pause_requested = True
        self.state = ExecutionState.PAUSED
        
        await self.streamer.stream_event(StreamEvent(
            event_type=StreamEventType.EXECUTION_PAUSED,
            session_id=self.session.session_id,
            user_id=self.session.user_id,
            timestamp=datetime.now(timezone.utc),
            data={'current_step': self.current_step.to_dict() if self.current_step else None},
            priority=StreamPriority.HIGH
        ))
    
    async def resume(self):
        """Resume paused execution"""
        logger.info("Resume requested for research execution")
        self.pause_requested = False
        self.state = ExecutionState.RUNNING
        
        await self.streamer.stream_event(StreamEvent(
            event_type=StreamEventType.EXECUTION_RESUMED,
            session_id=self.session.session_id,
            user_id=self.session.user_id,
            timestamp=datetime.now(timezone.utc),
            data={'resuming_step': self.current_step.to_dict() if self.current_step else None},
            priority=StreamPriority.HIGH
        ))
    
    async def stop(self):
        """Request execution stop"""
        logger.info("Stop requested for research execution")
        self.stop_requested = True
        self.is_running = False
        self.state = ExecutionState.COMPLETED
    
    async def handle_user_intervention(self, intervention_type: InterventionType, data: Dict[str, Any]):
        """Handle real-time user intervention"""
        intervention = {
            'type': intervention_type,
            'data': data,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'session_id': self.session.session_id
        }
        
        await self.user_intervention_queue.put(intervention)
        self.metrics.user_interventions += 1
        
        # Record in session
        self.session.record_user_intervention(intervention_type.value, data)
        
        logger.info(f"User intervention queued: {intervention_type.value}")
    
    # ============================================================================
    # CORE EXECUTION METHODS
    # ============================================================================
    
    async def _initialize_execution(self):
        """Initialize execution environment and background tasks"""
        self.is_running = True
        self.state = ExecutionState.RUNNING
        self.execution_start_time = datetime.now(timezone.utc)
        
        # Update session status
        self.session.update_status(ResearchStatus.ACTIVE)
        
        # Start background tasks
        self.intervention_handler_task = asyncio.create_task(self._handle_interventions())
        self.metrics_streaming_task = asyncio.create_task(self._stream_metrics())
        
        # Initialize cache breath phase
        await self.cache.set_breath_phase(BreathPhase.INHALE)
        
        # Stream session start event
        await self.streamer.stream_event(StreamEvent(
            event_type=StreamEventType.SESSION_START,
            session_id=self.session.session_id,
            user_id=self.session.user_id,
            timestamp=self.execution_start_time,
            data={
                'session_summary': self.session.get_session_summary(),
                'plan_summary': self.session.plan.to_dict() if self.session.plan else None
            },
            priority=StreamPriority.HIGH
        ))
        
        logger.info(f"Research execution initialized for session {self.session.session_id}")
    
    async def _execute_step(self, step: ResearchStep) -> Dict[str, Any]:
        """Execute a single research step based on its type"""
        
        logger.info(f"Executing step: {step.step_type.value} - {step.goal}")
        
        # Stream step start event
        await self.streamer.stream_event(StreamEvent(
            event_type=StreamEventType.STEP_START,
            session_id=self.session.session_id,
            user_id=self.session.user_id,
            timestamp=datetime.now(timezone.utc),
            data={'step': step.to_dict()},
            priority=StreamPriority.NORMAL
        ))
        
        # Execute based on step type
        match step.step_type:
            case StepType.PRELIMINARY_MEMORY_SEARCH:
                result = await self._execute_memory_search(step)
            case StepType.EXPLORATORY_SEARCH:
                result = await self._execute_browser_search(step, depth='exploratory')
            case StepType.DEEP_DIVE:
                result = await self._execute_browser_search(step, depth='deep')
            case StepType.SYNTHESIS:
                result = await self._execute_collaborative_synthesis(step)
            case StepType.VALIDATION_AND_CONTRADICTION_RESOLUTION:
                result = await self._execute_validation(step)
            case StepType.REPORT_GENERATION:
                result = await self._execute_report_generation(step)
            case _:
                logger.warning(f"Unknown step type: {step.step_type}")
                result = {'status': 'skipped', 'reason': 'unknown_step_type'}
        
        # Stream step completion
        await self.streamer.stream_event(StreamEvent(
            event_type=StreamEventType.STEP_COMPLETE,
            session_id=self.session.session_id,
            user_id=self.session.user_id,
            timestamp=datetime.now(timezone.utc),
            data={
                'step_id': step.step_id,
                'result_summary': result.get('summary', 'Step completed'),
                'entities_processed': result.get('entities_processed', 0)
            },
            priority=StreamPriority.NORMAL
        ))
        
        return result
    
    async def _execute_memory_search(self, step: ResearchStep) -> Dict[str, Any]:
        """Execute memory search to retrieve pre-existing knowledge"""
        
        logger.info(f"Executing memory search: {step.goal}")
        self.metrics.memory_queries_executed += 1
        
        try:
            # Construct semantic query from research objectives
            query_text = f"{self.session.query} {step.goal}"
            
            # Search memory system for relevant knowledge
            retrieved_memories = await self.memory_manager.retrieve_memories(
                user_id=self.session.user_id,
                query_text=query_text,
                memory_types=[MemoryType.KNOWLEDGE, MemoryType.RESEARCH],
                limit=step.parameters.get('max_memories', 20),
                min_relevance=step.parameters.get('min_relevance', 0.3)
            )
            
            # Convert memories to research entities
            entities_created = 0
            knowledge_saturation = 0.0
            
            for memory in retrieved_memories:
                # Create research entity from memory
                entity = self.session.add_entity(
                    content=memory.content,
                    source_url=f"memory://{memory.memory_id}",
                    source_type="memory",
                    extraction_method="semantic_search"
                )
                
                # Update entity scores based on memory metadata
                credibility = memory.relevance_score
                bias = 0.7  # Assume stored memories have reasonable bias score
                completeness = min(1.0, len(memory.content) / 500)  # Based on content length
                
                self.session.update_entity_scores(entity.entity_id, credibility, bias, completeness)
                
                # Store entity in cache
                await self.cache.store_entity(
                    entity_id=entity.entity_id,
                    content=entity.content_text,
                    source_url=entity.source_url,
                    metadata={
                        'memory_id': str(memory.memory_id),
                        'memory_importance': memory.importance.value,
                        'created_from_memory': True
                    }
                )
                
                entities_created += 1
            
            # Calculate knowledge saturation from memory
            if entities_created > 0:
                avg_relevance = sum(m.relevance_score for m in retrieved_memories) / len(retrieved_memories)
                knowledge_saturation = min(1.0, avg_relevance * (entities_created / 10))
            
            # Update session with memory context
            await self.memory_context.record_memory_usage(
                memory_ids=[m.memory_id for m in retrieved_memories],
                usage_type="research_retrieval",
                metadata={'step_id': step.step_id, 'entities_created': entities_created}
            )
            
            result = {
                'status': 'completed',
                'entities_processed': entities_created,
                'memories_retrieved': len(retrieved_memories),
                'knowledge_saturation': knowledge_saturation,
                'summary': f"Retrieved {len(retrieved_memories)} memories, created {entities_created} entities"
            }
            
            # Check if we have enough information to skip further searches
            if knowledge_saturation > 0.8:
                await self.planner.suggest_plan_adaptation(
                    self.session,
                    PlanAdaptationTrigger.HIGH_SATURATION,
                    {'saturation_score': knowledge_saturation, 'source': 'memory_search'}
                )
            
            logger.info(f"Memory search completed: {result['summary']}")
            return result
            
        except Exception as e:
            logger.error(f"Memory search failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'entities_processed': 0
            }
    
    async def _execute_browser_search(self, step: ResearchStep, depth: str = 'exploratory') -> Dict[str, Any]:
        """Execute browser-based research using AI browser agent"""
        
        logger.info(f"Executing browser search ({depth}): {step.goal}")
        self.metrics.browser_tasks_executed += 1
        
        # Set cache breath phase for information gathering
        await self.cache.set_breath_phase(BreathPhase.INHALE)
        
        try:
            # Create browser task
            browser_task = BrowserTask(
                task_id=str(uuid4()),
                task_type='research_search',
                objective=step.goal,
                queries=step.parameters.get('queries', [step.goal]),
                max_sources=step.parameters.get('max_sources', 10 if depth == 'exploratory' else 25),
                depth_level=depth,
                data_extraction_requirements=[
                    'full_text', 'metadata', 'citations', 'images'
                ],
                quality_filters={
                    'min_content_length': 200,
                    'exclude_domains': self.session.plan.blocked_domains if self.session.plan else [],
                    'prefer_domains': self.session.plan.trusted_domains if self.session.plan else []
                }
            )
            
            # Execute browser task
            browser_result: BrowserResult = await self.browser.execute_task(browser_task)
            
            # Process browser results into research entities
            entities_created = 0
            contradictions_detected = 0
            
            for source_data in browser_result.sources:
                # Create research entity
                entity = self.session.add_entity(
                    content=source_data['content'],
                    source_url=source_data['url'],
                    source_type='web_page',
                    extraction_method='browser_agent'
                )
                
                # Assess entity quality using intelligence engine
                quality_assessment = await self.intelligence.assess_source_quality(
                    content=source_data['content'],
                    url=source_data['url'],
                    metadata=source_data.get('metadata', {})
                )
                
                # Update entity scores
                self.session.update_entity_scores(
                    entity.entity_id,
                    quality_assessment.credibility_score,
                    quality_assessment.bias_score,
                    quality_assessment.completeness_score
                )
                
                # Store in cache with quality assessment
                await self.cache.store_entity(
                    entity_id=entity.entity_id,
                    content=entity.content_text,
                    source_url=entity.source_url,
                    metadata={
                        'quality_assessment': quality_assessment.to_dict(),
                        'extraction_timestamp': datetime.now(timezone.utc).isoformat(),
                        'browser_task_id': browser_task.task_id
                    }
                )
                
                # Check for ethical concerns
                if quality_assessment.ethical_flags:
                    for flag in quality_assessment.ethical_flags:
                        self.session.add_ethical_concern(
                            EthicalConcern(flag['type']),
                            flag['description'],
                            entity.entity_id
                        )
                
                entities_created += 1
            
            # Detect contradictions between new entities
            contradiction_analysis = await self.intelligence.detect_contradictions(
                [entity for entity in self.session.entities.values() 
                 if entity.entity_id in [e.entity_id for e in browser_result.sources]]
            )
            
            contradictions_detected = len(contradiction_analysis.contradictions)
            self.metrics.contradictions_detected += contradictions_detected
            
            # Store results in memory for future sessions
            await self.memory_context.store_research_findings(
                findings=browser_result.sources,
                search_query=step.goal,
                metadata={
                    'step_type': step.step_type.value,
                    'depth_level': depth,
                    'entities_created': entities_created
                }
            )
            
            result = {
                'status': 'completed',
                'entities_processed': entities_created,
                'sources_found': len(browser_result.sources),
                'contradictions_detected': contradictions_detected,
                'summary': f"Found {len(browser_result.sources)} sources, created {entities_created} entities"
            }
            
            logger.info(f"Browser search completed: {result['summary']}")
            return result
            
        except Exception as e:
            logger.error(f"Browser search failed: {e}")
            self.metrics.steps_failed += 1
            return {
                'status': 'failed',
                'error': str(e),
                'entities_processed': 0
            }
    
    async def _execute_collaborative_synthesis(self, step: ResearchStep) -> Dict[str, Any]:
        """Execute knowledge synthesis using multi-agent collaboration"""
        
        logger.info(f"Executing synthesis: {step.goal}")
        self.metrics.synthesis_operations += 1
        
        # Set cache breath phase for integration
        await self.cache.set_breath_phase(BreathPhase.HOLD_IN)
        
        try:
            # Select high-quality entities for synthesis
            synthesis_entities = [
                entity for entity in self.session.entities.values()
                if entity.credibility_score > 0.6 and entity.bias_score > 0.4
            ]
            
            if not synthesis_entities:
                return {
                    'status': 'skipped',
                    'reason': 'no_quality_entities',
                    'entities_processed': 0
                }
            
            # Create collaboration task
            collab_task = AgentTask(
                task_id=str(uuid4()),
                task_type='knowledge_synthesis',
                objective=step.goal,
                input_data={
                    'entities': [entity.to_dict() for entity in synthesis_entities],
                    'research_question': self.session.query,
                    'synthesis_requirements': step.parameters.get('requirements', [])
                },
                required_agents=['SynthesizerAgent', 'ValidatorAgent'],
                quality_requirements={
                    'min_confidence': 0.7,
                    'require_citations': True,
                    'check_contradictions': True
                }
            )
            
            # Execute collaboration
            collab_result: CollaborationResult = await self.collab_hub.execute_task(collab_task)
            
            # Process synthesis result
            if collab_result.success:
                # Create synthesis entity
                synthesis_content = collab_result.result.get('synthesis_text', '')
                synthesis_entity = self.session.add_entity(
                    content=synthesis_content,
                    source_url=f"synthesis://{collab_task.task_id}",
                    source_type='synthesis',
                    extraction_method='agent_collaboration'
                )
                
                # High scores for synthesis results
                self.session.update_entity_scores(
                    synthesis_entity.entity_id,
                    credibility=collab_result.confidence,
                    bias=0.8,  # Synthesis should be relatively unbiased
                    completeness=0.9  # Synthesis is comprehensive by nature
                )
                
                # Record synthesis in session
                self.session.add_synthesis_result({
                    'synthesis_id': synthesis_entity.entity_id,
                    'input_entities': [e.entity_id for e in synthesis_entities],
                    'result': synthesis_content,
                    'confidence': collab_result.confidence,
                    'method': 'multi_agent_collaboration'
                })
                
                result = {
                    'status': 'completed',
                    'entities_processed': 1,
                    'input_entities': len(synthesis_entities),
                    'synthesis_confidence': collab_result.confidence,
                    'summary': f"Synthesized knowledge from {len(synthesis_entities)} entities"
                }
            else:
                result = {
                    'status': 'failed',
                    'error': collab_result.error,
                    'entities_processed': 0
                }
            
            logger.info(f"Synthesis completed: {result.get('summary', result.get('error', 'Unknown result'))}")
            return result
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'entities_processed': 0
            }
    
    async def _execute_validation(self, step: ResearchStep) -> Dict[str, Any]:
        """Execute validation and contradiction resolution"""
        
        logger.info("Executing validation and contradiction resolution")
        
        # Set cache breath phase for analysis
        await self.cache.set_breath_phase(BreathPhase.EXHALE)
        
        try:
            # Get current contradictions
            contradictions = list(self.session.contradictions.values())
            unresolved_contradictions = [
                c for c in contradictions if c.resolution_status == 'unresolved'
            ]
            
            if not unresolved_contradictions:
                return {
                    'status': 'completed',
                    'entities_processed': 0,
                    'contradictions_resolved': 0,
                    'summary': 'No contradictions to resolve'
                }
            
            resolved_count = 0
            
            for contradiction in unresolved_contradictions:
                # Get the conflicting entities
                entity1 = self.session.entities.get(contradiction.entity1_id)
                entity2 = self.session.entities.get(contradiction.entity2_id)
                
                if not entity1 or not entity2:
                    continue
                
                # Create debate task for resolution
                debate_task = AgentTask(
                    task_id=str(uuid4()),
                    task_type='contradiction_resolution',
                    objective=f"Resolve contradiction: {contradiction.description}",
                    input_data={
                        'entity1': entity1.to_dict(),
                        'entity2': entity2.to_dict(),
                        'contradiction': contradiction.to_dict()
                    },
                    required_agents=['ProponentAgent', 'OpponentAgent', 'ModeratorAgent'],
                    quality_requirements={
                        'require_evidence': True,
                        'logical_consistency': True,
                        'min_confidence': 0.6
                    }
                )
                
                # Execute debate
                debate_result = await self.collab_hub.execute_task(debate_task)
                
                if debate_result.success:
                    # Update contradiction with resolution
                    contradiction.resolution_status = 'resolved'
                    contradiction.resolution_notes = debate_result.result.get('resolution_summary', '')
                    
                    # Create resolution entity if substantive
                    resolution_content = debate_result.result.get('resolution_text', '')
                    if len(resolution_content) > 100:
                        resolution_entity = self.session.add_entity(
                            content=resolution_content,
                            source_url=f"resolution://{debate_task.task_id}",
                            source_type='contradiction_resolution',
                            extraction_method='agent_debate'
                        )
                    
                    resolved_count += 1
                    self.metrics.contradictions_resolved += 1
            
            result = {
                'status': 'completed',
                'entities_processed': resolved_count,
                'contradictions_found': len(contradictions),
                'contradictions_resolved': resolved_count,
                'summary': f"Resolved {resolved_count}/{len(unresolved_contradictions)} contradictions"
            }
            
            logger.info(f"Validation completed: {result['summary']}")
            return result
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'entities_processed': 0
            }
    
    async def _execute_report_generation(self, step: ResearchStep) -> Dict[str, Any]:
        """Execute final report generation and artifact creation"""
        
        logger.info("Executing report generation")
        
        try:
            # Prepare comprehensive research data package
            research_package = {
                'session_summary': self.session.get_session_summary(),
                'entities': {eid: entity.to_dict() for eid, entity in self.session.entities.items()},
                'contradictions': {cid: contra.to_dict() for cid, contra in self.session.contradictions.items()},
                'synthesis_results': self.session.synthesis_results,
                'citations': self.session.citations,
                'ethical_concerns': self.session.ethical_concerns,
                'execution_metrics': self.metrics.to_dict(),
                'plan_history': self.session.plan.modification_history if self.session.plan else []
            }
            
            # Determine export formats
            export_formats = step.parameters.get('formats', ['interactive_html', 'markdown_summary'])
            
            artifacts_created = []
            
            for format_name in export_formats:
                export_format = ExportFormat(format_name)
                
                # Generate artifact
                artifact_ref: ArtifactReference = await self.exporter.generate_artifact(
                    research_data=research_package,
                    export_format=export_format,
                    session_id=self.session.session_id,
                    user_id=self.session.user_id
                )
                
                artifacts_created.append(artifact_ref.artifact_id)
                self.session.artifact_exports.append(artifact_ref.artifact_id)
            
            # Store final results in memory
            await self.memory_context.store_research_completion(
                final_results=research_package,
                artifacts_created=artifacts_created,
                metadata={
                    'total_entities': len(self.session.entities),
                    'execution_time': self.metrics.total_execution_time,
                    'quality_score': self.session.metrics.average_credibility
                }
            )
            
            # Update session status
            self.session.update_status(ResearchStatus.COMPLETED)
            
            result = {
                'status': 'completed',
                'artifacts_created': artifacts_created,
                'entities_processed': len(self.session.entities),
                'formats_generated': len(export_formats),
                'summary': f"Generated {len(artifacts_created)} artifacts from {len(self.session.entities)} entities"
            }
            
            logger.info(f"Report generation completed: {result['summary']}")
            return result
            
        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            return {
                'status': 'failed',
                'error': str(e),
                'entities_processed': 0
            }
    
    # ============================================================================
    # POST-EXECUTION PROCESSING
    # ============================================================================
    
    async def _process_step_results(self, step_result: Dict[str, Any]):
        """Process results from step execution and update intelligence metrics"""
        
        # Update session metrics
        self.session.metrics.update_metrics(
            list(self.session.entities.values()),
            list(self.session.contradictions.values())
        )
        
        # Get updated intelligence assessment
        intelligence_metrics = await self.intelligence.assess_session_state(self.session)
        
        # Stream metrics update
        await self.streamer.stream_event(StreamEvent(
            event_type=StreamEventType.METRICS_UPDATE,
            session_id=self.session.session_id,
            user_id=self.session.user_id,
            timestamp=datetime.now(timezone.utc),
            data={
                'session_metrics': self.session.metrics.to_dict(),
                'execution_metrics': self.metrics.to_dict(),
                'intelligence_metrics': intelligence_metrics.to_dict(),
                'step_result': step_result
            },
            priority=StreamPriority.NORMAL
        ))
    
    async def _check_plan_adaptation(self):
        """Check if plan adaptation is needed based on current state"""
        
        intelligence_metrics = await self.intelligence.assess_session_state(self.session)
        
        if self.intelligence.is_plan_adaptation_needed(intelligence_metrics):
            logger.info("Intelligence engine recommends plan adaptation")
            self.state = ExecutionState.ADAPTING_PLAN
            
            # Request plan adaptation
            adaptation_result = await self.planner.adapt_plan(self.session, intelligence_metrics)
            
            if adaptation_result.success:
                self.metrics.plan_adaptations += 1
                
                # Stream plan modification event
                await self.streamer.stream_event(StreamEvent(
                    event_type=StreamEventType.PLAN_MODIFIED,
                    session_id=self.session.session_id,
                    user_id=self.session.user_id,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        'adaptation_reason': adaptation_result.reason,
                        'changes_made': adaptation_result.changes,
                        'new_plan_version': self.session.plan.version if self.session.plan else None
                    },
                    priority=StreamPriority.HIGH
                ))
            
            self.state = ExecutionState.RUNNING
    
    # ============================================================================
    # BACKGROUND TASKS AND UTILITIES
    # ============================================================================
    
    async def _handle_interventions(self):
        """Background task to handle user interventions"""
        
        while self.is_running:
            try:
                # Wait for intervention with timeout
                intervention = await asyncio.wait_for(
                    self.user_intervention_queue.get(), 
                    timeout=1.0
                )
                
                intervention_type = InterventionType(intervention['type'])
                data = intervention['data']
                
                logger.info(f"Processing user intervention: {intervention_type.value}")
                
                match intervention_type:
                    case InterventionType.PAUSE:
                        await self.pause()
                    case InterventionType.RESUME:
                        await self.resume()
                    case InterventionType.STOP:
                        await self.stop()
                    case InterventionType.SKIP_STEP:
                        # Mark current step as skipped
                        self.metrics.steps_skipped += 1
                        logger.info("Current step skipped by user")
                    case InterventionType.MODIFY_PLAN:
                        # Apply plan modifications
                        if self.session.plan:
                            self.session.modify_plan(data.get('modifications', {}), 
                                                   data.get('reason', 'User intervention'))
                    case InterventionType.ADD_SOURCE:
                        # Add user-specified source
                        source_url = data.get('url')
                        if source_url:
                            # This would trigger a special browser task
                            logger.info(f"User added source: {source_url}")
                    case InterventionType.BLOCK_SOURCE:
                        # Block specified source
                        blocked_url = data.get('url')
                        if blocked_url and self.session.plan:
                            self.session.plan.blocked_domains.append(blocked_url)
                            logger.info(f"User blocked source: {blocked_url}")
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error handling intervention: {e}")
    
    async def _stream_metrics(self):
        """Background task to stream real-time metrics"""
        
        while self.is_running:
            try:
                await asyncio.sleep(5.0)  # Stream every 5 seconds
                
                await self.streamer.stream_event(StreamEvent(
                    event_type=StreamEventType.METRICS_UPDATE,
                    session_id=self.session.session_id,
                    user_id=self.session.user_id,
                    timestamp=datetime.now(timezone.utc),
                    data={
                        'execution_metrics': self.metrics.to_dict(),
                        'session_progress': self.session.get_progress_summary(),
                        'current_step': self.current_step.to_dict() if self.current_step else None
                    },
                    priority=StreamPriority.LOW
                ))
                
            except Exception as e:
                logger.error(f"Error streaming metrics: {e}")
    
    def _record_step_execution(self, step: ResearchStep, result: Dict[str, Any], duration: float):
        """Record step execution in history"""
        
        step_record = {
            'step_id': step.step_id,
            'step_type': step.step_type.value,
            'goal': step.goal,
            'start_time': (datetime.now(timezone.utc) - timedelta(seconds=duration)).isoformat(),
            'duration': duration,
            'result': result,
            'entities_processed': result.get('entities_processed', 0)
        }
        
        self.step_history.append(step_record)
        
        # Update session processing history
        self.session.processing_history.append({
            'type': 'step_execution',
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'data': step_record
        })
    
    async def _handle_pause(self):
        """Handle execution pause state"""
        
        logger.info("Research execution paused")
        
        while self.pause_requested and not self.stop_requested:
            await asyncio.sleep(1.0)
        
        logger.info("Research execution resumed from pause")
    
    async def _finalize_execution(self) -> Dict[str, Any]:
        """Finalize execution and return summary"""
        
        execution_duration = (datetime.now(timezone.utc) - self.execution_start_time).total_seconds()
        
        final_summary = {
            'session_id': self.session.session_id,
            'execution_summary': {
                'total_duration': execution_duration,
                'steps_completed': self.metrics.steps_completed,
                'entities_processed': self.metrics.knowledge_entities_processed,
                'contradictions_resolved': self.metrics.contradictions_resolved,
                'artifacts_created': len(self.session.artifact_exports),
                'efficiency_score': self.metrics._calculate_efficiency_score()
            },
            'final_session_state': self.session.get_session_summary(),
            'step_history': self.step_history
        }
        
        # Stream session completion
        await self.streamer.stream_event(StreamEvent(
            event_type=StreamEventType.SESSION_COMPLETE,
            session_id=self.session.session_id,
            user_id=self.session.user_id,
            timestamp=datetime.now(timezone.utc),
            data=final_summary,
            priority=StreamPriority.HIGH
        ))
        
        # Store execution summary in memory
        if self.memory_context:
            await self.memory_context.store_session_completion(final_summary)
        
        logger.info(f"Research execution completed: {final_summary['execution_summary']}")
        return final_summary
    
    async def _handle_execution_error(self, error: Exception):
        """Handle execution errors"""
        
        self.state = ExecutionState.ERROR
        
        error_data = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'step_context': self.current_step.to_dict() if self.current_step else None,
            'execution_metrics': self.metrics.to_dict(),
            'traceback': traceback.format_exc()
        }
        
        # Stream error event
        await self.streamer.stream_event(StreamEvent(
            event_type=StreamEventType.SESSION_ERROR,
            session_id=self.session.session_id,
            user_id=self.session.user_id,
            timestamp=datetime.now(timezone.utc),
            data=error_data,
            priority=StreamPriority.HIGH
        ))
        
        # Update session status
        self.session.update_status(ResearchStatus.ERROR)
        
        logger.error(f"Research execution failed: {error_data}")
    
    async def _cleanup_execution(self):
        """Cleanup execution resources"""
        
        self.is_running = False
        
        # Cancel background tasks
        if self.intervention_handler_task:
            self.intervention_handler_task.cancel()
        if self.metrics_streaming_task:
            self.metrics_streaming_task.cancel()
        
        # Clean up cache breath state
        await self.cache.set_breath_phase(BreathPhase.EXHALE)
        
        logger.info("Research execution cleanup completed")


# ============================================================================
# FACTORY FUNCTIONS AND UTILITIES
# ============================================================================

async def create_research_execution_engine(
    session: ResearchSession,
    system_components: Dict[str, Any]
) -> ResearchExecutionEngine:
    """
    Factory function to create research execution engine with all dependencies.
    
    Args:
        session: The research session to execute
        system_components: Dictionary containing all system components
    
    Returns:
        Configured ResearchExecutionEngine instance
    """
    
    return ResearchExecutionEngine(
        session=session,
        planner=system_components['planner'],
        intelligence_engine=system_components['intelligence_engine'],
        cache_engine=system_components['cache_engine'],
        stream_manager=system_components['stream_manager'],
        memory_manager=system_components['memory_manager'],
        memory_context=system_components['memory_context'],
        browser_agent=system_components['browser_agent'],
        collaboration_hub=system_components['collaboration_hub'],
        report_exporter=system_components['report_exporter']
    )


def get_execution_engine_config() -> Dict[str, Any]:
    """Get default configuration for research execution engine"""
    
    return {
        'max_step_duration': 300,  # 5 minutes per step
        'intervention_queue_size': 100,
        'metrics_streaming_interval': 5.0,
        'plan_adaptation_threshold': 0.8,
        'error_retry_attempts': 3,
        'memory_integration_enabled': True,
        'real_time_streaming_enabled': True
    }
