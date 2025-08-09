"""
SOMNUS RESEARCH - Deep Research Planner
Generates editable recursive research plans with multi-AI collaboration support.

Features:
- Recursive plan trees with sub-questions and domain delegation
- Multi-AI collaboration with domain expertise assignment
- Integration with prompt_manager for research personality stability
- VM/Memory persistence for research environment submenu
- Agent collaboration orchestration for large research tasks
- Post-research synthesis coordination
- Configurable research parameters and ethical guidelines
- Real-time plan modification and user intervention points

This creates the foundation for collaborative AI research teams working together
on complex research tasks, then collaborating on document creation via artifacts.
"""

import asyncio
import json
import logging
import yaml
import re
import hashlib
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import traceback

from pydantic import BaseModel, Field, validator

# Somnus system imports
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope
from core.memory_integration import SessionMemoryContext
from core.model_loader import ModelLoader
from prompt_manager import SystemPromptManager, PromptType, PromptContext
from agent_collaboration_core import AgentCollaborationHub, AgentProfile, AgentCapability
from agent_communication_protocol import CommunicationOrchestrator, MessageType
from schemas.session import SessionID, UserID
from research_stream_manager import ResearchStreamManager, StreamEvent, StreamEventType, StreamPriority

logger = logging.getLogger(__name__)


class ResearchComplexity(str, Enum):
    """Research complexity levels for AI assignment"""
    SIMPLE = "simple"           # Single AI, straightforward topics
    MODERATE = "moderate"       # 2-3 AIs, some specialization needed
    COMPLEX = "complex"         # 3-5 AIs, domain expertise required
    EXPERT = "expert"           # 5+ AIs, deep specialization needed
    MASSIVE = "massive"         # 10+ AIs, enterprise-level research


class DomainExpertise(str, Enum):
    """Domain expertise areas for AI agent assignment"""
    TECHNOLOGY = "technology"
    SCIENCE = "science"
    BUSINESS = "business"
    POLITICS = "politics"
    HEALTHCARE = "healthcare"
    EDUCATION = "education"
    FINANCE = "finance"
    LEGAL = "legal"
    ARTS_CULTURE = "arts_culture"
    HISTORY = "history"
    PHILOSOPHY = "philosophy"
    PSYCHOLOGY = "psychology"
    ECONOMICS = "economics"
    ENVIRONMENT = "environment"
    SOCIAL_ISSUES = "social_issues"


class CollaborationMode(str, Enum):
    """Collaboration modes for multi-AI research"""
    SOLO = "solo"                       # Single AI researcher
    PARALLEL = "parallel"               # Independent parallel research
    COLLABORATIVE = "collaborative"     # Real-time collaboration
    HIERARCHICAL = "hierarchical"       # Lead researcher + specialists
    DEMOCRATIC = "democratic"           # Consensus-based decisions


class ResearchPhase(str, Enum):
    """Research execution phases"""
    DOMAIN_ASSIGNMENT = "domain_assignment"
    PARALLEL_RESEARCH = "parallel_research"
    SYNTHESIS_PREPARATION = "synthesis_preparation"
    COLLABORATIVE_SYNTHESIS = "collaborative_synthesis"
    DOCUMENT_CREATION = "document_creation"
    FINALIZATION = "finalization"


class ResearchDepthLevel(str, Enum):
    """Research depth levels for recursive expansion"""
    SURFACE = "surface"        # Basic search results
    MODERATE = "moderate"      # Some source analysis
    DEEP = "deep"             # Comprehensive analysis
    EXHAUSTIVE = "exhaustive"  # Full recursive reasoning


@dataclass
class ResearchPlan:
    """Comprehensive research plan structure"""
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
class CollaborationPlan:
    """Plan for multi-AI collaboration"""
    collaboration_id: str
    mode: CollaborationMode
    participating_agents: List[str]
    domain_assignments: Dict[str, List[str]]  # agent_id -> domain list
    lead_agent_id: Optional[str] = None
    coordination_schedule: List[Dict[str, Any]] = field(default_factory=list)
    synthesis_plan: Dict[str, Any] = field(default_factory=dict)
    communication_channels: List[str] = field(default_factory=list)
    peer_review_enabled: bool = True
    quality_thresholds: Dict[str, float] = field(default_factory=dict)
    phase_deadlines: Dict[str, datetime] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'collaboration_id': self.collaboration_id,
            'mode': self.mode.value,
            'participating_agents': self.participating_agents,
            'domain_assignments': self.domain_assignments,
            'lead_agent_id': self.lead_agent_id,
            'coordination_schedule': self.coordination_schedule,
            'synthesis_plan': self.synthesis_plan,
            'communication_channels': self.communication_channels,
            'peer_review_enabled': self.peer_review_enabled,
            'quality_thresholds': self.quality_thresholds,
            'phase_deadlines': {k: v.isoformat() for k, v in self.phase_deadlines.items()}
        }


@dataclass
class ResearchPersonality:
    """Research personality profile for AI agents"""
    personality_id: str
    name: str
    research_approach: str  # systematic, creative, analytical, exploratory
    bias_tolerance: float  # 0.0-1.0
    depth_preference: ResearchDepthLevel
    source_preferences: List[str]  # academic, news, primary, secondary
    ethical_guidelines: Dict[str, Any]
    collaboration_style: str  # supportive, challenging, independent, leadership
    risk_tolerance: float  # 0.0-1.0
    time_management: str  # thorough, efficient, balanced
    reporting_frequency: str  # continuous, periodic, milestone, final
    detail_level: str  # summary, detailed, comprehensive
    
    def to_prompt_variables(self) -> Dict[str, Any]:
        """Convert to variables for prompt generation"""
        return {
            'research_approach': self.research_approach,
            'bias_tolerance': self.bias_tolerance,
            'depth_preference': self.depth_preference.value,
            'source_preferences': self.source_preferences,
            'collaboration_style': self.collaboration_style,
            'risk_tolerance': self.risk_tolerance,
            'time_management': self.time_management,
            'reporting_frequency': self.reporting_frequency,
            'detail_level': self.detail_level,
            'ethical_guidelines': self.ethical_guidelines
        }


class ResearchPlannerConfig(BaseModel):
    """Configuration for research planner"""
    # Basic settings
    max_agents_per_research: int = 10
    default_collaboration_mode: CollaborationMode = CollaborationMode.PARALLEL
    default_research_depth: ResearchDepthLevel = ResearchDepthLevel.DEEP
    
    # Quality settings
    min_sources_per_domain: int = 5
    max_sources_per_domain: int = 50
    credibility_threshold: float = 0.7
    bias_tolerance: float = 0.3
    
    # Time management
    default_time_per_domain_hours: int = 2
    max_research_duration_hours: int = 24
    synthesis_time_multiplier: float = 0.5  # Synthesis takes 50% of research time
    
    # Collaboration settings
    enable_peer_review: bool = True
    require_consensus: bool = False
    allow_domain_reassignment: bool = True
    
    # Memory and persistence
    store_plans_in_memory: bool = True
    store_plans_in_vm: bool = True
    plan_backup_enabled: bool = True
    
    # Prompt management
    use_dynamic_personalities: bool = True
    allow_personality_evolution: bool = True
    personality_stability_weight: float = 0.8


class DeepResearchPlanner:
    """
    Advanced research planner with multi-AI collaboration support.
    
    Creates recursive research plans that can be executed by teams of AI agents,
    each with specialized domain expertise and collaborative workflows.
    """
    
    def __init__(self,
                 memory_manager: MemoryManager,
                 model_loader: ModelLoader,
                 prompt_manager: SystemPromptManager,
                 collaboration_hub: AgentCollaborationHub,
                 stream_manager: ResearchStreamManager,
                 config: Optional[ResearchPlannerConfig] = None):
        
        self.memory_manager = memory_manager
        self.model_loader = model_loader
        self.prompt_manager = prompt_manager
        self.collaboration_hub = collaboration_hub
        self.stream_manager = stream_manager
        self.config = config or ResearchPlannerConfig()
        
        # Plan storage
        self.active_plans: Dict[str, ResearchPlan] = {}
        self.collaboration_plans: Dict[str, CollaborationPlan] = {}
        self.research_personalities: Dict[str, ResearchPersonality] = {}
        
        # VM and persistence paths
        self.vm_plans_path = Path("data/research_vm/plans")
        self.vm_personalities_path = Path("data/research_vm/personalities")
        self.vm_plans_path.mkdir(parents=True, exist_ok=True)
        self.vm_personalities_path.mkdir(parents=True, exist_ok=True)
        
        # Domain expertise mapping
        self.domain_agent_mapping: Dict[DomainExpertise, List[str]] = {}
        
        # Initialize default personalities
        asyncio.create_task(self._initialize_default_personalities())
        
        logger.info("Deep Research Planner initialized with multi-AI collaboration support")
    
    async def generate_research_plan(self,
                                   primary_question: str,
                                   user_id: str,
                                   session_id: str,
                                   research_config: Optional[Dict[str, Any]] = None) -> Tuple[ResearchPlan, Optional[CollaborationPlan]]:
        """
        Generate comprehensive research plan with optional multi-AI collaboration.
        """
        
        logger.info(f"Generating research plan for: {primary_question}")
        
        try:
            # Parse configuration
            config = research_config or {}
            enable_collaboration = config.get('enable_collaboration', False)
            max_agents = config.get('max_agents', 1)
            complexity = ResearchComplexity(config.get('complexity', ResearchComplexity.MODERATE.value))
            
            # Generate plan ID
            plan_id = f"research_plan_{int(datetime.now().timestamp())}_{session_id[:8]}"
            
            # Analyze question complexity and domains
            question_analysis = await self._analyze_research_question(primary_question)
            
            # Generate sub-questions using AI
            sub_questions = await self._generate_sub_questions(primary_question, question_analysis)
            
            # Determine research goals
            research_goals = await self._determine_research_goals(primary_question, question_analysis)
            
            # Create base research plan
            research_plan = ResearchPlan(
                plan_id=plan_id,
                primary_question=primary_question,
                sub_questions=sub_questions,
                research_goals=research_goals,
                depth_level=ResearchDepthLevel(config.get('depth_level', ResearchDepthLevel.DEEP.value)),
                max_sources=config.get('max_sources', 30),
                trusted_domains=config.get('trusted_domains', []),
                blocked_domains=config.get('blocked_domains', []),
                source_types=config.get('source_types', ["web", "academic", "news"]),
                ethical_guidelines=config.get('ethical_guidelines', {}),
                time_constraints=config.get('time_constraints', {})
            )
            
            # Generate collaboration plan if requested
            collaboration_plan = None
            if enable_collaboration and max_agents > 1:
                collaboration_plan = await self._generate_collaboration_plan(
                    research_plan, 
                    question_analysis, 
                    max_agents,
                    complexity,
                    user_id
                )
            
            # Store plans
            self.active_plans[plan_id] = research_plan
            if collaboration_plan:
                self.collaboration_plans[collaboration_plan.collaboration_id] = collaboration_plan
            
            # Persist plans
            await self._persist_plan(research_plan, collaboration_plan)
            
            # Stream plan generation event
            await self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.PLAN_GENERATED,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'plan_id': plan_id,
                    'primary_question': primary_question,
                    'collaboration_enabled': collaboration_plan is not None,
                    'agent_count': len(collaboration_plan.participating_agents) if collaboration_plan else 1,
                    'complexity': complexity.value,
                    'estimated_duration': self._estimate_research_duration(research_plan, collaboration_plan)
                },
                priority=StreamPriority.HIGH
            ))
            
            logger.info(f"Generated research plan {plan_id} with {len(sub_questions)} sub-questions")
            if collaboration_plan:
                logger.info(f"Collaboration plan created with {len(collaboration_plan.participating_agents)} agents")
            
            return research_plan, collaboration_plan
            
        except Exception as e:
            logger.error(f"Error generating research plan: {e}")
            logger.error(traceback.format_exc())
            raise
    
    async def _analyze_research_question(self, question: str) -> Dict[str, Any]:
        """Analyze research question to determine complexity and domains using AI"""
        
        try:
            # Create analysis prompt using prompt manager
            analysis_prompt = await self.prompt_manager.generate_system_prompt(
                user_id="research_planner",
                session_id="analysis",
                prompt_type=PromptType.RESEARCH_MODE,
                context=PromptContext.ANALYSIS,
                custom_variables={
                    'task': 'question_analysis',
                    'question': question,
                    'return_format': 'json'
                }
            )
            
            # Add detailed analysis instructions
            full_prompt = f"""{analysis_prompt}

Analyze this research question for complexity and domain requirements:

Question: "{question}"

Provide comprehensive analysis in JSON format:
{{
    "complexity": "simple|moderate|complex|expert|massive",
    "primary_domains": ["technology", "science", "business", "politics", "healthcare", "education", "finance", "legal", "arts_culture", "history", "philosophy", "psychology", "economics", "environment", "social_issues"],
    "estimated_agent_count": 1-10,
    "requires_specialization": true/false,
    "interdisciplinary": true/false,
    "time_sensitivity": "low|medium|high",
    "ethical_considerations": ["privacy", "bias", "misinformation", "harm", "fairness"],
    "source_types_needed": ["academic", "news", "primary", "secondary", "government", "industry"],
    "research_approaches": ["systematic", "exploratory", "analytical", "creative"],
    "potential_controversies": ["string array of potential controversial aspects"],
    "key_entities": ["important organizations, people, concepts to focus on"],
    "geographic_scope": "local|national|global",
    "temporal_scope": "historical|current|future|longitudinal"
}}

Provide thorough analysis considering:
1. Domain expertise requirements
2. Potential interdisciplinary connections
3. Ethical implications and bias risks
4. Resource and time requirements
5. Methodological approaches needed
"""
            
            # Get AI analysis using model loader
            model = await self.model_loader.get_model("research_analysis")
            response = await model.generate_completion(
                prompt=full_prompt,
                max_tokens=1500,
                temperature=0.3
            )
            
            # Parse JSON response
            try:
                analysis_data = json.loads(response.strip())
                
                # Validate and enhance analysis
                analysis_data = self._validate_and_enhance_analysis(analysis_data, question)
                
                logger.info(f"Successfully analyzed question with complexity: {analysis_data.get('complexity')}")
                return analysis_data
                
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse AI analysis JSON: {e}")
                return self._fallback_analysis(question)
            
        except Exception as e:
            logger.error(f"Error in question analysis: {e}")
            return self._fallback_analysis(question)
    
    def _validate_and_enhance_analysis(self, analysis: Dict[str, Any], question: str) -> Dict[str, Any]:
        """Validate and enhance AI analysis with additional heuristics"""
        
        # Ensure required fields exist
        required_fields = [
            'complexity', 'primary_domains', 'estimated_agent_count',
            'requires_specialization', 'interdisciplinary', 'time_sensitivity',
            'ethical_considerations', 'source_types_needed'
        ]
        
        for field in required_fields:
            if field not in analysis:
                analysis[field] = self._get_default_value(field)
        
        # Apply heuristic enhancements
        question_lower = question.lower()
        
        # Complexity heuristics
        complexity_indicators = {
            'simple': ['what is', 'define', 'explain simply'],
            'moderate': ['how does', 'why does', 'what are the effects'],
            'complex': ['analyze', 'evaluate', 'compare', 'impact of'],
            'expert': ['comprehensive analysis', 'deep dive', 'research'],
            'massive': ['global impact', 'systematic review', 'meta-analysis']
        }
        
        for complexity_level, indicators in complexity_indicators.items():
            if any(indicator in question_lower for indicator in indicators):
                if analysis['complexity'] in ['simple', 'moderate'] and complexity_level in ['complex', 'expert', 'massive']:
                    analysis['complexity'] = complexity_level
        
        # Domain detection enhancements
        domain_keywords = {
            'technology': ['ai', 'artificial intelligence', 'tech', 'digital', 'software', 'algorithm', 'automation'],
            'science': ['research', 'study', 'scientific', 'evidence', 'hypothesis', 'experiment'],
            'business': ['market', 'economy', 'company', 'corporate', 'business', 'industry', 'profit'],
            'politics': ['government', 'policy', 'political', 'legislation', 'regulation', 'election'],
            'healthcare': ['health', 'medical', 'hospital', 'disease', 'treatment', 'patient'],
            'environment': ['climate', 'environmental', 'sustainability', 'carbon', 'green', 'pollution'],
            'finance': ['financial', 'banking', 'investment', 'economic', 'money', 'currency'],
            'education': ['education', 'learning', 'school', 'university', 'teaching', 'academic'],
            'legal': ['law', 'legal', 'court', 'regulation', 'compliance', 'rights'],
            'social_issues': ['social', 'inequality', 'justice', 'community', 'society', 'demographic']
        }
        
        detected_domains = []
        for domain, keywords in domain_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                if domain not in analysis['primary_domains']:
                    detected_domains.append(domain)
        
        analysis['primary_domains'].extend(detected_domains)
        analysis['primary_domains'] = list(set(analysis['primary_domains']))  # Remove duplicates
        
        # Ethical considerations enhancements
        ethical_keywords = {
            'bias': ['bias', 'discrimination', 'prejudice', 'unfair', 'inequality'],
            'privacy': ['privacy', 'personal data', 'surveillance', 'tracking', 'confidential'],
            'misinformation': ['misinformation', 'fake news', 'disinformation', 'propaganda', 'false'],
            'harm': ['harm', 'safety', 'risk', 'danger', 'threat', 'negative impact']
        }
        
        for consideration, keywords in ethical_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                if consideration not in analysis['ethical_considerations']:
                    analysis['ethical_considerations'].append(consideration)
        
        # Adjust agent count based on complexity and domains
        if len(analysis['primary_domains']) > 3 and analysis['estimated_agent_count'] < 3:
            analysis['estimated_agent_count'] = min(len(analysis['primary_domains']), 5)
        
        return analysis
    
    def _get_default_value(self, field: str) -> Any:
        """Get default value for missing analysis fields"""
        defaults = {
            'complexity': 'moderate',
            'primary_domains': ['technology'],
            'estimated_agent_count': 2,
            'requires_specialization': True,
            'interdisciplinary': False,
            'time_sensitivity': 'medium',
            'ethical_considerations': ['bias'],
            'source_types_needed': ['web', 'academic', 'news'],
            'research_approaches': ['systematic'],
            'potential_controversies': [],
            'key_entities': [],
            'geographic_scope': 'global',
            'temporal_scope': 'current'
        }
        return defaults.get(field, None)
    
    def _fallback_analysis(self, question: str) -> Dict[str, Any]:
        """Provide fallback analysis when AI analysis fails"""
        
        logger.warning("Using fallback analysis for research question")
        
        # Basic keyword analysis
        question_lower = question.lower()
        
        # Determine complexity from question structure
        complexity = 'simple'
        if any(word in question_lower for word in ['analyze', 'evaluate', 'impact', 'effect']):
            complexity = 'moderate'
        if any(word in question_lower for word in ['comprehensive', 'deep', 'systematic']):
            complexity = 'complex'
        
        # Detect domains
        domains = []
        if any(word in question_lower for word in ['tech', 'ai', 'digital']):
            domains.append('technology')
        if any(word in question_lower for word in ['health', 'medical']):
            domains.append('healthcare')
        if any(word in question_lower for word in ['business', 'market', 'economy']):
            domains.append('business')
        if any(word in question_lower for word in ['climate', 'environment']):
            domains.append('environment')
        
        if not domains:
            domains = ['technology', 'science']
        
        return {
            "complexity": complexity,
            "primary_domains": domains,
            "estimated_agent_count": len(domains),
            "requires_specialization": True,
            "interdisciplinary": len(domains) > 1,
            "time_sensitivity": "medium",
            "ethical_considerations": ["bias", "misinformation"],
            "source_types_needed": ["academic", "news", "web"],
            "research_approaches": ["systematic", "analytical"],
            "potential_controversies": ["method validity", "source reliability"],
            "key_entities": [],
            "geographic_scope": "global",
            "temporal_scope": "current"
        }
    
    async def _generate_sub_questions(self, primary_question: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate sub-questions using AI with domain expertise"""
        
        try:
            # Create research-focused prompt
            sub_question_prompt = await self.prompt_manager.generate_system_prompt(
                user_id="research_planner",
                session_id="sub_questions",
                prompt_type=PromptType.RESEARCH_MODE,
                context=PromptContext.RESEARCH_SESSION,
                custom_variables={
                    'task': 'sub_question_generation',
                    'primary_question': primary_question,
                    'domains': analysis.get('primary_domains', []),
                    'complexity': analysis.get('complexity', 'moderate'),
                    'interdisciplinary': analysis.get('interdisciplinary', False)
                }
            )
            
            # Detailed sub-question generation prompt
            full_prompt = f"""{sub_question_prompt}

Generate comprehensive sub-questions for this research topic:

Primary Question: "{primary_question}"

Context Analysis:
- Complexity: {analysis.get('complexity', 'moderate')}
- Primary Domains: {', '.join(analysis.get('primary_domains', []))}
- Interdisciplinary: {analysis.get('interdisciplinary', False)}
- Geographic Scope: {analysis.get('geographic_scope', 'global')}
- Time Sensitivity: {analysis.get('time_sensitivity', 'medium')}

Generate 8-12 specific, actionable sub-questions that:
1. Break down the primary question into researchable components
2. Cover different perspectives and stakeholder viewpoints
3. Address methodology and evidence quality
4. Consider ethical implications and potential biases
5. Explore both current state and future implications
6. Include domain-specific technical aspects
7. Consider interdisciplinary connections if relevant

Format as a JSON array of strings:
["Sub-question 1", "Sub-question 2", ...]

Each sub-question should:
- Be specific and actionable
- Be answerable through research
- Contribute unique value to the overall investigation
- Be appropriately scoped for the complexity level
"""
            
            # Generate sub-questions using AI
            model = await self.model_loader.get_model("research_planning")
            response = await model.generate_completion(
                prompt=full_prompt,
                max_tokens=1200,
                temperature=0.4
            )
            
            # Parse JSON response
            try:
                sub_questions = json.loads(response.strip())
                
                if isinstance(sub_questions, list) and len(sub_questions) > 0:
                    # Validate and enhance sub-questions
                    validated_questions = self._validate_sub_questions(sub_questions, primary_question, analysis)
                    logger.info(f"Generated {len(validated_questions)} sub-questions")
                    return validated_questions
                else:
                    raise ValueError("Invalid sub-questions format")
                
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse sub-questions JSON: {e}")
                return self._generate_fallback_sub_questions(primary_question, analysis)
            
        except Exception as e:
            logger.error(f"Error generating sub-questions: {e}")
            return self._generate_fallback_sub_questions(primary_question, analysis)
    
    def _validate_sub_questions(self, sub_questions: List[str], primary_question: str, analysis: Dict[str, Any]) -> List[str]:
        """Validate and enhance generated sub-questions"""
        
        validated = []
        
        for question in sub_questions:
            # Basic validation
            if isinstance(question, str) and len(question.strip()) > 10:
                # Clean up the question
                clean_question = question.strip()
                if not clean_question.endswith('?'):
                    clean_question += '?'
                
                # Avoid duplicates
                if clean_question not in validated:
                    validated.append(clean_question)
        
        # Ensure minimum number of questions based on complexity
        min_questions = {
            'simple': 4,
            'moderate': 6,
            'complex': 8,
            'expert': 10,
            'massive': 12
        }
        
        complexity = analysis.get('complexity', 'moderate')
        target_count = min_questions.get(complexity, 6)
        
        # Add domain-specific questions if needed
        if len(validated) < target_count:
            additional = self._generate_domain_specific_questions(
                primary_question, 
                analysis.get('primary_domains', []),
                target_count - len(validated)
            )
            validated.extend(additional)
        
        return validated[:target_count]
    
    def _generate_fallback_sub_questions(self, primary_question: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate fallback sub-questions when AI generation fails"""
        
        logger.warning("Using fallback sub-question generation")
        
        base_questions = [
            f"What are the current trends and developments in {primary_question.lower()}?",
            f"What are the main challenges and opportunities related to {primary_question.lower()}?",
            f"What do experts and recent research say about {primary_question.lower()}?",
            f"What are the ethical and societal implications of {primary_question.lower()}?",
            f"What are the potential future developments and predictions?",
            f"Who are the key stakeholders and what are their perspectives?",
            f"What evidence exists to support different viewpoints on this topic?",
            f"What are the potential risks and benefits to consider?"
        ]
        
        # Add domain-specific questions
        domains = analysis.get('primary_domains', [])
        if domains:
            for domain in domains[:3]:  # Limit to first 3 domains
                base_questions.append(f"How does {domain} specifically relate to {primary_question.lower()}?")
        
        return base_questions[:10]
    
    def _generate_domain_specific_questions(self, primary_question: str, domains: List[str], needed_count: int) -> List[str]:
        """Generate domain-specific sub-questions"""
        
        domain_templates = {
            'technology': [
                f"What technological solutions exist for {primary_question.lower()}?",
                f"What are the technical challenges and limitations?",
                f"How might emerging technologies impact this area?"
            ],
            'science': [
                f"What does the scientific research reveal about {primary_question.lower()}?",
                f"What methodologies are used to study this topic?",
                f"What gaps exist in current scientific understanding?"
            ],
            'business': [
                f"What are the economic implications and market dynamics?",
                f"How do businesses and industries approach this challenge?",
                f"What are the competitive advantages and business models?"
            ],
            'politics': [
                f"What policies and regulations currently address this issue?",
                f"How do different political perspectives approach this topic?",
                f"What are the governance and policy recommendations?"
            ],
            'healthcare': [
                f"What are the health implications and medical considerations?",
                f"How do healthcare systems address related challenges?",
                f"What are the patient outcomes and quality of care issues?"
            ],
            'environment': [
                f"What are the environmental impacts and sustainability concerns?",
                f"How does this relate to climate change and conservation?",
                f"What are the long-term ecological implications?"
            ]
        }
        
        additional_questions = []
        for domain in domains:
            if domain in domain_templates and len(additional_questions) < needed_count:
                for template in domain_templates[domain]:
                    if len(additional_questions) < needed_count:
                        additional_questions.append(template)
        
        return additional_questions[:needed_count]
    
    async def _determine_research_goals(self, primary_question: str, analysis: Dict[str, Any]) -> List[str]:
        """Determine research goals based on question analysis"""
        
        try:
            # Generate comprehensive research goals using AI
            goals_prompt = await self.prompt_manager.generate_system_prompt(
                user_id="research_planner",
                session_id="goals",
                prompt_type=PromptType.RESEARCH_MODE,
                context=PromptContext.GOAL_SETTING,
                custom_variables={
                    'primary_question': primary_question,
                    'complexity': analysis.get('complexity', 'moderate'),
                    'domains': analysis.get('primary_domains', []),
                    'ethical_considerations': analysis.get('ethical_considerations', [])
                }
            )
            
            full_prompt = f"""{goals_prompt}

Define comprehensive research goals for this investigation:

Primary Question: "{primary_question}"

Context:
- Complexity: {analysis.get('complexity', 'moderate')}
- Domains: {', '.join(analysis.get('primary_domains', []))}
- Ethical Considerations: {', '.join(analysis.get('ethical_considerations', []))}
- Interdisciplinary: {analysis.get('interdisciplinary', False)}

Generate 6-10 specific, measurable research goals in JSON array format:
["Goal 1", "Goal 2", ...]

Goals should cover:
1. Information gathering and synthesis objectives
2. Quality and credibility standards
3. Perspective diversity and stakeholder representation
4. Bias detection and mitigation strategies
5. Evidence evaluation and methodology assessment
6. Ethical considerations and responsible research practices
7. Actionable insights and recommendation development
8. Knowledge gap identification

Each goal should be:
- Specific and measurable
- Achievable through systematic research
- Relevant to the primary question
- Time-bound where appropriate
"""
            
            model = await self.model_loader.get_model("research_planning")
            response = await model.generate_completion(
                prompt=full_prompt,
                max_tokens=800,
                temperature=0.3
            )
            
            try:
                goals = json.loads(response.strip())
                if isinstance(goals, list) and len(goals) > 0:
                    validated_goals = self._validate_research_goals(goals, analysis)
                    logger.info(f"Generated {len(validated_goals)} research goals")
                    return validated_goals
                else:
                    raise ValueError("Invalid goals format")
            
            except (json.JSONDecodeError, ValueError) as e:
                logger.error(f"Failed to parse research goals JSON: {e}")
                return self._generate_fallback_goals(primary_question, analysis)
        
        except Exception as e:
            logger.error(f"Error generating research goals: {e}")
            return self._generate_fallback_goals(primary_question, analysis)
    
    def _validate_research_goals(self, goals: List[str], analysis: Dict[str, Any]) -> List[str]:
        """Validate and enhance research goals"""
        
        validated = []
        essential_covered = {
            'overview': False,
            'credibility': False,
            'perspectives': False,
            'ethics': False
        }
        
        for goal in goals:
            if isinstance(goal, str) and len(goal.strip()) > 15:
                clean_goal = goal.strip()
                validated.append(clean_goal)
                
                # Track coverage of essential areas
                goal_lower = clean_goal.lower()
                if any(word in goal_lower for word in ['overview', 'comprehensive', 'understand']):
                    essential_covered['overview'] = True
                if any(word in goal_lower for word in ['credibility', 'reliability', 'quality']):
                    essential_covered['credibility'] = True
                if any(word in goal_lower for word in ['perspective', 'stakeholder', 'viewpoint']):
                    essential_covered['perspectives'] = True
                if any(word in goal_lower for word in ['ethical', 'bias', 'responsible']):
                    essential_covered['ethics'] = True
        
        # Add missing essential goals
        if not essential_covered['overview']:
            validated.insert(0, "Provide a comprehensive overview and understanding of the topic")
        if not essential_covered['credibility']:
            validated.append("Evaluate source credibility and evidence quality")
        if not essential_covered['perspectives']:
            validated.append("Capture diverse stakeholder perspectives and viewpoints")
        if not essential_covered['ethics']:
            validated.append("Address ethical considerations and identify potential biases")
        
        return validated[:10]
    
    def _generate_fallback_goals(self, primary_question: str, analysis: Dict[str, Any]) -> List[str]:
        """Generate fallback research goals"""
        
        logger.warning("Using fallback research goals generation")
        
        base_goals = [
            "Provide comprehensive overview of the topic",
            "Identify key stakeholders and perspectives",
            "Analyze current state and trends",
            "Evaluate evidence quality and credibility",
            "Address ethical implications and considerations",
            "Identify contradictions and resolve conflicts",
            "Develop actionable insights and recommendations"
        ]
        
        # Add complexity-specific goals
        complexity = analysis.get('complexity', 'moderate')
        if complexity in ['complex', 'expert', 'massive']:
            base_goals.extend([
                "Conduct deep domain expertise analysis",
                "Provide synthesis across multiple disciplines",
                "Identify knowledge gaps and future research directions"
            ])
        
        # Add domain-specific goals
        if analysis.get('interdisciplinary', False):
            base_goals.append("Synthesize insights across multiple disciplines")
        
        return base_goals[:8]
    
    async def _generate_collaboration_plan(self,
                                         research_plan: ResearchPlan,
                                         analysis: Dict[str, Any],
                                         max_agents: int,
                                         complexity: ResearchComplexity,
                                         user_id: str) -> CollaborationPlan:
        """Generate collaboration plan for multi-AI research"""
        
        logger.info(f"Generating collaboration plan for {max_agents} agents with {complexity.value} complexity")
        
        collaboration_id = f"collab_{research_plan.plan_id}"
        
        # Determine collaboration mode based on complexity
        mode = {
            ResearchComplexity.SIMPLE: CollaborationMode.SOLO,
            ResearchComplexity.MODERATE: CollaborationMode.PARALLEL,
            ResearchComplexity.COMPLEX: CollaborationMode.COLLABORATIVE,
            ResearchComplexity.EXPERT: CollaborationMode.HIERARCHICAL,
            ResearchComplexity.MASSIVE: CollaborationMode.DEMOCRATIC
        }.get(complexity, CollaborationMode.PARALLEL)
        
        # Get available agents from collaboration hub
        available_agents = await self.collaboration_hub.get_available_agents()
        
        # Select agents based on domain expertise
        selected_agents = await self._select_agents_for_domains(
            analysis.get('primary_domains', []),
            available_agents,
            max_agents
        )
        
        # Create domain assignments
        domain_assignments = await self._create_domain_assignments(
            analysis.get('primary_domains', []),
            research_plan.sub_questions,
            selected_agents,
            complexity
        )
        
        # Set up coordination schedule
        coordination_schedule = self._create_coordination_schedule(mode, len(selected_agents))
        
        # Create synthesis plan
        synthesis_plan = {
            'method': 'collaborative_synthesis',
            'lead_synthesizer': selected_agents[0].profile.agent_id if selected_agents else None,
            'review_cycles': 2 if complexity in [ResearchComplexity.EXPERT, ResearchComplexity.MASSIVE] else 1,
            'consensus_threshold': 0.8,
            'artifact_creation': True,
            'document_formats': ['html', 'markdown', 'json'],
            'synthesis_stages': [
                'individual_summaries',
                'cross_validation',
                'conflict_resolution',
                'final_synthesis',
                'document_creation'
            ]
        }
        
        collaboration_plan = CollaborationPlan(
            collaboration_id=collaboration_id,
            mode=mode,
            participating_agents=[agent.profile.agent_id for agent in selected_agents],
            domain_assignments=domain_assignments,
            lead_agent_id=selected_agents[0].profile.agent_id if selected_agents else None,
            coordination_schedule=coordination_schedule,
            synthesis_plan=synthesis_plan,
            communication_channels=[f"research_channel_{collaboration_id}"],
            peer_review_enabled=self.config.enable_peer_review,
            quality_thresholds={
                'min_credibility': self.config.credibility_threshold,
                'max_bias': self.config.bias_tolerance,
                'min_sources_per_domain': self.config.min_sources_per_domain,
                'consensus_threshold': 0.8,
                'review_completeness': 0.9
            }
        )
        
        # Set phase deadlines
        now = datetime.now(timezone.utc)
        time_per_domain = self.config.default_time_per_domain_hours
        collaboration_plan.phase_deadlines = {
            ResearchPhase.DOMAIN_ASSIGNMENT.value: now + timedelta(minutes=30),
            ResearchPhase.PARALLEL_RESEARCH.value: now + timedelta(hours=time_per_domain),
            ResearchPhase.SYNTHESIS_PREPARATION.value: now + timedelta(hours=time_per_domain + 1),
            ResearchPhase.COLLABORATIVE_SYNTHESIS.value: now + timedelta(hours=time_per_domain + 2),
            ResearchPhase.DOCUMENT_CREATION.value: now + timedelta(hours=time_per_domain + 3),
            ResearchPhase.FINALIZATION.value: now + timedelta(hours=time_per_domain + 4)
        }
        
        logger.info(f"Generated collaboration plan with {len(selected_agents)} agents in {mode.value} mode")
        return collaboration_plan
    
    async def _select_agents_for_domains(self,
                                       domains: List[str],
                                       available_agents: List[Any],
                                       max_agents: int) -> List[Any]:
        """Select best agents for domain expertise requirements"""
        
        if not available_agents:
            logger.warning("No available agents for domain selection")
            return []
        
        selected = []
        domain_coverage = {}
        
        # First pass: Select agents with direct domain expertise
        for domain in domains:
            if len(selected) >= max_agents:
                break
            
            best_agent = None
            best_score = 0
            
            for agent in available_agents:
                if agent in selected:
                    continue
                
                # Calculate domain expertise score
                score = self._calculate_domain_expertise_score(agent, domain)
                
                if score > best_score:
                    best_score = score
                    best_agent = agent
            
            if best_agent and best_score > 0.5:  # Minimum competency threshold
                selected.append(best_agent)
                domain_coverage[domain] = best_agent.profile.agent_id
                logger.info(f"Selected agent {best_agent.profile.agent_id} for {domain} domain (score: {best_score:.2f})")
        
        # Second pass: Fill remaining slots with generalist agents
        while len(selected) < max_agents and len(selected) < len(available_agents):
            best_generalist = None
            best_generalist_score = 0
            
            for agent in available_agents:
                if agent in selected:
                    continue
                
                # Calculate general research capability score
                score = self._calculate_general_research_score(agent)
                
                if score > best_generalist_score:
                    best_generalist_score = score
                    best_generalist = agent
            
            if best_generalist:
                selected.append(best_generalist)
                logger.info(f"Selected generalist agent {best_generalist.profile.agent_id} (score: {best_generalist_score:.2f})")
            else:
                break
        
        return selected
    
    def _calculate_domain_expertise_score(self, agent: Any, domain: str) -> float:
        """Calculate agent's expertise score for a specific domain"""
        
        try:
            # Check agent capabilities for domain relevance
            capabilities = getattr(agent.profile, 'capabilities', [])
            
            domain_mappings = {
                'technology': ['AI_RESEARCH', 'TECHNICAL_ANALYSIS', 'SOFTWARE_DEVELOPMENT'],
                'science': ['RESEARCH_ANALYSIS', 'DATA_ANALYSIS', 'ACADEMIC_RESEARCH'],
                'business': ['BUSINESS_ANALYSIS', 'MARKET_RESEARCH', 'STRATEGY'],
                'politics': ['POLICY_ANALYSIS', 'POLITICAL_RESEARCH', 'GOVERNANCE'],
                'healthcare': ['MEDICAL_RESEARCH', 'HEALTH_ANALYSIS', 'CLINICAL_RESEARCH'],
                'finance': ['FINANCIAL_ANALYSIS', 'ECONOMIC_RESEARCH', 'INVESTMENT'],
                'legal': ['LEGAL_RESEARCH', 'COMPLIANCE', 'REGULATORY_ANALYSIS'],
                'education': ['EDUCATIONAL_RESEARCH', 'CURRICULUM_ANALYSIS', 'PEDAGOGY'],
                'environment': ['ENVIRONMENTAL_RESEARCH', 'CLIMATE_ANALYSIS', 'SUSTAINABILITY']
            }
            
            relevant_capabilities = domain_mappings.get(domain, [])
            
            score = 0.0
            for capability in capabilities:
                if hasattr(capability, 'name') and capability.name in relevant_capabilities:
                    score += 0.3
                elif hasattr(capability, 'name') and 'RESEARCH' in capability.name:
                    score += 0.1
            
            # Add base research competency
            if any(hasattr(cap, 'name') and 'RESEARCH' in cap.name for cap in capabilities):
                score += 0.4
            
            return min(score, 1.0)
        
        except Exception as e:
            logger.error(f"Error calculating domain expertise score: {e}")
            return 0.5  # Default moderate score
    
    def _calculate_general_research_score(self, agent: Any) -> float:
        """Calculate agent's general research capability score"""
        
        try:
            capabilities = getattr(agent.profile, 'capabilities', [])
            
            research_capabilities = [
                'RESEARCH_ANALYSIS', 'DATA_ANALYSIS', 'CRITICAL_THINKING',
                'SYNTHESIS', 'WRITING', 'FACT_CHECKING'
            ]
            
            score = 0.0
            for capability in capabilities:
                if hasattr(capability, 'name'):
                    if capability.name in research_capabilities:
                        score += 0.2
                    elif 'RESEARCH' in capability.name or 'ANALYSIS' in capability.name:
                        score += 0.1
            
            return min(score, 1.0)
        
        except Exception as e:
            logger.error(f"Error calculating general research score: {e}")
            return 0.6  # Default moderate score
    
    async def _create_domain_assignments(self,
                                       domains: List[str],
                                       sub_questions: List[str],
                                       selected_agents: List[Any],
                                       complexity: ResearchComplexity) -> Dict[str, List[str]]:
        """Create domain and sub-question assignments for agents"""
        
        if not selected_agents:
            return {}
        
        assignments = {}
        
        # Initialize agent assignments
        for agent in selected_agents:
            assignments[agent.profile.agent_id] = []
        
        agent_ids = [agent.profile.agent_id for agent in selected_agents]
        
        # Assign domains based on expertise
        for i, domain in enumerate(domains):
            assigned_agent = agent_ids[i % len(agent_ids)]
            if domain not in assignments[assigned_agent]:
                assignments[assigned_agent].append(domain)
        
        # Assign sub-questions
        for i, question in enumerate(sub_questions):
            assigned_agent = agent_ids[i % len(agent_ids)]
            question_key = f"sub_question_{i+1}"
            assignments[assigned_agent].append(question_key)
        
        # Balance workload for complex research
        if complexity in [ResearchComplexity.EXPERT, ResearchComplexity.MASSIVE]:
            assignments = self._balance_agent_workload(assignments, selected_agents)
        
        logger.info(f"Created domain assignments for {len(selected_agents)} agents")
        return assignments
    
    def _balance_agent_workload(self, assignments: Dict[str, List[str]], agents: List[Any]) -> Dict[str, List[str]]:
        """Balance workload across agents for complex research"""
        
        # Calculate current workload
        workloads = {agent_id: len(tasks) for agent_id, tasks in assignments.items()}
        
        # Redistribute if imbalanced
        max_load = max(workloads.values()) if workloads else 0
        min_load = min(workloads.values()) if workloads else 0
        
        if max_load - min_load > 2:  # Significant imbalance
            all_tasks = []
            for agent_id, tasks in assignments.items():
                all_tasks.extend(tasks)
            
            # Redistribute evenly
            assignments = {}
            agent_ids = list(workloads.keys())
            
            for agent_id in agent_ids:
                assignments[agent_id] = []
            
            for i, task in enumerate(all_tasks):
                assigned_agent = agent_ids[i % len(agent_ids)]
                assignments[assigned_agent].append(task)
        
        return assignments
    
    def _create_coordination_schedule(self, mode: CollaborationMode, agent_count: int) -> List[Dict[str, Any]]:
        """Create coordination schedule based on collaboration mode"""
        
        now = datetime.now(timezone.utc)
        schedule = []
        
        if mode == CollaborationMode.PARALLEL:
            # Minimal coordination for parallel work
            schedule = [
                {
                    'type': 'kickoff_sync',
                    'time': (now + timedelta(minutes=15)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 15,
                    'purpose': 'Coordinate research approach and avoid overlap'
                },
                {
                    'type': 'midpoint_check',
                    'time': (now + timedelta(hours=1)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 10,
                    'purpose': 'Share preliminary findings and adjust focus'
                },
                {
                    'type': 'synthesis_planning',
                    'time': (now + timedelta(hours=2)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 30,
                    'purpose': 'Plan collaborative synthesis and document creation'
                }
            ]
        
        elif mode == CollaborationMode.COLLABORATIVE:
            # More frequent coordination
            schedule = [
                {
                    'type': 'kickoff_meeting',
                    'time': (now + timedelta(minutes=30)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 20,
                    'purpose': 'Detailed planning and role assignment'
                },
                {
                    'type': 'progress_sync',
                    'time': (now + timedelta(hours=1)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 15,
                    'purpose': 'Share findings and identify collaboration opportunities'
                },
                {
                    'type': 'integration_session',
                    'time': (now + timedelta(hours=1, minutes=30)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 25,
                    'purpose': 'Integrate findings and resolve contradictions'
                },
                {
                    'type': 'synthesis_planning',
                    'time': (now + timedelta(hours=2)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 30,
                    'purpose': 'Plan final synthesis and document structure'
                }
            ]
        
        elif mode == CollaborationMode.HIERARCHICAL:
            # Lead-driven coordination
            schedule = [
                {
                    'type': 'assignment_briefing',
                    'time': (now + timedelta(minutes=15)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 20,
                    'purpose': 'Lead assigns tasks and sets expectations'
                },
                {
                    'type': 'progress_report',
                    'time': (now + timedelta(hours=1)).isoformat(),
                    'participants': 'lead_plus_representatives',
                    'duration_minutes': 15,
                    'purpose': 'Agents report progress to lead'
                },
                {
                    'type': 'lead_review',
                    'time': (now + timedelta(hours=1, minutes=30)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 25,
                    'purpose': 'Lead reviews all findings and provides direction'
                }
            ]
        
        elif mode == CollaborationMode.DEMOCRATIC:
            # Consensus-driven coordination
            schedule = [
                {
                    'type': 'democratic_planning',
                    'time': (now + timedelta(minutes=45)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 30,
                    'purpose': 'Collaborative planning and consensus building'
                },
                {
                    'type': 'consensus_check',
                    'time': (now + timedelta(hours=1, minutes=15)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 20,
                    'purpose': 'Ensure consensus on research direction'
                },
                {
                    'type': 'democratic_synthesis',
                    'time': (now + timedelta(hours=2, minutes=15)).isoformat(),
                    'participants': 'all',
                    'duration_minutes': 45,
                    'purpose': 'Collaborative synthesis with democratic decision-making'
                }
            ]
        
        return schedule
    
    async def _persist_plan(self, research_plan: ResearchPlan, collaboration_plan: Optional[CollaborationPlan]):
        """Persist plans to VM and memory systems"""
        
        # Store in VM file system for research environment submenu
        plan_file = self.vm_plans_path / f"{research_plan.plan_id}.yaml"
        
        plan_data = {
            'research_plan': research_plan.to_dict(),
            'collaboration_plan': collaboration_plan.to_dict() if collaboration_plan else None,
            'created_at': datetime.now(timezone.utc).isoformat(),
            'version': '1.0',
            'planner_config': {
                'max_agents': self.config.max_agents_per_research,
                'collaboration_mode': self.config.default_collaboration_mode.value,
                'depth_level': self.config.default_research_depth.value,
                'quality_thresholds': {
                    'credibility': self.config.credibility_threshold,
                    'bias_tolerance': self.config.bias_tolerance
                }
            }
        }
        
        try:
            with open(plan_file, 'w') as f:
                yaml.dump(plan_data, f, default_flow_style=False, allow_unicode=True)
            
            # Also save as JSON for programmatic access
            json_file = self.vm_plans_path / f"{research_plan.plan_id}.json"
            with open(json_file, 'w') as f:
                json.dump(plan_data, f, indent=2, default=str, ensure_ascii=False)
            
            logger.info(f"Persisted research plan to VM: {plan_file}")
            
        except Exception as e:
            logger.error(f"Failed to persist plan to VM: {e}")
        
        # Store in memory system
        if self.config.store_plans_in_memory:
            try:
                await self.memory_manager.store_memory(
                    user_id="research_planner",
                    content=f"Research Plan: {research_plan.primary_question}",
                    memory_type=MemoryType.KNOWLEDGE,
                    importance=MemoryImportance.HIGH,
                    scope=MemoryScope.SYSTEM,
                    metadata={
                        'type': 'research_plan',
                        'plan_id': research_plan.plan_id,
                        'collaboration_enabled': collaboration_plan is not None,
                        'agent_count': len(collaboration_plan.participating_agents) if collaboration_plan else 1,
                        'primary_question': research_plan.primary_question,
                        'sub_questions_count': len(research_plan.sub_questions),
                        'depth_level': research_plan.depth_level.value,
                        'estimated_duration': self._estimate_research_duration(research_plan, collaboration_plan)
                    }
                )
                
                logger.info(f"Stored research plan in memory: {research_plan.plan_id}")
                
            except Exception as e:
                logger.error(f"Failed to store plan in memory: {e}")
    
    async def _initialize_default_personalities(self):
        """Initialize default research personalities"""
        
        default_personalities = [
            ResearchPersonality(
                personality_id="systematic_researcher",
                name="Systematic Researcher",
                research_approach="systematic",
                bias_tolerance=0.2,
                depth_preference=ResearchDepthLevel.DEEP,
                source_preferences=["academic", "primary"],
                ethical_guidelines={"thoroughness": "high", "accuracy": "critical"},
                collaboration_style="supportive",
                risk_tolerance=0.3,
                time_management="thorough",
                reporting_frequency="periodic",
                detail_level="detailed"
            ),
            ResearchPersonality(
                personality_id="creative_explorer",
                name="Creative Explorer",
                research_approach="creative",
                bias_tolerance=0.4,
                depth_preference=ResearchDepthLevel.MODERATE,
                source_preferences=["diverse", "unconventional"],
                ethical_guidelines={"innovation": "high", "inclusivity": "critical"},
                collaboration_style="challenging",
                risk_tolerance=0.7,
                time_management="efficient",
                reporting_frequency="continuous",
                detail_level="comprehensive"
            ),
            ResearchPersonality(
                personality_id="analytical_specialist",
                name="Analytical Specialist",
                research_approach="analytical",
                bias_tolerance=0.1,
                depth_preference=ResearchDepthLevel.EXHAUSTIVE,
                source_preferences=["academic", "peer_reviewed"],
                ethical_guidelines={"precision": "critical", "objectivity": "high"},
                collaboration_style="independent",
                risk_tolerance=0.2,
                time_management="thorough",
                reporting_frequency="milestone",
                detail_level="comprehensive"
            ),
            ResearchPersonality(
                personality_id="collaborative_synthesizer",
                name="Collaborative Synthesizer",
                research_approach="integrative",
                bias_tolerance=0.3,
                depth_preference=ResearchDepthLevel.DEEP,
                source_preferences=["multidisciplinary", "diverse"],
                ethical_guidelines={"collaboration": "critical", "consensus": "high"},
                collaboration_style="leadership",
                risk_tolerance=0.4,
                time_management="balanced",
                reporting_frequency="periodic",
                detail_level="detailed"
            ),
            ResearchPersonality(
                personality_id="rapid_scout",
                name="Rapid Scout",
                research_approach="exploratory",
                bias_tolerance=0.5,
                depth_preference=ResearchDepthLevel.SURFACE,
                source_preferences=["current", "trending"],
                ethical_guidelines={"speed": "high", "coverage": "critical"},
                collaboration_style="supportive",
                risk_tolerance=0.6,
                time_management="efficient",
                reporting_frequency="continuous",
                detail_level="summary"
            ),
            ResearchPersonality(
                personality_id="domain_specialist",
                name="Domain Specialist",
                research_approach="specialized",
                bias_tolerance=0.2,
                depth_preference=ResearchDepthLevel.EXHAUSTIVE,
                source_preferences=["domain_specific", "expert"],
                ethical_guidelines={"expertise": "critical", "accuracy": "high"},
                collaboration_style="consultative",
                risk_tolerance=0.3,
                time_management="thorough",
                reporting_frequency="milestone",
                detail_level="comprehensive"
            )
        ]
        
        for personality in default_personalities:
            self.research_personalities[personality.personality_id] = personality
            
            # Persist personality to VM
            personality_file = self.vm_personalities_path / f"{personality.personality_id}.yaml"
            try:
                with open(personality_file, 'w') as f:
                    yaml.dump(asdict(personality), f, default_flow_style=False)
            except Exception as e:
                logger.error(f"Failed to persist personality {personality.personality_id}: {e}")
        
        logger.info(f"Initialized {len(default_personalities)} default research personalities")
    
    async def assign_research_personalities(self, collaboration_plan: CollaborationPlan) -> Dict[str, str]:
        """Assign research personalities to participating agents"""
        
        personality_assignments = {}
        available_personalities = list(self.research_personalities.values())
        
        for i, agent_id in enumerate(collaboration_plan.participating_agents):
            # Assign personality based on role and preferences
            if i == 0 and collaboration_plan.lead_agent_id == agent_id:
                # Lead agent gets collaborative synthesizer personality
                personality = self.research_personalities.get("collaborative_synthesizer")
            else:
                # Rotate through other personalities
                personality_index = (i - 1) % (len(available_personalities) - 1)
                non_lead_personalities = [p for p in available_personalities 
                                        if p.personality_id != "collaborative_synthesizer"]
                personality = non_lead_personalities[personality_index] if non_lead_personalities else available_personalities[0]
            
            if personality:
                personality_assignments[agent_id] = personality.personality_id
                
                # Update agent profile through collaboration hub
                try:
                    await self.collaboration_hub.prompt_manager.update_user_profile(
                        user_id=agent_id,
                        updates={
                            'research_personality': personality.personality_id,
                            'research_approach': personality.research_approach,
                            'collaboration_style': personality.collaboration_style,
                            'preferred_response_length': personality.detail_level,
                            'technical_background': ['research', 'analysis'],
                            'personality_variables': personality.to_prompt_variables()
                        }
                    )
                    
                    logger.info(f"Assigned personality '{personality.name}' to agent {agent_id}")
                    
                except Exception as e:
                    logger.error(f"Failed to assign personality to agent {agent_id}: {e}")
        
        return personality_assignments
    
    async def initiate_collaborative_research(self, 
                                            research_plan: ResearchPlan,
                                            collaboration_plan: CollaborationPlan,
                                            user_id: str) -> Dict[str, Any]:
        """Initiate collaborative research session with multiple AI agents"""
        
        try:
            # Assign personalities to agents
            personality_assignments = await self.assign_research_personalities(collaboration_plan)
            
            # Create multi-agent task through collaboration hub
            task_description = f"""
Collaborative Research Task: {research_plan.primary_question}

Research Goals:
{chr(10).join(f"- {goal}" for goal in research_plan.research_goals)}

Sub-questions to investigate:
{chr(10).join(f"- {q}" for q in research_plan.sub_questions)}

Collaboration Mode: {collaboration_plan.mode.value}
Expected Duration: {self._estimate_research_duration(research_plan, collaboration_plan)} hours

Quality Requirements:
- Minimum {collaboration_plan.quality_thresholds.get('min_sources_per_domain', 5)} sources per domain
- Credibility threshold: {collaboration_plan.quality_thresholds.get('min_credibility', 0.7)}
- Maximum bias tolerance: {collaboration_plan.quality_thresholds.get('max_bias', 0.3)}

Each agent should focus on their assigned domain expertise while maintaining
coordination with the research team for synthesis and document creation.
"""
            
            # Create collaborative task
            task_result = await self.collaboration_hub.create_collaborative_task(
                task_id=f"research_{research_plan.plan_id}",
                task_description=task_description,
                participating_agents=collaboration_plan.participating_agents,
                lead_agent_id=collaboration_plan.lead_agent_id,
                coordination_schedule=collaboration_plan.coordination_schedule,
                quality_requirements=collaboration_plan.quality_thresholds,
                user_id=user_id
            )
            
            # Stream initiation event
            await self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.COLLABORATION_INITIATED,
                session_id=research_plan.plan_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'collaboration_id': collaboration_plan.collaboration_id,
                    'participating_agents': collaboration_plan.participating_agents,
                    'mode': collaboration_plan.mode.value,
                    'personality_assignments': personality_assignments,
                    'task_result': task_result
                },
                priority=StreamPriority.HIGH
            ))
            
            logger.info(f"Successfully initiated collaborative research for plan {research_plan.plan_id}")
            
            return {
                'success': True,
                'collaboration_id': collaboration_plan.collaboration_id,
                'participating_agents': collaboration_plan.participating_agents,
                'personality_assignments': personality_assignments,
                'coordination_schedule': collaboration_plan.coordination_schedule,
                'task_result': task_result
            }
            
        except Exception as e:
            logger.error(f"Failed to initiate collaborative research: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _estimate_research_duration(self, research_plan: ResearchPlan, collaboration_plan: Optional[CollaborationPlan]) -> float:
        """Estimate research duration in hours"""
        
        base_time = len(research_plan.sub_questions) * 0.5  # 30 minutes per sub-question
        
        # Adjust for depth
        depth_multipliers = {
            ResearchDepthLevel.SURFACE: 0.5,
            ResearchDepthLevel.MODERATE: 1.0,
            ResearchDepthLevel.DEEP: 1.5,
            ResearchDepthLevel.EXHAUSTIVE: 2.5
        }
        
        base_time *= depth_multipliers.get(research_plan.depth_level, 1.0)
        
        # Adjust for number of sources
        if research_plan.max_sources > 30:
            base_time *= 1.2
        elif research_plan.max_sources > 50:
            base_time *= 1.5
        
        # Adjust for collaboration
        if collaboration_plan:
            # Parallel work reduces time
            agent_count = len(collaboration_plan.participating_agents)
            parallel_efficiency = max(0.3, 0.7 + (0.3 / agent_count))  # Efficiency decreases with more agents
            base_time *= parallel_efficiency
            
            # Add synthesis time
            synthesis_time = base_time * self.config.synthesis_time_multiplier
            base_time += synthesis_time
            
            # Add coordination overhead
            coordination_overhead = {
                CollaborationMode.SOLO: 0.0,
                CollaborationMode.PARALLEL: 0.1,
                CollaborationMode.COLLABORATIVE: 0.2,
                CollaborationMode.HIERARCHICAL: 0.15,
                CollaborationMode.DEMOCRATIC: 0.3
            }
            base_time *= (1 + coordination_overhead.get(collaboration_plan.mode, 0.1))
        
        return round(base_time, 1)
    
    async def get_plan_status(self, plan_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of research plan"""
        
        research_plan = self.active_plans.get(plan_id)
        if not research_plan:
            return None
        
        collaboration_plan = None
        collaboration_id = f"collab_{plan_id}"
        if collaboration_id in self.collaboration_plans:
            collaboration_plan = self.collaboration_plans[collaboration_id]
        
        status = {
            'plan_id': plan_id,
            'primary_question': research_plan.primary_question,
            'created_at': research_plan.created_at.isoformat(),
            'last_modified': research_plan.last_modified.isoformat(),
            'version': research_plan.version,
            'collaboration_enabled': collaboration_plan is not None,
            'sub_questions_count': len(research_plan.sub_questions),
            'research_goals_count': len(research_plan.research_goals),
            'depth_level': research_plan.depth_level.value,
            'max_sources': research_plan.max_sources,
            'estimated_duration': self._estimate_research_duration(research_plan, collaboration_plan)
        }
        
        if collaboration_plan:
            status.update({
                'collaboration_id': collaboration_plan.collaboration_id,
                'collaboration_mode': collaboration_plan.mode.value,
                'participating_agents': collaboration_plan.participating_agents,
                'lead_agent_id': collaboration_plan.lead_agent_id,
                'coordination_schedule_count': len(collaboration_plan.coordination_schedule)
            })
        
        return status
    
    async def modify_research_plan(self, 
                                 plan_id: str,
                                 modifications: Dict[str, Any],
                                 reason: str,
                                 user_id: str) -> bool:
        """Modify existing research plan"""
        
        try:
            research_plan = self.active_plans.get(plan_id)
            if not research_plan:
                logger.error(f"Research plan {plan_id} not found")
                return False
            
            # Apply modifications
            research_plan.modify_plan(modifications, reason)
            
            # Re-persist the modified plan
            collaboration_plan = None
            collaboration_id = f"collab_{plan_id}"
            if collaboration_id in self.collaboration_plans:
                collaboration_plan = self.collaboration_plans[collaboration_id]
            
            await self._persist_plan(research_plan, collaboration_plan)
            
            # Stream modification event
            await self.stream_manager.stream_event(StreamEvent(
                event_type=StreamEventType.PLAN_MODIFIED,
                session_id=plan_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    'plan_id': plan_id,
                    'modifications': modifications,
                    'reason': reason,
                    'new_version': research_plan.version
                },
                priority=StreamPriority.HIGH
            ))
            
            logger.info(f"Modified research plan {plan_id}: {reason}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to modify research plan {plan_id}: {e}")
            return False
    
    async def load_plan_from_file(self, plan_file_path: str) -> Optional[Tuple[ResearchPlan, Optional[CollaborationPlan]]]:
        """Load research plan from YAML or JSON file"""
        
        try:
            plan_path = Path(plan_file_path)
            if not plan_path.exists():
                logger.error(f"Plan file not found: {plan_file_path}")
                return None
            
            if plan_path.suffix == '.yaml':
                with open(plan_path, 'r') as f:
                    plan_data = yaml.safe_load(f)
            elif plan_path.suffix == '.json':
                with open(plan_path, 'r') as f:
                    plan_data = json.load(f)
            else:
                logger.error(f"Unsupported file format: {plan_path.suffix}")
                return None
            
            # Reconstruct research plan
            research_plan_data = plan_data.get('research_plan')
            if not research_plan_data:
                logger.error("No research plan data found in file")
                return None
            
            research_plan = ResearchPlan(
                plan_id=research_plan_data['plan_id'],
                primary_question=research_plan_data['primary_question'],
                sub_questions=research_plan_data['sub_questions'],
                research_goals=research_plan_data['research_goals'],
                depth_level=ResearchDepthLevel(research_plan_data['depth_level']),
                max_sources=research_plan_data['max_sources'],
                trusted_domains=research_plan_data.get('trusted_domains', []),
                blocked_domains=research_plan_data.get('blocked_domains', []),
                source_types=research_plan_data.get('source_types', ["web", "academic", "news"]),
                ethical_guidelines=research_plan_data.get('ethical_guidelines', {}),
                time_constraints=research_plan_data.get('time_constraints', {})
            )
            
            # Reconstruct collaboration plan if present
            collaboration_plan = None
            collaboration_data = plan_data.get('collaboration_plan')
            if collaboration_data:
                collaboration_plan = CollaborationPlan(
                    collaboration_id=collaboration_data['collaboration_id'],
                    mode=CollaborationMode(collaboration_data['mode']),
                    participating_agents=collaboration_data['participating_agents'],
                    domain_assignments=collaboration_data['domain_assignments'],
                    lead_agent_id=collaboration_data.get('lead_agent_id'),
                    coordination_schedule=collaboration_data.get('coordination_schedule', []),
                    synthesis_plan=collaboration_data.get('synthesis_plan', {}),
                    communication_channels=collaboration_data.get('communication_channels', []),
                    peer_review_enabled=collaboration_data.get('peer_review_enabled', True),
                    quality_thresholds=collaboration_data.get('quality_thresholds', {})
                )
            
            logger.info(f"Successfully loaded plan from {plan_file_path}")
            return research_plan, collaboration_plan
            
        except Exception as e:
            logger.error(f"Error loading plan from file {plan_file_path}: {e}")
            return None
    
    def get_planner_metrics(self) -> Dict[str, Any]:
        """Get research planner metrics"""
        
        active_collaborations = len(self.collaboration_plans)
        total_agents_involved = sum(
            len(cp.participating_agents) 
            for cp in self.collaboration_plans.values()
        )
        
        return {
            'active_plans': len(self.active_plans),
            'active_collaborations': active_collaborations,
            'total_agents_involved': total_agents_involved,
            'available_personalities': len(self.research_personalities),
            'vm_plans_stored': len(list(self.vm_plans_path.glob('*.yaml'))),
            'collaboration_modes_used': {
                mode.value: sum(
                    1 for cp in self.collaboration_plans.values() 
                    if cp.mode == mode
                ) for mode in CollaborationMode
            },
            'complexity_distribution': self._get_complexity_distribution(),
            'average_agents_per_collaboration': total_agents_involved / max(active_collaborations, 1),
            'personality_usage': self._get_personality_usage_stats(),
            'plan_modification_rate': self._get_modification_rate(),
            'average_plan_duration': self._get_average_duration()
        }
    
    def _get_complexity_distribution(self) -> Dict[str, int]:
        """Get distribution of research complexities from active plans"""
        distribution = {complexity.value: 0 for complexity in ResearchComplexity}
        
        # This would analyze actual plan data - simplified for now
        for plan in self.active_plans.values():
            # Infer complexity from plan characteristics
            sub_q_count = len(plan.sub_questions)
            if sub_q_count <= 4:
                distribution['simple'] += 1
            elif sub_q_count <= 6:
                distribution['moderate'] += 1
            elif sub_q_count <= 8:
                distribution['complex'] += 1
            elif sub_q_count <= 10:
                distribution['expert'] += 1
            else:
                distribution['massive'] += 1
        
        return distribution
    
    def _get_personality_usage_stats(self) -> Dict[str, int]:
        """Get personality usage statistics"""
        usage_stats = {personality_id: 0 for personality_id in self.research_personalities.keys()}
        
        # This would track actual personality assignments - simplified for now
        for collab_plan in self.collaboration_plans.values():
            agent_count = len(collab_plan.participating_agents)
            # Distribute across available personalities
            for i, personality_id in enumerate(self.research_personalities.keys()):
                if i < agent_count:
                    usage_stats[personality_id] += 1
        
        return usage_stats
    
    def _get_modification_rate(self) -> float:
        """Get average modification rate for plans"""
        if not self.active_plans:
            return 0.0
        
        total_modifications = sum(len(plan.modification_history) for plan in self.active_plans.values())
        return total_modifications / len(self.active_plans)
    
    def _get_average_duration(self) -> float:
        """Get average estimated duration for plans"""
        if not self.active_plans:
            return 0.0
        
        total_duration = 0.0
        for plan in self.active_plans.values():
            collab_plan = self.collaboration_plans.get(f"collab_{plan.plan_id}")
            total_duration += self._estimate_research_duration(plan, collab_plan)
        
        return total_duration / len(self.active_plans)


# Factory function for easy integration
async def create_research_planner(
    memory_manager: MemoryManager,
    model_loader: ModelLoader,
    prompt_manager: SystemPromptManager,
    collaboration_hub: AgentCollaborationHub,
    stream_manager: ResearchStreamManager,
    config: Optional[ResearchPlannerConfig] = None
) -> DeepResearchPlanner:
    """Create and initialize deep research planner"""
    
    planner = DeepResearchPlanner(
        memory_manager=memory_manager,
        model_loader=model_loader,
        prompt_manager=prompt_manager,
        collaboration_hub=collaboration_hub,
        stream_manager=stream_manager,
        config=config
    )
    
    logger.info("Deep Research Planner created successfully")
    return planner