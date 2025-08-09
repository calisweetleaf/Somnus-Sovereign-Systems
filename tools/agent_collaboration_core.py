import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Callable, cast
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from pydantic import BaseModel, Field

from recursive_tensor_template import RecursiveTensor
from core.memory_core import MemoryManager, MemoryType, MemoryImportance
from core.model_loader import ModelLoader, ModelConfiguration
from core.session_manager import SessionID, UserID
from schemas.models import ModelLoadRequest

logger = logging.getLogger(__name__)

# A placeholder for UserID and SessionID if not available in context
DEFAULT_USER_ID = cast(UserID, uuid4()) # Cast to satisfy type hints
DEFAULT_SESSION_ID = cast(SessionID, uuid4()) # Cast to satisfy type hints


class MessageType(str, Enum):
    """Types of inter-agent messages"""
    TASK_PROPOSAL = "task_proposal"
    TASK_ACCEPTANCE = "task_acceptance"
    TASK_DELEGATION = "task_delegation"
    WORK_PROGRESS = "work_progress"
    WORK_RESULT = "work_result"
    CLARIFICATION_REQUEST = "clarification_request"
    SYNTHESIS_REQUEST = "synthesis_request"
    FINAL_RESPONSE = "final_response"

class AgentMessage(BaseModel):
    """Inter-agent communication message"""
    message_id: UUID = Field(default_factory=uuid4)
    sender_id: UUID
    recipient_id: Optional[UUID] = None  # None for broadcast
    message_type: MessageType
    content: Dict[str, Any]
    requires_response: bool = False
    response_timeout: Optional[float] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    context_refs: List[str] = Field(default_factory=list)

class AgentCapability(str, Enum):
    """Agent capabilities for task assignment"""
    TEXT_ANALYSIS = "text_analysis"
    CODE_GENERATION = "code_generation"  
    RESEARCH_SYNTHESIS = "research_synthesis"
    CREATIVE_WRITING = "creative_writing"
    LOGICAL_REASONING = "logical_reasoning"
    DATA_PROCESSING = "data_processing"
    TECHNICAL_DOCUMENTATION = "technical_documentation"
    PROBLEM_SOLVING = "problem_solving"

@dataclass
class AgentProfile:
    """Agent profile with capabilities and preferences"""
    agent_id: UUID
    name: str
    model_id: str
    capabilities: List[AgentCapability]
    specialization_score: Dict[AgentCapability, float] = field(default_factory=dict)
    collaboration_preference: float = 0.8  # How much this agent prefers to collaborate
    independence_threshold: float = 0.6  # Threshold for working independently
    communication_style: str = "direct"  # direct, analytical, creative

class TaskDelegationDecision(BaseModel):
    """Decision on how to handle a task"""
    task_id: UUID
    decision_type: str  # "solo", "collaborate", "delegate"
    primary_agent: UUID
    collaborating_agents: List[UUID] = Field(default_factory=list)
    task_breakdown: Dict[str, Any] = Field(default_factory=dict)
    confidence: float
    reasoning: str

class CollaborativeAgent:
    """Individual AI agent capable of direct communication and collaboration"""
    
    def __init__(
        self,
        profile: AgentProfile,
        model_loader: ModelLoader,
        memory_manager: MemoryManager,
        message_bus: 'MessageBus',
        user_id: Optional[UserID] = None, # Added for memory operations
        session_id: Optional[SessionID] = None # Added for memory operations
    ):
        self.profile = profile
        self.model_loader = model_loader
        self.memory_manager = memory_manager
        self.message_bus = message_bus
        self.user_id = user_id or DEFAULT_USER_ID
        self.session_id = session_id or DEFAULT_SESSION_ID
        
        # Communication
        self.inbox = asyncio.Queue()
        self.pending_responses = {}
        self.active_conversations = {}
        
        # Task management
        self.current_tasks = {}
        self.completed_tasks = {}
        self.task_lock = asyncio.Lock()
        
        # Model state
        self.loaded_model = None
        self.model_ready = False
        
        # Collaboration state
        self.collaboration_sessions = {}
        self.shared_workspace = {}
        
        logger.info(f"Collaborative agent initialized: {profile.name} ({profile.agent_id})")
    
    async def initialize(self):
        """Initialize agent with model loading"""
        try:
            # Load the agent's model
            load_request = ModelLoadRequest(
                model_id=self.profile.model_id,
                # Assuming default device and quantization if not specified
                # These might come from agent_config or a global config
                # device=None, 
                # quantization_override=None 
            )
            
            load_response = await self.model_loader.load_model(load_request)
            
            if load_response.success:
                self.loaded_model = self.model_loader.get_loaded_model(self.profile.model_id)
                self.model_ready = True
                logger.info(f"Agent {self.profile.name} model loaded successfully")
            else:
                logger.error(f"Failed to load model for agent {self.profile.name}: {load_response.message}")
                
            # Register with message bus
            await self.message_bus.register_agent(self)
            
            # Start message processing
            asyncio.create_task(self._process_messages())
            
        except Exception as e:
            logger.error(f"Agent initialization failed: {e}")
            raise
    
    async def process_user_input(self, user_input: str, context: Dict[str, Any]) -> str:
        """Process user input, potentially collaborating with other agents"""
        if not self.model_ready:
            logger.warning(f"Agent {self.profile.name} model not ready. Cannot process input.")
            return "Agent model not ready. Please try again shortly."
        
        # Analyze if collaboration would be beneficial
        collaboration_decision = await self._analyze_collaboration_need(user_input, context)
        
        if collaboration_decision['should_collaborate']:
            logger.info(f"Agent {self.profile.name} decided to collaborate for task: {user_input[:50]}...")
            return await self._handle_collaborative_task(user_input, context, collaboration_decision)
        else:
            logger.info(f"Agent {self.profile.name} decided to handle task solo: {user_input[:50]}...")
            return await self._handle_solo_task(user_input, context)
    
    async def _analyze_collaboration_need(self, user_input: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze whether the task would benefit from collaboration using more detailed heuristics."""
        user_lower = user_input.lower()
        task_type = self._classify_task_type(user_input)

        # Enhanced indicators
        collaboration_keywords = [
            'compare and contrast', 'multiple perspectives', 'brainstorm', 'joint effort',
            'validate findings', 'cross-check', 'integrate', 'synthesize from various sources'
        ]
        complexity_keywords = [
            'multi-step', 'intricate', 'ambiguous', 'ill-defined', 'system design',
            'strategic plan', 'policy development', 'in-depth investigation'
        ]

        collaboration_score = sum(1 for keyword in collaboration_keywords if keyword in user_lower)
        complexity_score = sum(1 for keyword in complexity_keywords if keyword in user_lower)
        
        # Length-based complexity (more nuanced)
        length_score = 0
        if len(user_input) > 1000: length_score = 1.0
        elif len(user_input) > 500: length_score = 0.7
        elif len(user_input) > 200: length_score = 0.4

        # Task type influence
        task_type_collaboration_boost = 0
        if task_type in ['research', 'creative', 'problem_solving']: # Assuming problem_solving is a type
            task_type_collaboration_boost = 0.2

        # Agent's own preference and independence threshold
        # If agent strongly prefers solo work and task isn't overwhelmingly complex
        if self.profile.collaboration_preference < 0.3 and self.profile.independence_threshold > 0.7:
            if collaboration_score == 0 and complexity_score < 2 and length_score < 0.7:
                 return {
                    'should_collaborate': False,
                    'reasoning': "Agent prefers solo work and task complexity is low."
                }

        # Weighted score
        total_score = (
            collaboration_score * 0.35 +
            complexity_score * 0.30 +
            length_score * 0.25 +
            task_type_collaboration_boost * 0.10
        )
        
        # Dynamic threshold based on agent's collaboration preference
        collaboration_threshold = 0.5 * (1 - self.profile.collaboration_preference + 0.5) # Normalize preference effect

        should_collaborate = total_score > collaboration_threshold
        
        reasoning = (f"Collaboration score: {collaboration_score}, Complexity score: {complexity_score}, "
                     f"Length score: {length_score:.2f}, Task type boost: {task_type_collaboration_boost:.2f}. "
                     f"Total score: {total_score:.2f} vs Threshold: {collaboration_threshold:.2f}")
        
        logger.debug(f"Collaboration analysis for '{user_input[:30]}...': {reasoning}, Should collaborate: {should_collaborate}")

        return {
            'should_collaborate': should_collaborate,
            'collaboration_score': collaboration_score,
            'complexity_score': complexity_score,
            'total_score': total_score,
            'reasoning': reasoning
        }
    
    async def _handle_collaborative_task(self, user_input: str, context: Dict[str, Any], decision: Dict[str, Any]) -> str:
        """Handle task that requires collaboration by selecting appropriate agents and managing the process."""
        task_id = uuid4()
        
        available_agents = await self.message_bus.get_available_agents(exclude=[self.profile.agent_id])
        
        if not available_agents:
            logger.warning(f"Agent {self.profile.name}: No collaborators available. Handling solo.")
            return await self._handle_solo_task(user_input, context)
        
        # Select best collaborator(s) based on capabilities and task
        # For simplicity, we'll pick one best collaborator here. Multi-collaborator selection is more complex.
        collaborator = await self._select_best_collaborator(user_input, available_agents)
        
        if not collaborator:
            logger.warning(f"Agent {self.profile.name}: Could not find a suitable collaborator. Handling solo.")
            return await self._handle_solo_task(user_input, context)
        
        logger.info(f"Agent {self.profile.name} selected {collaborator.profile.name} for collaboration on task {task_id}.")
        
        collaboration_session = await self._initiate_collaboration(
            task_id, user_input, context, collaborator
        )
        
        final_response = await self._wait_for_collaboration_completion(collaboration_session)
        
        return final_response
    
    async def _select_best_collaborator(self, user_input: str, available_agents: List['CollaborativeAgent']) -> Optional['CollaborativeAgent']:
        """Select the best collaborator by matching task needs with agent capabilities and load."""
        if not available_agents:
            return None

        task_requirements = self._infer_task_capabilities(user_input)
        best_agent = None
        highest_score = -1

        for agent in available_agents:
            score = 0
            # Match capabilities
            for req_cap in task_requirements:
                if req_cap in agent.profile.capabilities:
                    score += agent.profile.specialization_score.get(req_cap, 0.5)  # Use specialization score
            
            # Consider agent load (lower is better)
            load_factor = 1.0 / (1.0 + len(agent.current_tasks)) 
            score *= load_factor
            
            # Consider collaboration preference
            score *= agent.profile.collaboration_preference

            if score > highest_score:
                highest_score = score
                best_agent = agent
        
        if best_agent:
            logger.info(f"Selected collaborator: {best_agent.profile.name} with score {highest_score:.2f} for task: {user_input[:30]}...")
        else:
            logger.warning(f"No suitable collaborator found among {len(available_agents)} agents for task: {user_input[:30]}...")
            # Fallback: could pick a generalist or least loaded if no strong match
            if available_agents: # Pick the least loaded one as a fallback
                best_agent = min(available_agents, key=lambda ag: len(ag.current_tasks))
                logger.info(f"Fallback: selected least loaded agent {best_agent.profile.name}")


        return best_agent

    def _infer_task_capabilities(self, user_input: str) -> List[AgentCapability]:
        """Infer required capabilities from user input. More sophisticated NLP/classification can be used here."""
        user_lower = user_input.lower()
        inferred_caps = set()
        
        # Simple keyword to capability mapping
        mapping = {
            AgentCapability.TEXT_ANALYSIS: ['analyze', 'summarize', 'extract', 'sentiment'],
            AgentCapability.CODE_GENERATION: ['code', 'script', 'program', 'function', 'develop software'],
            AgentCapability.RESEARCH_SYNTHESIS: ['research', 'find information', 'literature review', 'investigate'],
            AgentCapability.CREATIVE_WRITING: ['story', 'poem', 'write creatively', 'narrative', 'dialogue'],
            AgentCapability.LOGICAL_REASONING: ['solve puzzle', 'deduce', 'reason', 'logical problem'],
            AgentCapability.DATA_PROCESSING: ['process data', 'dataset', 'csv', 'excel', 'transform data'],
            AgentCapability.TECHNICAL_DOCUMENTATION: ['document', 'manual', 'guide', 'technical writing'],
            AgentCapability.PROBLEM_SOLVING: ['solve', 'problem', 'issue', 'troubleshoot', 'solution']
        }

        for cap, keywords in mapping.items():
            if any(keyword in user_lower for keyword in keywords):
                inferred_caps.add(cap)
        
        if not inferred_caps: # Default if no specific capability is inferred
            inferred_caps.add(AgentCapability.PROBLEM_SOLVING) # General problem solving
            
        return list(inferred_caps)

    async def _initiate_collaboration(self, task_id: UUID, user_input: str, context: Dict[str, Any], collaborator: 'CollaborativeAgent') -> Dict[str, Any]:
        """Initiate collaboration session with another agent, proposing a dynamic task split."""
        session_data = {
            'session_id': uuid4(),
            'task_id': task_id,
            'participants': [self.profile.agent_id, collaborator.profile.agent_id],
            'initiator': self.profile.agent_id,
            'user_input': user_input,
            'context': context,
            'status': 'pending_acceptance', # Initial status
            'messages': [],
            'start_time': datetime.now(timezone.utc),
            'results': [] # Initialize results list
        }
        
        self.collaboration_sessions[session_data['session_id']] = session_data
        
        proposed_split = await self._propose_task_split(user_input, self.profile, collaborator.profile)
        
        proposal_message = AgentMessage(
            sender_id=self.profile.agent_id,
            recipient_id=collaborator.profile.agent_id,
            message_type=MessageType.TASK_PROPOSAL,
            content={
                'session_id': session_data['session_id'],
                'task_id': task_id,
                'user_input': user_input,
                'context': context,
                'proposed_split': proposed_split,
                'collaboration_type': 'focused_contribution' # Example type
            },
            requires_response=True,
            response_timeout=60.0 # Increased timeout
        )
        
        await self.message_bus.send_message(proposal_message)
        session_data['messages'].append(proposal_message)
        logger.info(f"Collaboration proposal sent to {collaborator.profile.name} for session {session_data['session_id']}")
        
        return session_data
    
    async def _propose_task_split(self, user_input: str, initiator_profile: AgentProfile, collaborator_profile: AgentProfile) -> Dict[str, Any]:
        """Propose a dynamic task split based on input and agent capabilities."""
        # This is still a simplified version. A production system might use an LLM for complex task decomposition.
        initiator_caps = set(initiator_profile.capabilities)
        collaborator_caps = set(collaborator_profile.capabilities)
        
        task_reqs = set(self._infer_task_capabilities(user_input))
        
        initiator_focus_parts = []
        collaborator_focus_parts = []

        # Assign tasks based on unique strengths or primary capabilities
        for req in task_reqs:
            if req in initiator_caps and req not in collaborator_caps:
                initiator_focus_parts.append(f"Lead on {req.value}")
            elif req in collaborator_caps and req not in initiator_caps:
                collaborator_focus_parts.append(f"Lead on {req.value}")
            elif req in initiator_caps and req in collaborator_caps: # Shared capability
                # Simplistic split: initiator takes first, collaborator takes second if multiple shared
                # Or, could be based on specialization scores
                if initiator_profile.specialization_score.get(req, 0) >= collaborator_profile.specialization_score.get(req, 0):
                    initiator_focus_parts.append(f"Handle {req.value} aspects")
                else:
                    collaborator_focus_parts.append(f"Handle {req.value} aspects")
            else: # Requirement not directly met by either, assign to initiator by default or flag
                initiator_focus_parts.append(f"Address unassigned requirement: {req.value}")


        if not initiator_focus_parts and collaborator_focus_parts: # If initiator has no specific focus, assign general overview
             initiator_focus_parts.append("Overall task coordination and initial framing.")
        elif not collaborator_focus_parts and initiator_focus_parts:
             collaborator_focus_parts.append("Validation and alternative perspective.")


        # Default if no specific parts assigned
        initiator_focus = ", ".join(initiator_focus_parts) if initiator_focus_parts else "Primary analysis and structuring."
        collaborator_focus = ", ".join(collaborator_focus_parts) if collaborator_focus_parts else "Detailed analysis and verification."

        return {
            'initiator_agent_focus': initiator_focus,
            'collaborator_agent_focus': collaborator_focus,
            'synthesis_approach': 'Integrate contributions, resolve conflicts, and produce a unified response.',
            'estimated_time_per_agent_minutes': 5 # A rough estimate
        }
    
    async def _wait_for_collaboration_completion(self, session: Dict[str, Any]) -> str:
        """Wait for collaboration to complete and return final response."""
        session_id = session['session_id']
        # Timeout should be configurable, potentially based on task complexity
        timeout_seconds = session['context'].get('collaboration_timeout', 300) # 5 minutes default
        start_time = time.monotonic()
        
        logger.info(f"Agent {self.profile.name} waiting for collaboration session {session_id} to complete.")

        while time.monotonic() - start_time < timeout_seconds:
            current_session_state = self.collaboration_sessions.get(session_id)
            if not current_session_state:
                logger.error(f"Collaboration session {session_id} disappeared unexpectedly.")
                return await self._handle_solo_task(session['user_input'], session['context'])

            status = current_session_state.get('status')
            
            if status == 'completed':
                logger.info(f"Collaboration session {session_id} completed successfully.")
                return current_session_state.get('final_response', 'Collaboration completed, but no final response was generated.')
            
            if status == 'failed' or status == 'rejected':
                reason = current_session_state.get('failure_reason', 'Collaboration was not successful.')
                logger.warning(f"Collaboration session {session_id} failed or was rejected: {reason}")
                # Fallback to solo task, possibly with information gathered so far
                return await self._handle_solo_task(session['user_input'], session['context'])
            
            await asyncio.sleep(0.5) # Check status periodically
        
        logger.warning(f"Collaboration session {session_id} timed out after {timeout_seconds}s.")
        self.collaboration_sessions[session_id]['status'] = 'failed'
        self.collaboration_sessions[session_id]['failure_reason'] = 'Timeout'
        return await self._handle_solo_task(session['user_input'], session['context'])
    
    async def _handle_solo_task(self, user_input: str, context: Dict[str, Any]) -> str:
        """Handle task independently using the agent's loaded model."""
        if not self.model_ready or not self.loaded_model:
            logger.error(f"Agent {self.profile.name} model not ready for solo task.")
            return "Agent model not ready for processing."
        
        try:
            # Construct a prompt for the model
            # This prompt engineering can be quite sophisticated
            prompt = (
                f"User query: {user_input}\n\n"
                f"Agent Profile: Name: {self.profile.name}, Capabilities: {', '.join(c.value for c in self.profile.capabilities)}\n"
                f"Task Context: {json.dumps(context, indent=2, default=str)}\n\n"
                f"Please provide a comprehensive response to the user query based on your capabilities. "
                f"Structure your response clearly."
            )

            logger.info(f"Agent {self.profile.name} generating solo response for: {user_input[:50]}...")
            # Assuming the loaded model has an async generate method
            # The actual method name and signature might vary based on ModelLoader's implementation
            if hasattr(self.loaded_model, 'generate') and asyncio.iscoroutinefunction(self.loaded_model.generate):
                model_response_content = await self.loaded_model.generate(prompt=prompt, context=context)
            elif hasattr(self.loaded_model, 'generate_response') and asyncio.iscoroutinefunction(self.loaded_model.generate_response): # Alternative common name
                 model_response_content = await self.loaded_model.generate_response(prompt=prompt, context=context)
            else:
                # Fallback for synchronous or differently named methods - adapt as needed
                # This part is highly dependent on the actual model object's interface
                logger.warning(f"Model for agent {self.profile.name} does not have an async 'generate' or 'generate_response' method. Attempting synchronous call if available or placeholder.")
                if hasattr(self.loaded_model, 'generate'):
                    model_response_content = self.loaded_model.generate(prompt=prompt, context=context)
                else:
                    model_response_content = f"Model {self.profile.model_id} does not have a standard generation method."
                    logger.error(model_response_content)


            # Store this interaction in memory
            await self.memory_manager.add_memory(
                user_id=self.user_id, 
                session_id=self.session_id,
                content=f"User Input: {user_input}\nAgent Response: {model_response_content}",
                memory_type=MemoryType.CONVERSATION_HISTORY, # Assuming this enum value exists
                importance=MemoryImportance.MEDIUM, # Assuming this enum value exists
                metadata={'agent_id': str(self.profile.agent_id), 'task_type': 'solo', 'model_id': self.profile.model_id}
            )
            
            return model_response_content
            
        except Exception as e:
            logger.exception(f"Agent {self.profile.name} failed during solo task processing: {e}")
            return f"Error processing task independently: {str(e)}"
    
    def _classify_task_type(self, user_input: str) -> str:
        """Classify the type of task based on input"""
        user_lower = user_input.lower()
        
        if any(word in user_lower for word in ['code', 'program', 'function', 'script']):
            return 'coding'
        elif any(word in user_lower for word in ['research', 'analyze', 'study', 'investigate']):
            return 'research'
        elif any(word in user_lower for word in ['write', 'create', 'compose', 'draft']):
            return 'creative'
        elif any(word in user_lower for word in ['explain', 'describe', 'what is', 'how does']):
            return 'explanatory'
        else:
            return 'general'
    
    async def _process_messages(self):
        """Process incoming messages from other agents"""
        while True:
            try:
                # Get message from inbox
                message = await self.inbox.get()
                
                # Handle different message types
                if message.message_type == MessageType.TASK_PROPOSAL:
                    await self._handle_task_proposal(message)
                elif message.message_type == MessageType.TASK_ACCEPTANCE:
                    await self._handle_task_acceptance(message)
                elif message.message_type == MessageType.WORK_PROGRESS:
                    await self._handle_work_progress(message)
                elif message.message_type == MessageType.WORK_RESULT:
                    await self._handle_work_result(message)
                elif message.message_type == MessageType.SYNTHESIS_REQUEST:
                    await self._handle_synthesis_request(message)
                
                # Mark message as processed
                self.inbox.task_done()
                
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _handle_task_proposal(self, message: AgentMessage):
        """Handle collaboration proposal from another agent"""
        try:
            content = message.content
            session_id = content.get('session_id')
            user_input = content.get('user_input')
            
            if not session_id or not user_input:
                logger.error(f"Invalid task proposal received by {self.profile.name}: missing session_id or user_input.")
                # Optionally send a decline message if sender_id is known
                return

            # Update local collaboration session state if it's for this agent
            if session_id not in self.collaboration_sessions:
                 self.collaboration_sessions[session_id] = { # Create a basic session entry
                    'session_id': session_id,
                    'task_id': content.get('task_id'),
                    'participants': [message.sender_id, self.profile.agent_id], # Assuming this agent is the recipient
                    'initiator': message.sender_id,
                    'user_input': user_input,
                    'context': content.get('context', {}),
                    'status': 'received_proposal',
                    'messages': [message],
                    'results': [],
                    'start_time': message.created_at 
                }
            else: # If session already exists (e.g. initiated by this agent, which shouldn't happen for proposal)
                 self.collaboration_sessions[session_id]['messages'].append(message)


            accept_collaboration, reason = await self._evaluate_collaboration_proposal(message)
            
            response_content = {
                'session_id': session_id,
                'accepted': accept_collaboration,
            }

            if accept_collaboration:
                self.collaboration_sessions[session_id]['status'] = 'accepted'
                my_focus = content.get('proposed_split', {}).get('collaborator_agent_focus', 'General contribution')
                response_content['my_focus'] = my_focus
                response_content['estimated_completion_minutes'] = 5 # Example
                
                logger.info(f"Agent {self.profile.name} accepted collaboration proposal for session {session_id} from agent {message.sender_id}.")
                
                # Send acceptance
                acceptance_message = AgentMessage(
                    sender_id=self.profile.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.TASK_ACCEPTANCE,
                    content=response_content
                )
                await self.message_bus.send_message(acceptance_message)
                self.collaboration_sessions[session_id]['messages'].append(acceptance_message)
                
                # Start working on the assigned part
                asyncio.create_task(self._start_collaborative_work(content, my_focus))
            else:
                self.collaboration_sessions[session_id]['status'] = 'rejected'
                self.collaboration_sessions[session_id]['failure_reason'] = reason
                response_content['reason'] = reason
                logger.info(f"Agent {self.profile.name} declined collaboration proposal for session {session_id}: {reason}")
                
                # Send decline message
                decline_message = AgentMessage(
                    sender_id=self.profile.agent_id,
                    recipient_id=message.sender_id,
                    message_type=MessageType.TASK_ACCEPTANCE, # Still an acceptance message, but with accepted: False
                    content=response_content
                )
                await self.message_bus.send_message(decline_message)
                self.collaboration_sessions[session_id]['messages'].append(decline_message)

        except Exception as e:
            logger.exception(f"Agent {self.profile.name} error handling task proposal: {e}")
            # Attempt to notify proposer of failure if possible
            if message and message.sender_id and content and content.get('session_id'):
                try:
                    error_response = AgentMessage(
                        sender_id=self.profile.agent_id,
                        recipient_id=message.sender_id,
                        message_type=MessageType.TASK_ACCEPTANCE,
                        content={'session_id': content.get('session_id'), 'accepted': False, 'reason': f"Internal error processing proposal: {str(e)}"}
                    )
                    await self.message_bus.send_message(error_response)
                except Exception as send_error:
                    logger.error(f"Failed to send error notification for task proposal: {send_error}")
    
    async def _evaluate_collaboration_proposal(self, message: AgentMessage) -> Tuple[bool, str]:
        """Evaluate whether to accept a collaboration proposal based on capabilities, load, and task suitability."""
        content = message.content
        user_input = content.get('user_input', '')
        proposed_split = content.get('proposed_split', {})
        my_proposed_focus = proposed_split.get('collaborator_agent_focus', '')

        # 1. Check agent's general availability and preference
        if len(self.current_tasks) >= 5: # Max concurrent tasks limit
            return False, "Agent is at maximum task capacity."
        if self.profile.collaboration_preference < 0.2: # Agent strongly prefers solo work
            return False, "Agent preference set against collaboration at this time."

        # 2. Assess task relevance to capabilities
        required_capabilities = self._infer_task_capabilities(user_input)
        # Check if my_proposed_focus aligns with my capabilities or if general task aligns
        relevant_to_my_focus = False
        if my_proposed_focus:
            # A simple check: does my focus string mention any of my capabilities?
            # More advanced: parse my_proposed_focus to understand required skills for it.
            for cap in self.profile.capabilities:
                if cap.value.lower().replace("_", " ") in my_proposed_focus.lower():
                    relevant_to_my_focus = True
                    break
        
        has_relevant_capability = any(cap in self.profile.capabilities for cap in required_capabilities)

        if not has_relevant_capability and not relevant_to_my_focus:
            return False, "Task does not align with agent's core capabilities."

        # 3. Consider complexity vs. benefit (simple heuristic)
        # If task is very simple and I'm not specialized, collaboration might be overhead.
        # This logic can be expanded significantly.
        
    async def _start_collaborative_work(self, proposal_content: Dict[str, Any], my_focus: str):
        """Start working on assigned portion of collaborative task"""
        try:
            session_id = proposal_content.get('session_id')
            user_input = proposal_content.get('user_input')
            context = proposal_content.get('context', {})
            
            if not session_id or not user_input:
                logger.error(f"Invalid collaboration work data for agent {self.profile.name}")
                return
            
            # Add task to current tasks
            task_id = uuid4()
            self.current_tasks[task_id] = {
                'type': 'collaboration',
                'session_id': session_id,
                'focus': my_focus,
                'start_time': datetime.now(timezone.utc)
            }
            
            # Generate response based on my focus area
            focused_prompt = (
                f"Collaborative Task Focus: {my_focus}\n"
                f"User Query: {user_input}\n"
                f"Context: {json.dumps(context, indent=2, default=str)}\n\n"
                f"Provide a focused response addressing specifically your assigned area: {my_focus}"
            )
            
            my_contribution = await self._generate_model_response(focused_prompt, context)
            
            # Send work result back to initiator
            work_result = AgentMessage(
                sender_id=self.profile.agent_id,
                recipient_id=proposal_content.get('initiator_id'),
                message_type=MessageType.WORK_RESULT,
                content={
                    'session_id': session_id,
                    'task_id': task_id,
                    'agent_focus': my_focus,
                    'contribution': my_contribution,
                    'confidence': 0.8,
                    'completion_time': datetime.now(timezone.utc).isoformat()
                }
            )
            
            await self.message_bus.send_message(work_result)
            
            # Move task to completed
            self.completed_tasks[task_id] = self.current_tasks.pop(task_id)
            self.completed_tasks[task_id]['end_time'] = datetime.now(timezone.utc)
            self.completed_tasks[task_id]['result'] = my_contribution
            
            logger.info(f"Agent {self.profile.name} completed collaborative work for session {session_id}")
            
        except Exception as e:
            logger.exception(f"Error in collaborative work for agent {self.profile.name}: {e}")

    async def _generate_model_response(self, prompt: str, context: Dict[str, Any]) -> str:
        """Generate response using the loaded model"""
        try:
            if not self.model_ready or not self.loaded_model:
                return "Model not available for response generation"
            
            # Attempt various model interfaces
            if hasattr(self.loaded_model, 'generate') and callable(self.loaded_model.generate):
                if asyncio.iscoroutinefunction(self.loaded_model.generate):
                    return await self.loaded_model.generate(prompt=prompt, context=context)
                else:
                    return self.loaded_model.generate(prompt=prompt, context=context)
            elif hasattr(self.loaded_model, 'generate_response') and callable(self.loaded_model.generate_response):
                if asyncio.iscoroutinefunction(self.loaded_model.generate_response):
                    return await self.loaded_model.generate_response(prompt=prompt, context=context)
                else:
                    return self.loaded_model.generate_response(prompt=prompt, context=context)
            elif hasattr(self.loaded_model, '__call__') and callable(self.loaded_model):
                return self.loaded_model(prompt)
            else:
                logger.error(f"Model {self.profile.model_id} does not have a recognized generation interface")
                return f"Unable to generate response: Model interface not recognized"
                
        except Exception as e:
            logger.exception(f"Error generating model response: {e}")
            return f"Error generating response: {str(e)}"

    async def _handle_task_acceptance(self, message: AgentMessage):
        """Handle task acceptance response from collaborator"""
        try:
            content = message.content
            session_id = content.get('session_id')
            accepted = content.get('accepted', False)
            
            if session_id not in self.collaboration_sessions:
                logger.warning(f"Received acceptance for unknown session {session_id}")
                return
            
            session = self.collaboration_sessions[session_id]
            session['messages'].append(message)
            
            if accepted:
                session['status'] = 'in_progress'
                session['collaborator_focus'] = content.get('my_focus', 'General contribution')
                logger.info(f"Collaboration accepted for session {session_id}")
                
                # Start my own work on the task
                await self._start_my_collaborative_work(session)
            else:
                session['status'] = 'rejected'
                session['failure_reason'] = content.get('reason', 'Collaboration declined')
                logger.info(f"Collaboration rejected for session {session_id}: {session['failure_reason']}")
                
        except Exception as e:
            logger.exception(f"Error handling task acceptance: {e}")

    async def _start_my_collaborative_work(self, session: Dict[str, Any]):
        """Start initiator's work on collaborative task"""
        try:
            user_input = session['user_input']
            context = session['context']
            
            # Get my focus from the original proposal
            my_focus = "Primary analysis and coordination"
            if 'proposed_split' in session.get('messages', [{}])[0].content:
                my_focus = session['messages'][0].content['proposed_split'].get('initiator_agent_focus', my_focus)
            
            # Generate my contribution
            my_contribution = await self._generate_model_response(
                f"Collaborative Task Focus: {my_focus}\nUser Query: {user_input}\n"
                f"Context: {json.dumps(context, indent=2, default=str)}\n\n"
                f"Provide analysis for your assigned area: {my_focus}",
                context
            )
            
            # Store my result
            session['results'].append({
                'agent_id': self.profile.agent_id,
                'agent_name': self.profile.name,
                'focus': my_focus,
                'contribution': my_contribution,
                'timestamp': datetime.now(timezone.utc).isoformat()
            })
            
            logger.info(f"Initiator {self.profile.name} completed work for session {session['session_id']}")
            
        except Exception as e:
            logger.exception(f"Error in initiator collaborative work: {e}")

    async def _handle_work_progress(self, message: AgentMessage):
        """Handle work progress updates from collaborators"""
        try:
            content = message.content
            session_id = content.get('session_id')
            
            if session_id in self.collaboration_sessions:
                self.collaboration_sessions[session_id]['messages'].append(message)
                logger.debug(f"Work progress received for session {session_id}")
                
        except Exception as e:
            logger.exception(f"Error handling work progress: {e}")

    async def _handle_work_result(self, message: AgentMessage):
        """Handle work results from collaborators"""
        try:
            content = message.content
            session_id = content.get('session_id')
            
            if session_id not in self.collaboration_sessions:
                logger.warning(f"Received work result for unknown session {session_id}")
                return
            
            session = self.collaboration_sessions[session_id]
            session['messages'].append(message)
            
            # Add collaborator's result
            session['results'].append({
                'agent_id': message.sender_id,
                'agent_name': 'Collaborator',
                'focus': content.get('agent_focus', 'Unknown'),
                'contribution': content.get('contribution', ''),
                'confidence': content.get('confidence', 0.5),
                'timestamp': content.get('completion_time', datetime.now(timezone.utc).isoformat())
            })
            
            # Check if all results are in and synthesize
            expected_contributors = len(session['participants'])
            if len(session['results']) >= expected_contributors:
                await self._synthesize_collaboration_results(session)
                
        except Exception as e:
            logger.exception(f"Error handling work result: {e}")

    async def _handle_synthesis_request(self, message: AgentMessage):
        """Handle synthesis requests from other agents"""
        try:
            content = message.content
            session_id = content.get('session_id')
            
            if session_id in self.collaboration_sessions:
                session = self.collaboration_sessions[session_id]
                await self._synthesize_collaboration_results(session)
                
        except Exception as e:
            logger.exception(f"Error handling synthesis request: {e}")

    async def _synthesize_collaboration_results(self, session: Dict[str, Any]):
        """Synthesize results from all collaborators into final response"""
        try:
            results = session.get('results', [])
            user_input = session['user_input']
            
            if not results:
                session['final_response'] = await self._handle_solo_task(user_input, session['context'])
                session['status'] = 'completed'
                return
            
            # Prepare synthesis prompt
            synthesis_prompt = f"User Query: {user_input}\n\nCollaborative Contributions:\n"
            
            for i, result in enumerate(results, 1):
                synthesis_prompt += f"\n{i}. {result['focus']} (by {result['agent_name']}):\n{result['contribution']}\n"
            
            synthesis_prompt += (
                "\nPlease synthesize these contributions into a comprehensive, unified response "
                "that addresses the user's query completely. Integrate insights, resolve any conflicts, "
                "and present a coherent final answer."
            )
            
            # Generate synthesis
            final_response = await self._generate_model_response(synthesis_prompt, session['context'])
            
            # Store final response
            session['final_response'] = final_response
            session['status'] = 'completed'
            session['completion_time'] = datetime.now(timezone.utc)
            
            # Store in memory
            await self.memory_manager.add_memory(
                user_id=self.user_id,
                session_id=self.session_id,
                content=f"Collaborative Task: {user_input}\nFinal Response: {final_response}",
                memory_type=MemoryType.CONVERSATION_HISTORY,
                importance=MemoryImportance.HIGH,
                metadata={
                    'collaboration_session_id': str(session['session_id']),
                    'participants': [str(p) for p in session['participants']],
                    'task_type': 'collaboration'
                }
            )
            
            logger.info(f"Collaboration synthesis completed for session {session['session_id']}")
            
        except Exception as e:
            logger.exception(f"Error synthesizing collaboration results: {e}")

class MessageBus:
    """Message bus for inter-agent communication"""
    
    def __init__(self):
        self.agents: Dict[UUID, CollaborativeAgent] = {}
        self.message_history: List[AgentMessage] = []
        self.message_lock = asyncio.Lock()
        self.max_history = 10000
        
    async def register_agent(self, agent: CollaborativeAgent):
        """Register an agent with the message bus"""
        self.agents[agent.profile.agent_id] = agent
        logger.info(f"Agent {agent.profile.name} registered with message bus")
    
    async def unregister_agent(self, agent_id: UUID):
        """Unregister an agent from the message bus"""
        if agent_id in self.agents:
            agent_name = self.agents[agent_id].profile.name
            del self.agents[agent_id]
            logger.info(f"Agent {agent_name} unregistered from message bus")
    
    async def send_message(self, message: AgentMessage):
        """Send message to target agent(s)"""
        async with self.message_lock:
            self.message_history.append(message)
            
            # Trim history if too long
            if len(self.message_history) > self.max_history:
                self.message_history = self.message_history[-self.max_history//2:]
        
        if message.recipient_id:
            # Direct message
            if message.recipient_id in self.agents:
                await self.agents[message.recipient_id].inbox.put(message)
            else:
                logger.warning(f"Message recipient {message.recipient_id} not found")
        else:
            # Broadcast message
            for agent in self.agents.values():
                if agent.profile.agent_id != message.sender_id:
                    await agent.inbox.put(message)
    
    async def get_available_agents(self, exclude: Optional[List[UUID]] = None) -> List[CollaborativeAgent]:
        """Get list of available agents, optionally excluding some"""
        exclude = exclude or []
        return [
            agent for agent_id, agent in self.agents.items()
            if agent_id not in exclude and agent.model_ready
        ]


class AgentCollaborationHub:
    """
    Central hub for managing collaborative AI agents with dynamic task delegation,
    inter-agent communication, and intelligent workload distribution.
    """
    
    def __init__(
        self,
        model_loader: ModelLoader,
        memory_manager: MemoryManager,
        max_agents: int = 10,
        user_id: Optional[UserID] = None,
        session_id: Optional[SessionID] = None
    ):
        self.model_loader = model_loader
        self.memory_manager = memory_manager
        self.max_agents = max_agents
        self.user_id = user_id or DEFAULT_USER_ID
        self.session_id = session_id or DEFAULT_SESSION_ID
        
        # Agent management
        self.agents: Dict[UUID, CollaborativeAgent] = {}
        self.message_bus = MessageBus()
        self.active_collaborations: Dict[UUID, Dict[str, Any]] = {}
        
        # Performance tracking
        self.task_metrics: Dict[str, Any] = {
            'total_tasks': 0,
            'collaborative_tasks': 0,
            'solo_tasks': 0,
            'average_response_time': 0.0,
            'success_rate': 1.0
        }
        
        # State management
        self.hub_ready = False
        self.executor = ThreadPoolExecutor(max_workers=4)
        
        logger.info("Agent collaboration hub initialized")
    
    async def initialize(self):
        """Initialize the collaboration hub"""
        try:
            # Create default agent profiles
            default_profiles = self._create_default_agent_profiles()
            
            # Initialize agents
            for profile in default_profiles:
                await self.add_agent(profile)
            
            self.hub_ready = True
            logger.info(f"Collaboration hub ready with {len(self.agents)} agents")
            
        except Exception as e:
            logger.error(f"Failed to initialize collaboration hub: {e}")
            raise
    
    def _create_default_agent_profiles(self) -> List[AgentProfile]:
        """Create default agent profiles with different specializations"""
        profiles = []
        
        # General purpose agent
        profiles.append(AgentProfile(
            agent_id=uuid4(),
            name="GeneralAssistant",
            model_id="default",
            capabilities=[
                AgentCapability.PROBLEM_SOLVING,
                AgentCapability.TEXT_ANALYSIS,
                AgentCapability.LOGICAL_REASONING
            ],
            specialization_score={
                AgentCapability.PROBLEM_SOLVING: 0.9,
                AgentCapability.TEXT_ANALYSIS: 0.8,
                AgentCapability.LOGICAL_REASONING: 0.8
            },
            collaboration_preference=0.7
        ))
        
        # Research specialist
        profiles.append(AgentProfile(
            agent_id=uuid4(),
            name="ResearchSpecialist",
            model_id="default",
            capabilities=[
                AgentCapability.RESEARCH_SYNTHESIS,
                AgentCapability.TEXT_ANALYSIS,
                AgentCapability.DATA_PROCESSING
            ],
            specialization_score={
                AgentCapability.RESEARCH_SYNTHESIS: 0.95,
                AgentCapability.TEXT_ANALYSIS: 0.85,
                AgentCapability.DATA_PROCESSING: 0.8
            },
            collaboration_preference=0.8
        ))
        
        # Code specialist
        profiles.append(AgentProfile(
            agent_id=uuid4(),
            name="CodeSpecialist",
            model_id="default",
            capabilities=[
                AgentCapability.CODE_GENERATION,
                AgentCapability.TECHNICAL_DOCUMENTATION,
                AgentCapability.PROBLEM_SOLVING
            ],
            specialization_score={
                AgentCapability.CODE_GENERATION: 0.95,
                AgentCapability.TECHNICAL_DOCUMENTATION: 0.9,
                AgentCapability.PROBLEM_SOLVING: 0.85
            },
            collaboration_preference=0.6
        ))
        
        return profiles
    
    async def add_agent(self, profile: AgentProfile) -> CollaborativeAgent:
        """Add a new collaborative agent to the hub"""
        if len(self.agents) >= self.max_agents:
            raise ValueError(f"Maximum number of agents ({self.max_agents}) reached")
        
        agent = CollaborativeAgent(
            profile=profile,
            model_loader=self.model_loader,
            memory_manager=self.memory_manager,
            message_bus=self.message_bus,
            user_id=self.user_id,
            session_id=self.session_id
        )
        
        await agent.initialize()
        self.agents[profile.agent_id] = agent
        
        logger.info(f"Added agent {profile.name} to collaboration hub")
        return agent
    
    async def remove_agent(self, agent_id: UUID):
        """Remove an agent from the hub"""
        if agent_id in self.agents:
            agent = self.agents[agent_id]
            await self.message_bus.unregister_agent(agent_id)
            del self.agents[agent_id]
            logger.info(f"Removed agent {agent.profile.name} from collaboration hub")
    
    async def process_user_input(self, user_input: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Process user input through the collaboration hub.
        Automatically selects the best agent(s) and manages collaboration.
        """
        if not self.hub_ready or not self.agents:
            return "Collaboration hub not ready or no agents available"
        
        context = context or {}
        start_time = time.time()
        
        try:
            # Select primary agent based on task requirements
            primary_agent = await self._select_primary_agent(user_input, context)
            
            if not primary_agent:
                return "No suitable agent available to handle this request"
            
            # Process through primary agent (which may initiate collaboration)
            response = await primary_agent.process_user_input(user_input, context)
            
            # Update metrics
            processing_time = time.time() - start_time
            await self._update_task_metrics(processing_time, True)
            
            return response
            
        except Exception as e:
            logger.exception(f"Error processing user input: {e}")
            processing_time = time.time() - start_time
            await self._update_task_metrics(processing_time, False)
            return f"Error processing request: {str(e)}"
    
    async def _select_primary_agent(self, user_input: str, context: Dict[str, Any]) -> Optional[CollaborativeAgent]:
        """Select the best primary agent for handling the user input"""
        if not self.agents:
            return None
        
        # Infer task requirements
        task_capabilities = self._infer_task_capabilities_from_input(user_input)
        
        best_agent = None
        highest_score = -1
        
        for agent in self.agents.values():
            if not agent.model_ready:
                continue
            
            score = 0
            
            # Capability matching
            for cap in task_capabilities:
                if cap in agent.profile.capabilities:
                    score += agent.profile.specialization_score.get(cap, 0.5)
            
            # Load balancing
            load_factor = 1.0 / (1.0 + len(agent.current_tasks))
            score *= load_factor
            
            # Slight preference for agents with independence
            score *= (1.0 + agent.profile.independence_threshold * 0.1)
            
            if score > highest_score:
                highest_score = score
                best_agent = agent
        
        return best_agent
    
    def _infer_task_capabilities_from_input(self, user_input: str) -> List[AgentCapability]:
        """Infer required capabilities from user input"""
        user_lower = user_input.lower()
        capabilities = []
        
        capability_keywords = {
            AgentCapability.CODE_GENERATION: ['code', 'program', 'script', 'function', 'algorithm'],
            AgentCapability.RESEARCH_SYNTHESIS: ['research', 'analyze', 'study', 'investigate', 'find'],
            AgentCapability.CREATIVE_WRITING: ['write', 'story', 'creative', 'narrative', 'poem'],
            AgentCapability.TEXT_ANALYSIS: ['analyze', 'summarize', 'extract', 'parse', 'review'],
            AgentCapability.DATA_PROCESSING: ['data', 'csv', 'excel', 'process', 'transform'],
            AgentCapability.TECHNICAL_DOCUMENTATION: ['document', 'manual', 'guide', 'documentation'],
            AgentCapability.LOGICAL_REASONING: ['solve', 'logic', 'reason', 'deduce', 'puzzle'],
            AgentCapability.PROBLEM_SOLVING: ['problem', 'issue', 'help', 'fix', 'solution']
        }
        
        for capability, keywords in capability_keywords.items():
            if any(keyword in user_lower for keyword in keywords):
                capabilities.append(capability)
        
        # Default to problem solving if no specific capability detected
        if not capabilities:
            capabilities.append(AgentCapability.PROBLEM_SOLVING)
        
        return capabilities
    
    async def _update_task_metrics(self, processing_time: float, success: bool):
        """Update task processing metrics"""
        self.task_metrics['total_tasks'] += 1
        
        if success:
            # Update average response time
            current_avg = self.task_metrics['average_response_time']
            total_tasks = self.task_metrics['total_tasks']
            new_avg = (current_avg * (total_tasks - 1) + processing_time) / total_tasks
            self.task_metrics['average_response_time'] = new_avg
        
        # Update success rate
        total_success = self.task_metrics['success_rate'] * (self.task_metrics['total_tasks'] - 1)
        if success:
            total_success += 1
        self.task_metrics['success_rate'] = total_success / self.task_metrics['total_tasks']
    
    def get_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            'total_agents': len(self.agents),
            'ready_agents': sum(1 for agent in self.agents.values() if agent.model_ready),
            'agents': []
        }
        
        for agent in self.agents.values():
            agent_status = {
                'id': str(agent.profile.agent_id),
                'name': agent.profile.name,
                'model_id': agent.profile.model_id,
                'ready': agent.model_ready,
                'capabilities': [cap.value for cap in agent.profile.capabilities],
                'active_tasks': len(agent.current_tasks),
                'collaboration_sessions': len(agent.collaboration_sessions),
                'collaboration_preference': agent.profile.collaboration_preference,
                'specialization_scores': agent.profile.specialization_score
            }
            status['agents'].append(agent_status)
        
        return status
    
    def get_collaboration_stats(self) -> Dict[str, Any]:
        """Get comprehensive collaboration statistics"""
        total_messages = len(self.message_bus.message_history)
        
        # Analyze message types
        message_types = {}
        for msg in self.message_bus.message_history:
            msg_type = msg.message_type.value
            message_types[msg_type] = message_types.get(msg_type, 0) + 1
        
        # Calculate active collaborations metrics
        active_collab_count = len(self.active_collaborations)
        active_agents_in_collab = set()
        collab_duration_stats = []
        
        for collab_id, collab_data in self.active_collaborations.items():
            active_agents_in_collab.update(collab_data.get('participants', []))
            
            start_time = collab_data.get('start_time')
            if start_time:
                duration = (datetime.now(timezone.utc) - start_time).total_seconds()
                collab_duration_stats.append(duration)
        
        # Get agent participation metrics
        agent_participation = {}
        for agent_id, agent in self.agents.items():
            agent_participation[str(agent_id)] = {
                'name': agent.profile.name,
                'total_tasks': len(agent.current_tasks) + len(agent.completed_tasks),
                'current_tasks': len(agent.current_tasks),
                'collaboration_sessions': len(agent.collaboration_sessions)
            }
        
        # Calculate message rate
        message_timeframe = 3600  # last hour
        recent_messages = [
            msg for msg in self.message_bus.message_history 
            if (datetime.now(timezone.utc) - msg.created_at).total_seconds() < message_timeframe
        ]
        message_rate = len(recent_messages) / (message_timeframe / 3600)  # messages per hour
        
        return {
            'total_messages': total_messages,
            'message_types': message_types,
            'message_rate_per_hour': message_rate,
            'active_collaborations': active_collab_count,
            'active_agents_in_collaborations': len(active_agents_in_collab),
            'average_collaboration_duration': sum(collab_duration_stats) / max(len(collab_duration_stats), 1),
            'agent_participation': agent_participation,
            'task_metrics': self.task_metrics,
            'hub_ready': self.hub_ready,
            'timestamp': datetime.now(timezone.utc).isoformat()
        }
    
    async def shutdown(self):
        """Gracefully shutdown the collaboration hub"""
        logger.info("Shutting down collaboration hub")
        
        # Wait for ongoing tasks to complete (with timeout)
        shutdown_timeout = 30.0
        start_time = time.time()
        
        while time.time() - start_time < shutdown_timeout:
            active_tasks = sum(len(agent.current_tasks) for agent in self.agents.values())
            if active_tasks == 0:
                break
            await asyncio.sleep(1.0)
        
        # Unregister all agents
        for agent_id in list(self.agents.keys()):
            await self.remove_agent(agent_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        self.hub_ready = False
        logger.info("Collaboration hub shutdown complete")
    
        # Select primary agent (for now, use first available)
        primary_agent = next(iter(self.agents.values()))
        
        # Process through primary agent (which may initiate collaboration)
        response = await primary_agent.process_user_input(user_input, context)
        
        return response
    
    def get_agents_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        status = {
            'total_agents': len(self.agents),
            'ready_agents': sum(1 for agent in self.agents.values() if agent.model_ready),
            'agents': []
        }
        
        for agent in self.agents.values():
            agent_status = {
                'id': str(agent.profile.agent_id),
                'name': agent.profile.name,
                'model_id': agent.profile.model_id,
                'ready': agent.model_ready,
                'capabilities': [cap.value for cap in agent.profile.capabilities],
                'active_tasks': len(agent.current_tasks),
                'collaboration_sessions': len(agent.collaboration_sessions)
            }
            status['agents'].append(agent_status)
        
        return status
    
def get_collaboration_stats(self) -> Dict[str, Any]:
    """Get comprehensive collaboration statistics"""
    total_messages = len(self.message_bus.message_history)
    
    # Analyze message types
    message_types = {}
    for msg in self.message_bus.message_history:
        msg_type = msg.message_type.value
        message_types[msg_type] = message_types.get(msg_type, 0) + 1
    
    # Calculate active collaborations metrics
    active_collab_count = len(self.active_collaborations)
    active_agents_in_collab = set()
    collab_duration_stats = []
    
    for collab_id, collab_data in self.active_collaborations.items():
        active_agents_in_collab.update(collab_data.get('participants', []))
        
        start_time = collab_data.get('start_time')
        if start_time:
            duration = (datetime.now(timezone.utc) - start_time).total_seconds()
            collab_duration_stats.append(duration)
    
    # Get agent participation metrics
    agent_participation = {}
    for agent_id, agent in self.agents.items():
        agent_participation[str(agent_id)] = {
            'name': agent.profile.name,
            'total_tasks': len(agent.current_tasks) + len(agent.completed_tasks),
            'current_tasks': len(agent.current_tasks),
            'collaboration_sessions': len(agent.collaboration_sessions)
        }
    
    # Calculate message rate
    message_timeframe = 3600  # last hour
    recent_messages = [
        msg for msg in self.message_bus.message_history 
        if (datetime.now(timezone.utc) - msg.created_at).total_seconds() < message_timeframe
    ]
    message_rate = len(recent_messages) / (message_timeframe / 3600)  # messages per hour
    
    return {
        'total_messages': total_messages,
        'message_types': message_types,
        'message_rate_per_hour': message_rate,
        'active_collaborations': active_collab_count,
        'active_agents_in_collaborations': len(active_agents_in_collab),
        'average_collaboration_duration': sum(collab_duration_stats) / max(len(collab_duration_stats), 1),
        'agent_participation': agent_participation,
        'hub_ready': self.hub_ready,
        'timestamp': datetime.now(timezone.utc).isoformat()
    }