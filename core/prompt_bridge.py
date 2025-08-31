"""
MORPHEUS CHAT - Prompt System Bridge
Integration bridge between modifying_prompts.py and prompt_manager.py for unified prompting.

This module implements the bridge described in TODO.md to create a unified prompt system
where different subsystems can use either adaptive prompting or identity-anchored prompting
based on their needs.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from datetime import datetime, timezone

from core.modifying_prompts import AutonomousPromptSystem, PromptEvolutionTrigger
from core.prompt_manager import PromptManager, PromptType, PromptContext, UserProfile
from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope
from backend.system_cache import SomnusCache, CacheNamespace, CachePriority
from schemas.session import PromptEventMemoryObject, PromptEventType, PromptMemoryTags

logger = logging.getLogger(__name__)


class SubsystemType(str, Enum):
    """Different subsystems that require prompting"""
    CHAT = "chat"
    PROJECTS = "projects" 
    ARTIFACTS = "artifacts"
    WEB_RESEARCH = "web_research"
    MULTI_AGENT_COLLABORATION = "multi_agent_collaboration"
    DEEP_RESEARCH = "deep_research"


class PromptSystemBridge:
    """
    Bridge between adaptive prompting (modifying_prompts.py) and identity-stabilized prompting (prompt_manager.py).
    
    Routes prompting requests to the appropriate system based on subsystem requirements:
    - Chat, Projects, Artifacts: Uses adaptive prompting (modifying_prompts.py)
    - Research, Multi-Agent: Uses identity-stabilized prompting (prompt_manager.py)
    """
    
    def __init__(self, 
                 cache: SomnusCache,
                 memory_manager: MemoryManager,
                 config: Optional[Dict[str, Any]] = None):
        
        self.cache = cache
        self.memory_manager = memory_manager
        self.config = config or {}
        
        # Initialize prompt managers
        self.prompt_manager = PromptManager(memory_manager=memory_manager, config=config)
        self.autonomous_prompt_systems: Dict[str, AutonomousPromptSystem] = {}
        
        # Subsystem routing configuration
        self.identity_stabilized_subsystems = {
            SubsystemType.WEB_RESEARCH,
            SubsystemType.MULTI_AGENT_COLLABORATION,
            SubsystemType.DEEP_RESEARCH
        }
        
        self.adaptive_subsystems = {
            SubsystemType.CHAT,
            SubsystemType.PROJECTS,
            SubsystemType.ARTIFACTS
        }
        
        logger.info("Prompt system bridge initialized")
    
    async def initialize(self):
        """Initialize the prompt bridge systems"""
        try:
            await self.prompt_manager.initialize()
            logger.info("Prompt system bridge initialization complete")
        except Exception as e:
            logger.error(f"Failed to initialize prompt system bridge: {e}")
            raise
    
    async def get_prompt_for_subsystem(self,
                                     subsystem: SubsystemType,
                                     user_id: str,
                                     user_input: str,
                                     session_id: str,
                                     context: Optional[Dict[str, Any]] = None) -> str:
        """
        Route prompt generation to appropriate system based on subsystem type.
        
        Returns the generated prompt for the specified subsystem.
        """
        try:
            if subsystem in self.identity_stabilized_subsystems:
                # Use identity-stabilized prompting for research/collaboration
                return await self._get_identity_stabilized_prompt(
                    subsystem, user_id, user_input, session_id, context
                )
            elif subsystem in self.adaptive_subsystems:
                # Use adaptive prompting for chat/projects/artifacts
                return await self._get_adaptive_prompt(
                    subsystem, user_id, user_input, session_id, context
                )
            else:
                logger.warning(f"Unknown subsystem {subsystem}, defaulting to adaptive prompting")
                return await self._get_adaptive_prompt(
                    subsystem, user_id, user_input, session_id, context
                )
                
        except Exception as e:
            logger.error(f"Failed to generate prompt for subsystem {subsystem}: {e}")
            return self._get_fallback_prompt(user_input)
    
    async def _get_identity_stabilized_prompt(self,
                                            subsystem: SubsystemType,
                                            user_id: str,
                                            user_input: str,
                                            session_id: str,
                                            context: Optional[Dict[str, Any]] = None) -> str:
        """Generate identity-stabilized prompt using prompt_manager.py"""
        
        # Map subsystem to prompt type and context
        prompt_type_mapping = {
            SubsystemType.WEB_RESEARCH: PromptType.RESEARCH_MODE,
            SubsystemType.MULTI_AGENT_COLLABORATION: PromptType.ANALYSIS_MODE,
            SubsystemType.DEEP_RESEARCH: PromptType.RESEARCH_MODE
        }
        
        prompt_context_mapping = {
            SubsystemType.WEB_RESEARCH: PromptContext.RESEARCH_SESSION,
            SubsystemType.MULTI_AGENT_COLLABORATION: PromptContext.PROBLEM_SOLVING,
            SubsystemType.DEEP_RESEARCH: PromptContext.RESEARCH_SESSION
        }
        
        prompt_type = prompt_type_mapping.get(subsystem, PromptType.BASE_SYSTEM)
        prompt_context = prompt_context_mapping.get(subsystem, PromptContext.TASK_FOCUSED)
        
        # Generate prompt using identity-stabilized system
        prompt = await self.prompt_manager.generate_system_prompt(
            user_id=user_id,
            prompt_type=prompt_type,
            context=prompt_context,
            session_id=session_id,
            user_message=user_input,
            additional_context=context or {}
        )
        
        logger.info(f"Generated identity-stabilized prompt for {subsystem} ({len(prompt)} chars)")
        return prompt
    
    async def _get_adaptive_prompt(self,
                                 subsystem: SubsystemType,
                                 user_id: str,
                                 user_input: str,
                                 session_id: str,
                                 context: Optional[Dict[str, Any]] = None) -> str:
        """Generate adaptive prompt using modifying_prompts.py"""
        
        # Get or create autonomous prompt system for this user
        autonomous_system = await self._get_or_create_autonomous_system(user_id)
        
        # Add subsystem-specific context
        task_context = context or {}
        task_context.update({
            'subsystem': subsystem.value,
            'subsystem_type': 'adaptive',
            'identity_anchored': False
        })
        
        # Generate adaptive prompt
        prompt = await autonomous_system.generate_active_response_prompt(
            user_input=user_input,
            session_id=session_id,
            task_context=task_context
        )
        
        logger.info(f"Generated adaptive prompt for {subsystem} ({len(prompt)} chars)")
        return prompt
    
    async def _get_or_create_autonomous_system(self, user_id: str) -> AutonomousPromptSystem:
        """Get or create autonomous prompt system for user"""
        if user_id not in self.autonomous_prompt_systems:
            autonomous_system = AutonomousPromptSystem(
                cache=self.cache,
                memory_manager=self.memory_manager,
                user_id=user_id,
                config=self.config
            )
            
            # Initialize the system
            await autonomous_system.initialize_with_config()
            
            self.autonomous_prompt_systems[user_id] = autonomous_system
            logger.info(f"Created autonomous prompt system for user {user_id}")
        
        return self.autonomous_prompt_systems[user_id]
    
    def _get_fallback_prompt(self, user_input: str) -> str:
        """Fallback prompt in case of system failure"""
        return f"""You are a helpful AI assistant. The user has asked: {user_input}

Please provide a helpful, accurate, and relevant response. Use your knowledge and capabilities to assist the user effectively."""
    
    async def store_prompt_event(self,
                                subsystem: SubsystemType,
                                user_id: str,
                                session_id: str,
                                user_input: str,
                                generated_prompt: str,
                                response_content: Optional[str] = None,
                                quality_score: Optional[float] = None) -> PromptEventMemoryObject:
        """
        Store prompt generation event as first-class memory object.
        
        This integrates with the memory system to ensure prompt events are indexed
        for future retrieval and learning.
        """
        
        # Determine event type based on subsystem
        event_type_mapping = {
            SubsystemType.WEB_RESEARCH: PromptEventType.MEMORY_SYNTHESIS,
            SubsystemType.MULTI_AGENT_COLLABORATION: PromptEventType.AUTONOMOUS_ADAPTATION,
            SubsystemType.DEEP_RESEARCH: PromptEventType.MEMORY_SYNTHESIS,
            SubsystemType.CHAT: PromptEventType.PROMPT_GENERATED,
            SubsystemType.PROJECTS: PromptEventType.CONTEXT_INJECTION,
            SubsystemType.ARTIFACTS: PromptEventType.PROMPT_GENERATED
        }
        
        event_type = event_type_mapping.get(subsystem, PromptEventType.PROMPT_GENERATED)
        
        # Create memory tags
        memory_tags = PromptMemoryTags(
            event_type=event_type,
            user_id=user_id,
            session_id=session_id,
            semantic_tags=[subsystem.value, "prompt_bridge", "unified_prompting"],
            technical_context=[subsystem.value, "prompt_generation"],
            collaboration_context=["system_integration", "prompt_bridge"],
            quality_score=quality_score,
            relevance_score=0.8,  # Default high relevance for prompt events
            memory_importance="high" if subsystem in self.identity_stabilized_subsystems else "medium"
        )
        
        # Create prompt event memory object
        prompt_event = PromptEventMemoryObject(
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            prompt_content=generated_prompt,
            user_input=user_input,
            response_content=response_content,
            content_hash=self._calculate_content_hash(generated_prompt),
            memory_tags=memory_tags,
            prompt_length=len(generated_prompt)
        )
        
        # Store in memory system for semantic indexing
        memory_content = f"""Prompt Bridge Event - {subsystem.value}
User Input: {user_input}
Generated Prompt Length: {len(generated_prompt)} chars
Subsystem: {subsystem.value}
Event Type: {event_type.value}
Quality Score: {quality_score or 'N/A'}"""
        
        memory_id = await self.memory_manager.store_memory(
            user_id=user_id,
            content=memory_content,
            memory_type=MemoryType.SYSTEM_EVENT,
            importance=MemoryImportance.HIGH if subsystem in self.identity_stabilized_subsystems else MemoryImportance.MEDIUM,
            scope=MemoryScope.PRIVATE,
            source_session=session_id,
            tags=[
                'prompt_bridge',
                'unified_prompting',
                subsystem.value,
                event_type.value,
                'system_integration'
            ],
            metadata={
                'prompt_event_id': str(prompt_event.event_id),
                'subsystem': subsystem.value,
                'prompt_system_type': 'identity_stabilized' if subsystem in self.identity_stabilized_subsystems else 'adaptive',
                'generation_timestamp': datetime.now(timezone.utc).isoformat(),
                'quality_score': quality_score,
                'user_input_length': len(user_input),
                'prompt_length': len(generated_prompt)
            }
        )
        
        prompt_event.mark_consolidated(str(memory_id))
        
        logger.info(f"Stored prompt event {prompt_event.event_id} for subsystem {subsystem}")
        return prompt_event
    
    def _calculate_content_hash(self, content: str) -> str:
        """Calculate SHA-256 hash of content for deduplication"""
        import hashlib
        return hashlib.sha256(content.encode()).hexdigest()
    
    async def update_performance_feedback(self,
                                        user_id: str,
                                        prompt_event_id: str,
                                        user_feedback: Optional[str] = None,
                                        task_completed: bool = True,
                                        response_quality: Optional[float] = None):
        """
        Update performance feedback for prompts from both systems.
        
        This ensures both adaptive and identity-stabilized systems learn from feedback.
        """
        try:
            # Update autonomous system if user has one
            if user_id in self.autonomous_prompt_systems:
                autonomous_system = self.autonomous_prompt_systems[user_id]
                await autonomous_system.update_response_performance(
                    user_feedback=user_feedback,
                    task_completed=task_completed,
                    session_id=prompt_event_id  # Use event ID as session marker
                )
            
            # Update prompt manager performance (if applicable)
            await self.prompt_manager.update_template_performance(
                template_feedback={
                    'user_feedback': user_feedback,
                    'task_completed': task_completed,
                    'response_quality': response_quality
                }
            )
            
            logger.info(f"Updated performance feedback for prompt event {prompt_event_id}")
            
        except Exception as e:
            logger.error(f"Failed to update performance feedback: {e}")
    
    async def get_subsystem_routing_info(self) -> Dict[str, Any]:
        """Get information about subsystem routing configuration"""
        return {
            'identity_stabilized_subsystems': [s.value for s in self.identity_stabilized_subsystems],
            'adaptive_subsystems': [s.value for s in self.adaptive_subsystems],
            'active_autonomous_systems': len(self.autonomous_prompt_systems),
            'prompt_manager_initialized': self.prompt_manager is not None,
            'bridge_status': 'operational'
        }
    
    async def shutdown(self):
        """Shutdown all prompt systems"""
        try:
            # Shutdown autonomous systems
            for user_id, autonomous_system in self.autonomous_prompt_systems.items():
                await autonomous_system.shutdown_synthesis()
                logger.info(f"Shutdown autonomous system for user {user_id}")
            
            # Clear systems
            self.autonomous_prompt_systems.clear()
            
            logger.info("Prompt system bridge shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during prompt bridge shutdown: {e}")


# Factory function for easy integration
async def create_unified_prompt_system(
    cache: SomnusCache,
    memory_manager: MemoryManager,
    config: Optional[Dict[str, Any]] = None
) -> PromptSystemBridge:
    """
    Factory function to create unified prompt system bridge.
    
    This implements the architecture described in TODO.md for unified prompting.
    """
    bridge = PromptSystemBridge(cache=cache, memory_manager=memory_manager, config=config)
    await bridge.initialize()
    
    logger.info("Created unified prompt system bridge")
    return bridge