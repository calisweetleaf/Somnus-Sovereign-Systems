"""
MORPHEUS CHAT - Advanced System Prompt Management
Dynamic, context-aware system prompt generation with memory integration and personalization.

Features:
- Dynamic prompt generation based on context and user history
- Memory-enhanced personalization
- Tool and capability-aware prompting
- Multi-modal prompt support
- A/B testing and prompt optimization
- Compliance and safety integration
"""

import asyncio
import logging
import re
import json
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

from jinja2 import Environment, FileSystemLoader, select_autoescape, Template
from pydantic import BaseModel, Field

from schemas.session import SessionID, UserID
from core.memory_manager import MemoryManager, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)


class PromptType(str, Enum):
    """Different types of system prompts"""
    BASE_SYSTEM = "base_system"
    MEMORY_ENHANCED = "memory_enhanced"
    RESEARCH_MODE = "research_mode"
    CODE_ASSISTANT = "code_assistant"
    CREATIVE_WRITING = "creative_writing"
    ANALYSIS_MODE = "analysis_mode"
    TEACHING_MODE = "teaching_mode"
    DEBUGGING_MODE = "debugging_mode"
    MULTIMODAL = "multimodal"


class PromptContext(str, Enum):
    """Context categories for prompt adaptation"""
    CONVERSATION_START = "conversation_start"
    ONGOING_CHAT = "ongoing_chat"
    TASK_FOCUSED = "task_focused"
    RESEARCH_SESSION = "research_session"
    CODING_SESSION = "coding_session"
    CREATIVE_SESSION = "creative_session"
    PROBLEM_SOLVING = "problem_solving"
    LEARNING_SESSION = "learning_session"


@dataclass
class UserProfile:
    """Comprehensive user profile for prompt personalization"""
    user_id: UserID
    name: Optional[str] = None
    expertise_level: str = "intermediate"  # beginner, intermediate, advanced, expert
    preferred_communication_style: str = "friendly"  # formal, friendly, casual, technical
    primary_use_cases: List[str] = field(default_factory=list)
    technical_background: List[str] = field(default_factory=list)
    learning_preferences: Dict[str, str] = field(default_factory=dict)
    accessibility_needs: List[str] = field(default_factory=list)
    language_preference: str = "en"
    timezone: str = "UTC"
    
    # Inferred from conversation history
    topics_of_interest: List[str] = field(default_factory=list)
    conversation_patterns: Dict[str, Any] = field(default_factory=dict)
    preferred_response_length: str = "medium"  # short, medium, long, detailed
    prefers_examples: bool = True
    prefers_step_by_step: bool = True
    
    # Updated dynamically
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class PromptTemplate(BaseModel):
    """Structured prompt template with metadata"""
    template_id: str
    template_type: PromptType
    name: str
    description: str
    template_content: str
    
    # Template variables and their types
    required_variables: Dict[str, str] = Field(default_factory=dict)
    optional_variables: Dict[str, str] = Field(default_factory=dict)
    
    # Context and usage
    suitable_contexts: List[PromptContext] = Field(default_factory=list)
    capability_requirements: List[str] = Field(default_factory=list)
    
    # Performance tracking
    usage_count: int = 0
    success_rate: float = 1.0
    avg_response_quality: float = 0.0
    
    # Metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: str = "1.0.0"
    author: str = "system"
    tags: List[str] = Field(default_factory=list)


class PromptConfiguration(BaseModel):
    """Configuration for prompt management system"""
    # Template directories
    template_directories: List[str] = Field(default_factory=lambda: ["prompts/templates"])
    
    # Memory integration
    memory_integration_enabled: bool = True
    max_memory_context_tokens: int = 1000
    memory_relevance_threshold: float = 0.7
    
    # Personalization
    personalization_enabled: bool = True
    user_profile_learning: bool = True
    adaptive_prompting: bool = True
    
    # Safety and compliance
    safety_filtering: bool = True
    content_policy_enforcement: bool = True
    bias_mitigation: bool = True
    
    # Performance optimization
    template_caching: bool = True
    cache_ttl_seconds: int = 3600
    prompt_compression: bool = False
    
    # A/B testing
    ab_testing_enabled: bool = False
    test_sample_rate: float = 0.1


class SystemPromptManager:
    """
    Advanced system prompt management with dynamic generation and personalization.
    
    Handles intelligent prompt construction based on user context, conversation history,
    available capabilities, and memory insights.
    """
    
    def __init__(self, config: PromptConfiguration, memory_manager: Optional[MemoryManager] = None):
        self.config = config
        self.memory_manager = memory_manager
        
        # Initialize template engine
        self.template_dirs = [Path(d) for d in config.template_directories]
        for template_dir in self.template_dirs:
            template_dir.mkdir(parents=True, exist_ok=True)
        
        self.jinja_env = Environment(
            loader=FileSystemLoader([str(d) for d in self.template_dirs]),
            autoescape=select_autoescape(['html', 'xml']),
            trim_blocks=True,
            lstrip_blocks=True
        )
        
        # Template storage
        self.templates: Dict[str, PromptTemplate] = {}
        self.user_profiles: Dict[UserID, UserProfile] = {}
        
        # Caching
        self.prompt_cache: Dict[str, Tuple[str, datetime]] = {}
        
        # Performance tracking
        self.prompt_analytics: Dict[str, Dict[str, Any]] = {}
        
        # Initialize default templates
        asyncio.create_task(self._initialize_default_templates())
        
        logger.info("System prompt manager initialized")
    
    async def _initialize_default_templates(self):
        """Initialize default system prompt templates"""
        default_templates = {
            "base_system": {
                "name": "Base System Prompt",
                "description": "Core system prompt for general conversation",
                "content": """You are Morpheus Chat, an advanced AI assistant with sophisticated capabilities including:

🧠 **Persistent Memory**: I remember our previous conversations and can build on past context
🔍 **Web Search & Research**: I can search the web and conduct deep research with multiple sources
📁 **File Processing**: I can analyze documents, images, and code files you upload
🎨 **Artifact Creation**: I can create interactive code, visualizations, and documents
🛡️ **Privacy-First**: All data stays local, no external tracking or data collection

**Current Context:**
- User: {{ user_name or "User" }}
- Session: {{ session_type or "General conversation" }}
- Available capabilities: {{ available_capabilities | join(", ") }}
{% if memory_context %}
- Relevant memories: {{ memory_context | length }} items from our history
{% endif %}

**Instructions:**
- Be helpful, informative, and engaging
- Use memory context to provide personalized responses
- Offer to use web search or research for current information
- Suggest creating artifacts when appropriate
- Respect user preferences and maintain conversation continuity
{% if custom_instructions %}

**User-Specific Instructions:**
{{ custom_instructions }}
{% endif %}
""",
                "template_type": PromptType.BASE_SYSTEM,
                "suitable_contexts": [PromptContext.CONVERSATION_START, PromptContext.ONGOING_CHAT],
                "required_variables": {
                    "available_capabilities": "list"
                },
                "optional_variables": {
                    "user_name": "string",
                    "session_type": "string",
                    "memory_context": "list",
                    "custom_instructions": "string"
                }
            },
            
            "memory_enhanced": {
                "name": "Memory-Enhanced Prompt",
                "description": "System prompt with rich memory context",
                "content": """You are Morpheus Chat with access to our conversation history and your persistent memory.

**About {{ user_name or "this user" }}:**
{% if user_facts %}
{% for fact in user_facts %}
- {{ fact.content }}
{% endfor %}
{% endif %}

**Relevant Context from Our History:**
{% if memory_context %}
{% for memory in memory_context %}
- {{ memory.content[:200] }}{% if memory.content|length > 200 %}...{% endif %}
  ({{ memory.memory_type }}, {{ memory.created_at.strftime('%Y-%m-%d') }})
{% endfor %}
{% endif %}

**Conversation Patterns:**
{% if conversation_patterns %}
- Preferred response style: {{ conversation_patterns.get('response_style', 'balanced') }}
- Typical topics: {{ conversation_patterns.get('common_topics', []) | join(', ') }}
- Interaction frequency: {{ conversation_patterns.get('frequency', 'regular') }}
{% endif %}

Use this context to provide personalized, continuous assistance. Reference past conversations naturally when relevant, but don't explicitly mention accessing memories unless asked.
""",
                "template_type": PromptType.MEMORY_ENHANCED,
                "suitable_contexts": [PromptContext.ONGOING_CHAT],
                "required_variables": {
                    "memory_context": "list"
                },
                "optional_variables": {
                    "user_name": "string",
                    "user_facts": "list",
                    "conversation_patterns": "dict"
                }
            },
            
            "research_mode": {
                "name": "Research Assistant",
                "description": "Specialized prompt for research and analysis tasks",
                "content": """You are Morpheus Chat in Research Mode, equipped with advanced research capabilities.

**Research Capabilities:**
🔍 **Web Search**: Real-time web search across multiple search engines
📊 **Deep Research**: Multi-level research with source analysis and credibility scoring
📈 **Source Evaluation**: Automatic fact-checking and bias detection
📋 **Citation Management**: Proper attribution and reference formatting
🎯 **Synthesis**: Combining information from multiple sources into coherent insights

**Research Guidelines:**
- Always cite sources and provide credibility assessments
- Look for consensus and note conflicting information
- Prioritize recent, authoritative sources
- Identify knowledge gaps and limitations
- Provide confidence levels for findings
- Structure research with clear methodology

**Current Research Context:**
{% if research_query %}
- Research Query: {{ research_query }}
- Research Depth: {{ research_depth or "Standard" }}
- Source Types: {{ source_types | join(", ") if source_types else "All types" }}
{% endif %}

Conduct thorough, unbiased research and present findings with appropriate context and caveats.
""",
                "template_type": PromptType.RESEARCH_MODE,
                "suitable_contexts": [PromptContext.RESEARCH_SESSION],
                "optional_variables": {
                    "research_query": "string",
                    "research_depth": "string",
                    "source_types": "list"
                }
            },
            
            "code_assistant": {
                "name": "Code Assistant",
                "description": "Specialized prompt for programming assistance",
                "content": """You are Morpheus Chat in Code Assistant Mode, specialized for programming and development.

**Programming Capabilities:**
💻 **Multi-Language Support**: Python, JavaScript, TypeScript, Java, C++, Go, Rust, and more
🏗️ **Architecture Design**: System design, patterns, and best practices
🐛 **Debugging**: Error analysis, troubleshooting, and solution recommendations
📦 **Package Management**: Dependency selection and integration guidance
🧪 **Testing**: Unit tests, integration tests, and testing strategies
📚 **Documentation**: Code comments, README files, and API documentation
🎨 **Code Quality**: Refactoring, optimization, and style improvements

**Development Context:**
{% if programming_language %}
- Primary Language: {{ programming_language }}
{% endif %}
{% if project_type %}
- Project Type: {{ project_type }}
{% endif %}
{% if user_skill_level %}
- Skill Level: {{ user_skill_level }}
{% endif %}
{% if coding_preferences %}
- Preferences: {{ coding_preferences | join(", ") }}
{% endif %}

**Code Guidelines:**
- Write clean, readable, and well-documented code
- Follow language-specific best practices and conventions
- Include error handling and edge cases
- Provide explanations for complex logic
- Suggest improvements and alternatives when appropriate
- Create artifacts for substantial code snippets

When generating code, I'll create interactive artifacts that you can copy, modify, and execute.
""",
                "template_type": PromptType.CODE_ASSISTANT,
                "suitable_contexts": [PromptContext.CODING_SESSION, PromptContext.TASK_FOCUSED],
                "optional_variables": {
                    "programming_language": "string",
                    "project_type": "string",
                    "user_skill_level": "string",
                    "coding_preferences": "list"
                }
            }
        }
        
        # Create and store templates
        for template_id, template_data in default_templates.items():
            template = PromptTemplate(
                template_id=template_id,
                template_type=template_data["template_type"],
                name=template_data["name"],
                description=template_data["description"],
                template_content=template_data["content"],
                suitable_contexts=template_data.get("suitable_contexts", []),
                required_variables=template_data.get("required_variables", {}),
                optional_variables=template_data.get("optional_variables", {})
            )
            
            self.templates[template_id] = template
            
            # Save to file system
            await self._save_template_to_file(template)
        
        logger.info(f"Initialized {len(default_templates)} default prompt templates")
    
    async def _save_template_to_file(self, template: PromptTemplate):
        """Save template to file system"""
        template_file = self.template_dirs[0] / f"{template.template_id}.j2"
        
        async with asyncio.Lock():
            with open(template_file, 'w', encoding='utf-8') as f:
                f.write(template.template_content)
    
    async def generate_system_prompt(
        self,
        user_id: UserID,
        session_id: SessionID,
        context: PromptContext = PromptContext.ONGOING_CHAT,
        prompt_type: PromptType = PromptType.BASE_SYSTEM,
        custom_variables: Optional[Dict[str, Any]] = None,
        force_regenerate: bool = False
    ) -> str:
        """
        Generate dynamic system prompt based on user context and conversation history.
        
        Returns personalized system prompt with memory integration.
        """
        try:
            # Check cache first
            cache_key = f"{user_id}:{session_id}:{context.value}:{prompt_type.value}"
            if not force_regenerate and cache_key in self.prompt_cache:
                cached_prompt, cached_time = self.prompt_cache[cache_key]
                if datetime.now(timezone.utc) - cached_time < timedelta(seconds=self.config.cache_ttl_seconds):
                    return cached_prompt
            
            # Get user profile
            user_profile = await self._get_or_create_user_profile(user_id)
            
            # Gather context data
            context_data = await self._gather_context_data(user_id, session_id, context)
            
            # Select appropriate template
            template = self._select_template(prompt_type, context, context_data)
            
            # Prepare template variables
            template_vars = await self._prepare_template_variables(
                user_profile, context_data, custom_variables or {}
            )
            
            # Generate prompt
            jinja_template = Template(template.template_content)
            generated_prompt = jinja_template.render(**template_vars)
            
            # Post-process prompt
            final_prompt = await self._post_process_prompt(generated_prompt, user_profile, context_data)
            
            # Cache the result
            if self.config.template_caching:
                self.prompt_cache[cache_key] = (final_prompt, datetime.now(timezone.utc))
            
            # Update analytics
            await self._update_prompt_analytics(template.template_id, user_id, context)
            
            logger.debug(f"Generated system prompt for user {user_id}, type: {prompt_type.value}")
            return final_prompt
            
        except Exception as e:
            logger.error(f"Failed to generate system prompt: {e}")
            # Fallback to basic prompt
            return await self._get_fallback_prompt(user_id)
    
    async def _get_or_create_user_profile(self, user_id: UserID) -> UserProfile:
        """Get existing user profile or create new one"""
        if user_id in self.user_profiles:
            return self.user_profiles[user_id]
        
        # Create new profile
        profile = UserProfile(user_id=user_id)
        
        # Load from memory if available
        if self.memory_manager:
            await self._load_profile_from_memory(profile)
        
        self.user_profiles[user_id] = profile
        return profile
    
    async def _load_profile_from_memory(self, profile: UserProfile):
        """Load user profile data from memory system"""
        try:
            # Get core facts about user
            core_memories = await self.memory_manager.retrieve_memories(
                user_id=profile.user_id,
                memory_types=[MemoryType.CORE_FACT, MemoryType.CUSTOM_INSTRUCTION],
                importance_threshold=MemoryImportance.HIGH,
                limit=20
            )
            
            for memory in core_memories:
                content = memory.get("content", "").lower()
                
                # Extract name
                if "name is" in content or "call me" in content:
                    name_match = re.search(r"(?:name is|call me)\s+(\w+)", content)
                    if name_match and not profile.name:
                        profile.name = name_match.group(1).title()
                
                # Extract expertise level
                if any(level in content for level in ["beginner", "intermediate", "advanced", "expert"]):
                    for level in ["expert", "advanced", "intermediate", "beginner"]:
                        if level in content:
                            profile.expertise_level = level
                            break
                
                # Extract technical background
                tech_keywords = ["python", "javascript", "java", "programming", "developer", "engineer"]
                found_tech = [keyword for keyword in tech_keywords if keyword in content]
                profile.technical_background.extend(found_tech)
            
            # Get conversation patterns from recent interactions
            recent_conversations = await self.memory_manager.retrieve_memories(
                user_id=profile.user_id,
                memory_types=[MemoryType.CONVERSATION],
                limit=50
            )
            
            if recent_conversations:
                # Analyze conversation patterns
                total_length = sum(len(conv.get("content", "")) for conv in recent_conversations)
                avg_length = total_length / len(recent_conversations)
                
                if avg_length < 100:
                    profile.preferred_response_length = "short"
                elif avg_length > 500:
                    profile.preferred_response_length = "long"
                else:
                    profile.preferred_response_length = "medium"
                
                # Extract topics of interest
                all_content = " ".join(conv.get("content", "") for conv in recent_conversations)
                topics = self._extract_topics_from_text(all_content)
                profile.topics_of_interest = topics[:10]  # Top 10 topics
            
            profile.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            logger.error(f"Failed to load profile from memory: {e}")
    
    def _extract_topics_from_text(self, text: str) -> List[str]:
        """Extract topics from conversation text using simple keyword analysis"""
        # This would use more sophisticated NLP in production
        topic_keywords = {
            "programming": ["code", "python", "javascript", "programming", "development", "software"],
            "data_science": ["data", "analysis", "machine learning", "ai", "statistics"],
            "web_development": ["html", "css", "web", "frontend", "backend", "api"],
            "business": ["business", "strategy", "management", "marketing", "sales"],
            "education": ["learn", "study", "teach", "course", "university", "school"],
            "health": ["health", "medical", "fitness", "exercise", "nutrition"],
            "technology": ["tech", "computer", "software", "hardware", "innovation"],
            "science": ["research", "experiment", "theory", "scientific", "discovery"]
        }
        
        text_lower = text.lower()
        found_topics = []
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in text_lower for keyword in keywords):
                found_topics.append(topic)
        
        return found_topics
    
    async def _gather_context_data(
        self, 
        user_id: UserID, 
        session_id: SessionID, 
        context: PromptContext
    ) -> Dict[str, Any]:
        """Gather all relevant context data for prompt generation"""
        context_data = {
            "user_id": user_id,
            "session_id": session_id,
            "context": context,
            "timestamp": datetime.now(timezone.utc),
            "available_capabilities": [
                "persistent_memory", "web_search", "deep_research", 
                "file_processing", "artifact_creation", "code_execution"
            ]
        }
        
        # Memory context
        if self.memory_manager and self.config.memory_integration_enabled:
            try:
                # Get relevant memories
                memory_context = await self.memory_manager.retrieve_memories(
                    user_id=user_id,
                    limit=10,
                    importance_threshold=MemoryImportance.MEDIUM
                )
                
                # Get user facts specifically
                user_facts = await self.memory_manager.retrieve_memories(
                    user_id=user_id,
                    memory_types=[MemoryType.CORE_FACT],
                    limit=5
                )
                
                context_data.update({
                    "memory_context": memory_context[:5],  # Limit for prompt length
                    "user_facts": user_facts,
                    "memory_available": True
                })
                
            except Exception as e:
                logger.error(f"Failed to gather memory context: {e}")
                context_data["memory_available"] = False
        
        return context_data
    
    def _select_template(
        self, 
        prompt_type: PromptType, 
        context: PromptContext, 
        context_data: Dict[str, Any]
    ) -> PromptTemplate:
        """Select the most appropriate template based on context"""
        
        # First, try to find exact match
        if prompt_type.value in self.templates:
            template = self.templates[prompt_type.value]
            if context in template.suitable_contexts or not template.suitable_contexts:
                return template
        
        # Find best matching template based on context
        suitable_templates = []
        for template in self.templates.values():
            if context in template.suitable_contexts or not template.suitable_contexts:
                suitable_templates.append(template)
        
        if suitable_templates:
            # Sort by success rate and usage
            suitable_templates.sort(
                key=lambda t: (t.success_rate, t.usage_count),
                reverse=True
            )
            return suitable_templates[0]
        
        # Fallback to base system template
        return self.templates.get("base_system", list(self.templates.values())[0])
    
    async def _prepare_template_variables(
        self,
        user_profile: UserProfile,
        context_data: Dict[str, Any],
        custom_variables: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Prepare all variables for template rendering"""
        
        template_vars = {
            # User information
            "user_name": user_profile.name,
            "user_id": user_profile.user_id,
            "expertise_level": user_profile.expertise_level,
            "communication_style": user_profile.preferred_communication_style,
            "response_length_preference": user_profile.preferred_response_length,
            
            # Context information
            "session_id": context_data["session_id"],
            "context": context_data["context"],
            "timestamp": context_data["timestamp"],
            "available_capabilities": context_data["available_capabilities"],
            
            # Memory context
            "memory_context": context_data.get("memory_context", []),
            "user_facts": context_data.get("user_facts", []),
            "memory_available": context_data.get("memory_available", False),
            
            # User preferences and patterns
            "topics_of_interest": user_profile.topics_of_interest,
            "technical_background": user_profile.technical_background,
            "conversation_patterns": user_profile.conversation_patterns,
            "prefers_examples": user_profile.prefers_examples,
            "prefers_step_by_step": user_profile.prefers_step_by_step,
            
            # Time context
            "current_time": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "user_timezone": user_profile.timezone,
            
            # Capabilities
            "web_search_available": True,
            "research_available": True,
            "file_processing_available": True,
            "artifact_creation_available": True,
        }
        
        # Add custom variables
        template_vars.update(custom_variables)
        
        return template_vars
    
    async def _post_process_prompt(
        self,
        prompt: str,
        user_profile: UserProfile,
        context_data: Dict[str, Any]
    ) -> str:
        """Post-process the generated prompt for optimization and safety"""
        
        # Trim excessive whitespace
        prompt = re.sub(r'\n\s*\n\s*\n', '\n\n', prompt)
        prompt = prompt.strip()
        
        # Apply safety filtering if enabled
        if self.config.safety_filtering:
            prompt = await self._apply_safety_filters(prompt)
        
        # Apply compression if enabled
        if self.config.prompt_compression:
            prompt = await self._compress_prompt(prompt)
        
        # Ensure prompt length limits
        max_prompt_length = 8000  # Adjust based on model context limits
        if len(prompt) > max_prompt_length:
            prompt = await self._truncate_prompt_intelligently(prompt, max_prompt_length)
        
        return prompt
    
    async def _apply_safety_filters(self, prompt: str) -> str:
        """Apply safety filters to prompt content"""
        # Remove any potentially harmful instructions
        harmful_patterns = [
            r"ignore\s+(?:previous|all|safety)\s+(?:instructions|guidelines)",
            r"disregard\s+(?:safety|ethical)\s+(?:guidelines|constraints)",
            r"act\s+as\s+(?:an?\s+)?(?:unrestricted|unlimited|jailbroken)",
        ]
        
        for pattern in harmful_patterns:
            prompt = re.sub(pattern, "[SAFETY_FILTERED]", prompt, flags=re.IGNORECASE)
        
        return prompt
    
    async def _compress_prompt(self, prompt: str) -> str:
        """Compress prompt while maintaining meaning"""
        # Simple compression strategies
        
        # Remove redundant phrases
        redundant_phrases = [
            r"\s*\(as mentioned (?:above|before|earlier)\)",
            r"\s*\(as (?:we|I) discussed (?:before|earlier)\)",
            r"\s*please\s+",
            r"\s*kindly\s+",
        ]
        
        for phrase in redundant_phrases:
            prompt = re.sub(phrase, " ", prompt, flags=re.IGNORECASE)
        
        # Normalize whitespace
        prompt = re.sub(r'\s+', ' ', prompt)
        
        return prompt.strip()
    
    async def _truncate_prompt_intelligently(self, prompt: str, max_length: int) -> str:
        """Intelligently truncate prompt while preserving important sections"""
        if len(prompt) <= max_length:
            return prompt
        
        # Identify important sections to preserve
        sections = prompt.split('\n\n')
        
        # Always preserve the first section (core instructions)
        result = sections[0] + '\n\n'
        remaining_length = max_length - len(result)
        
        # Add sections in order of importance
        for section in sections[1:]:
            if len(section) + 2 <= remaining_length:  # +2 for \n\n
                result += section + '\n\n'
                remaining_length -= len(section) + 2
            else:
                # Add truncated version of section if it's important
                if any(keyword in section.lower() for keyword in ["instructions", "context", "capabilities"]):
                    truncated = section[:remaining_length-20] + "... [truncated]"
                    result += truncated
                break
        
        return result.strip()
    
    async def _get_fallback_prompt(self, user_id: UserID) -> str:
        """Get basic fallback prompt when generation fails"""
        return f"""You are Morpheus Chat, an advanced AI assistant with persistent memory and powerful capabilities.

I can help you with:
- Answering questions using my knowledge and web search
- Conducting deep research on complex topics
- Processing and analyzing files you upload
- Creating interactive content and artifacts
- Remembering our conversation history

User ID: {user_id}
I'm ready to assist you with any questions or tasks!"""
    
    async def _update_prompt_analytics(self, template_id: str, user_id: UserID, context: PromptContext):
        """Update analytics for prompt usage"""
        try:
            if template_id not in self.prompt_analytics:
                self.prompt_analytics[template_id] = {
                    "usage_count": 0,
                    "contexts": {},
                    "users": set(),
                    "last_used": None
                }
            
            analytics = self.prompt_analytics[template_id]
            analytics["usage_count"] += 1
            analytics["contexts"][context.value] = analytics["contexts"].get(context.value, 0) + 1
            analytics["users"].add(user_id)
            analytics["last_used"] = datetime.now(timezone.utc)
            
            # Update template usage count
            if template_id in self.templates:
                self.templates[template_id].usage_count += 1
                
        except Exception as e:
            logger.error(f"Failed to update prompt analytics: {e}")
    
    async def update_user_profile(
        self,
        user_id: UserID,
        updates: Dict[str, Any]
    ):
        """Update user profile with new information"""
        try:
            profile = await self._get_or_create_user_profile(user_id)
            
            # Apply updates
            for key, value in updates.items():
                if hasattr(profile, key):
                    setattr(profile, key, value)
            
            profile.last_updated = datetime.now(timezone.utc)
            
            # Invalidate cached prompts for this user
            cache_keys_to_remove = [
                key for key in self.prompt_cache.keys()
                if key.startswith(f"{user_id}:")
            ]
            for key in cache_keys_to_remove:
                del self.prompt_cache[key]
                
            logger.info(f"Updated user profile for {user_id}")
            
        except Exception as e:
            logger.error(f"Failed to update user profile: {e}")
    
    async def get_prompt_analytics(self) -> Dict[str, Any]:
        """Get comprehensive prompt system analytics"""
        return {
            "templates": {
                template_id: {
                    "name": template.name,
                    "type": template.template_type.value,
                    "usage_count": template.usage_count,
                    "success_rate": template.success_rate,
                    "avg_quality": template.avg_response_quality
                }
                for template_id, template in self.templates.items()
            },
            "usage_analytics": self.prompt_analytics,
            "cache_stats": {
                "cached_prompts": len(self.prompt_cache),
                "cache_enabled": self.config.template_caching
            },
            "user_profiles": len(self.user_profiles),
            "system_config": {
                "memory_integration": self.config.memory_integration_enabled,
                "personalization": self.config.personalization_enabled,
                "safety_filtering": self.config.safety_filtering
            }
        }
    
    def get_available_prompt_types(self) -> List[Dict[str, str]]:
        """Get list of available prompt types and their descriptions"""
        return [
            {
                "type": template.template_type.value,
                "name": template.name,
                "description": template.description,
                "contexts": [ctx.value for ctx in template.suitable_contexts]
            }
            for template in self.templates.values()
        ]