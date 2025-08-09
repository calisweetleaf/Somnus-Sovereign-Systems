"""
MORPHEUS CHAT - Project Memory Integration
Seamless integration of project activities with persistent memory system

Revolutionary Features:
- Project-scoped memory namespaces
- Automatic memory creation from project activities
- Cross-project memory insights and patterns
- Project memory analytics and optimization
- Intelligent memory prioritization based on project context
"""

import asyncio
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

from ..memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope

logger = logging.getLogger(__name__)


class ProjectMemoryType(str, Enum):
    """Project-specific memory types"""
    PROJECT_CREATION = "project_creation"
    FILE_UPLOAD = "file_upload"
    KNOWLEDGE_DISCOVERY = "knowledge_discovery"
    COLLABORATION_SESSION = "collaboration_session"
    AUTOMATION_EXECUTION = "automation_execution"
    ARTIFACT_CREATION = "artifact_creation"
    PROJECT_INSIGHT = "project_insight"
    USER_INTERACTION = "user_interaction"
    SYSTEM_EVENT = "system_event"


@dataclass
class ProjectMemoryContext:
    """Context for project memory operations"""
    project_id: str
    project_name: str
    user_id: str
    session_context: Dict[str, Any] = field(default_factory=dict)
    memory_namespace: str = ""
    
    def __post_init__(self):
        if not self.memory_namespace:
            self.memory_namespace = f"project:{self.project_id}"


class ProjectMemoryManager:
    """
    Project-specific memory management with intelligent integration
    
    Features:
    - Automatic memory creation from project activities
    - Project-scoped memory namespaces for organization
    - Intelligent memory importance scoring based on project context
    - Cross-project memory analysis and insights
    - Memory-driven project recommendations
    """
    
    def __init__(self, memory_manager: MemoryManager):
        self.memory_manager = memory_manager
        
        # Project memory tracking
        self.active_project_contexts: Dict[str, ProjectMemoryContext] = {}
        self.project_memory_stats: Dict[str, Dict[str, Any]] = {}
        
        # Memory patterns and insights
        self.cross_project_patterns: Dict[str, Any] = {}
        self.memory_insights_cache: Dict[str, List[Dict[str, Any]]] = {}
        
        logger.info("Project memory manager initialized")
    
    async def initialize_project_memory(
        self,
        project_id: str,
        project_name: str,
        user_id: str,
        project_metadata: Dict[str, Any]
    ) -> ProjectMemoryContext:
        """Initialize memory context for new project"""
        
        context = ProjectMemoryContext(
            project_id=project_id,
            project_name=project_name,
            user_id=user_id,
            session_context=project_metadata
        )
        
        # Store project creation memory
        await self.store_project_memory(
            context=context,
            memory_type=ProjectMemoryType.PROJECT_CREATION,
            content=f"Created project '{project_name}' with configuration: {json.dumps(project_metadata)}",
            importance=MemoryImportance.HIGH,
            metadata={
                "project_type": project_metadata.get("project_type", "unknown"),
                "specifications": project_metadata.get("specs", {}),
                "collaboration_enabled": project_metadata.get("collaboration_enabled", False),
                "automation_enabled": project_metadata.get("automation_enabled", False)
            }
        )
        
        # Initialize project memory stats
        self.project_memory_stats[project_id] = {
            "total_memories": 1,
            "memories_by_type": {ProjectMemoryType.PROJECT_CREATION.value: 1},
            "last_activity": datetime.now(timezone.utc),
            "importance_distribution": {MemoryImportance.HIGH.value: 1}
        }
        
        # Track active context
        self.active_project_contexts[project_id] = context
        
        logger.info(f"Initialized memory context for project {project_name}")
        return context
    
    async def store_project_memory(
        self,
        context: ProjectMemoryContext,
        memory_type: ProjectMemoryType,
        content: str,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        metadata: Optional[Dict[str, Any]] = None,
        tags: Optional[Set[str]] = None
    ) -> UUID:
        """Store memory with project context"""
        
        # Enhanced metadata with project context
        enhanced_metadata = {
            "project_id": context.project_id,
            "project_name": context.project_name,
            "memory_type": memory_type.value,
            "namespace": context.memory_namespace,
            **(metadata or {})
        }
        
        # Enhanced tags with project context
        enhanced_tags = {
            "project",
            f"project:{context.project_id}",
            memory_type.value,
            **(tags or set())
        }
        
        # Store memory
        memory_id = await self.memory_manager.store_memory(
            user_id=context.user_id,
            content=content,
            memory_type=MemoryType.CUSTOM,
            importance=importance,
            scope=MemoryScope.USER,
            metadata=enhanced_metadata,
            tags=enhanced_tags
        )
        
        # Update project memory stats
        await self._update_project_stats(context.project_id, memory_type, importance)
        
        # Check for memory insights
        await self._analyze_memory_patterns(context.project_id)
        
        return memory_id
    
    async def store_file_upload_memory(
        self,
        context: ProjectMemoryContext,
        filename: str,
        file_analysis: Dict[str, Any]
    ) -> UUID:
        """Store memory for file upload and analysis"""
        
        content = f"""
        File uploaded to project: {filename}
        
        Analysis Results:
        - File type: {file_analysis.get('file_type', 'unknown')}
        - Size: {file_analysis.get('size_bytes', 0)} bytes
        - Key concepts: {', '.join(file_analysis.get('key_concepts', []))}
        - Importance score: {file_analysis.get('importance_score', 0.0)}
        - Categories: {file_analysis.get('suggested_category', 'general')}
        
        Summary: {file_analysis.get('content_summary', 'No summary available')}
        """
        
        importance = MemoryImportance.HIGH if file_analysis.get('importance_score', 0) > 0.7 else MemoryImportance.MEDIUM
        
        return await self.store_project_memory(
            context=context,
            memory_type=ProjectMemoryType.FILE_UPLOAD,
            content=content,
            importance=importance,
            metadata={
                "filename": filename,
                "file_analysis": file_analysis,
                "knowledge_items_created": file_analysis.get('knowledge_items_created', 0)
            },
            tags={"file_upload", "analysis", file_analysis.get('file_type', 'unknown')}
        )
    
    async def store_collaboration_memory(
        self,
        context: ProjectMemoryContext,
        task_title: str,
        collaboration_results: Dict[str, Any]
    ) -> UUID:
        """Store memory for collaboration session"""
        
        content = f"""
        Collaborative task completed: {task_title}
        
        Collaboration Details:
        - Participating agents: {collaboration_results.get('participating_agents', 0)}
        - Task type: {collaboration_results.get('task_type', 'general')}
        - Execution time: {collaboration_results.get('execution_time', 0)} seconds
        - Artifacts created: {len(collaboration_results.get('artifacts', []))}
        
        Results Summary:
        {collaboration_results.get('output', 'No output available')[:500]}...
        
        Agent Contributions:
        {chr(10).join(f"- {agent}: {contribution[:100]}..." for agent, contribution in collaboration_results.get('agent_contributions', {}).items())}
        """
        
        return await self.store_project_memory(
            context=context,
            memory_type=ProjectMemoryType.COLLABORATION_SESSION,
            content=content,
            importance=MemoryImportance.HIGH,
            metadata={
                "task_title": task_title,
                "collaboration_data": collaboration_results,
                "participating_agents": collaboration_results.get('participating_agents', 0)
            },
            tags={"collaboration", "multi_agent", collaboration_results.get('task_type', 'general')}
        )
    
    async def store_automation_memory(
        self,
        context: ProjectMemoryContext,
        rule_name: str,
        execution_results: Dict[str, Any]
    ) -> UUID:
        """Store memory for automation execution"""
        
        content = f"""
        Automation rule executed: {rule_name}
        
        Execution Details:
        - Status: {execution_results.get('status', 'unknown')}
        - Duration: {execution_results.get('duration', 0)} seconds
        - Artifacts created: {len(execution_results.get('artifacts', []))}
        - Output: {execution_results.get('output', 'No output')[:300]}...
        
        Rule Configuration:
        - Trigger: {execution_results.get('trigger_type', 'unknown')}
        - Action: {execution_results.get('action_type', 'unknown')}
        """
        
        importance = MemoryImportance.HIGH if execution_results.get('status') == 'completed' else MemoryImportance.MEDIUM
        
        return await self.store_project_memory(
            context=context,
            memory_type=ProjectMemoryType.AUTOMATION_EXECUTION,
            content=content,
            importance=importance,
            metadata={
                "rule_name": rule_name,
                "execution_data": execution_results,
                "automation_successful": execution_results.get('status') == 'completed'
            },
            tags={"automation", execution_results.get('trigger_type', 'unknown'), execution_results.get('action_type', 'unknown')}
        )
    
    async def store_artifact_memory(
        self,
        context: ProjectMemoryContext,
        artifact_title: str,
        artifact_data: Dict[str, Any]
    ) -> UUID:
        """Store memory for artifact creation and execution"""
        
        content = f"""
        Artifact created in project: {artifact_title}
        
        Artifact Details:
        - Type: {artifact_data.get('artifact_type', 'unknown')}
        - Uses project VM: {artifact_data.get('uses_project_vm', False)}
        - Knowledge integrated: {artifact_data.get('knowledge_integrated', False)}
        - Collaboration enabled: {artifact_data.get('collaboration_enabled', False)}
        
        Execution Results:
        {artifact_data.get('execution_output', 'Not executed yet')[:400]}...
        
        Project Integration:
        - Knowledge references: {len(artifact_data.get('knowledge_refs', []))}
        - File dependencies: {len(artifact_data.get('file_dependencies', []))}
        """
        
        return await self.store_project_memory(
            context=context,
            memory_type=ProjectMemoryType.ARTIFACT_CREATION,
            content=content,
            importance=MemoryImportance.HIGH,
            metadata={
                "artifact_title": artifact_title,
                "artifact_data": artifact_data,
                "vm_enabled": artifact_data.get('uses_project_vm', False),
                "collaborative": artifact_data.get('collaboration_enabled', False)
            },
            tags={"artifact", artifact_data.get('artifact_type', 'unknown'), "creation"}
        )
    
    async def store_knowledge_discovery_memory(
        self,
        context: ProjectMemoryContext,
        discovery_type: str,
        discovery_data: Dict[str, Any]
    ) -> UUID:
        """Store memory for knowledge discoveries and insights"""
        
        content = f"""
        Knowledge discovery in project: {discovery_type}
        
        Discovery Details:
        - Type: {discovery_type}
        - Confidence: {discovery_data.get('confidence', 0.0)}
        - Source files: {len(discovery_data.get('source_files', []))}
        - Related concepts: {', '.join(discovery_data.get('related_concepts', []))}
        
        Discovery Content:
        {discovery_data.get('content', 'No content available')[:400]}...
        
        Insights Generated:
        {chr(10).join(f"- {insight}" for insight in discovery_data.get('insights', [])[:5])}
        """
        
        importance = MemoryImportance.HIGH if discovery_data.get('confidence', 0) > 0.8 else MemoryImportance.MEDIUM
        
        return await self.store_project_memory(
            context=context,
            memory_type=ProjectMemoryType.KNOWLEDGE_DISCOVERY,
            content=content,
            importance=importance,
            metadata={
                "discovery_type": discovery_type,
                "discovery_data": discovery_data,
                "confidence_score": discovery_data.get('confidence', 0.0)
            },
            tags={"knowledge", "discovery", discovery_type}
        )
    
    async def retrieve_project_memories(
        self,
        project_id: str,
        memory_types: Optional[List[ProjectMemoryType]] = None,
        limit: int = 20,
        importance_threshold: Optional[MemoryImportance] = None
    ) -> List[Dict[str, Any]]:
        """Retrieve memories for specific project"""
        
        # Build search query
        search_tags = [f"project:{project_id}"]
        
        if memory_types:
            search_tags.extend([mem_type.value for mem_type in memory_types])
        
        # Retrieve memories
        memories = await self.memory_manager.search_memories(
            query=" ".join(search_tags),
            user_id=self.active_project_contexts.get(project_id, {}).get('user_id'),
            limit=limit,
            importance_threshold=importance_threshold
        )
        
        return memories
    
    async def get_project_memory_summary(self, project_id: str) -> Dict[str, Any]:
        """Get comprehensive memory summary for project"""
        
        stats = self.project_memory_stats.get(project_id, {})
        
        # Get recent memories
        recent_memories = await self.retrieve_project_memories(
            project_id=project_id,
            limit=10
        )
        
        # Get memory insights
        insights = self.memory_insights_cache.get(project_id, [])
        
        return {
            "project_id": project_id,
            "total_memories": stats.get("total_memories", 0),
            "memories_by_type": stats.get("memories_by_type", {}),
            "importance_distribution": stats.get("importance_distribution", {}),
            "last_activity": stats.get("last_activity", datetime.now(timezone.utc)).isoformat(),
            "recent_memories": [
                {
                    "content": memory.get("content", "")[:200],
                    "type": memory.get("metadata", {}).get("memory_type", "unknown"),
                    "importance": memory.get("importance", "medium"),
                    "created_at": memory.get("created_at", "")
                }
                for memory in recent_memories[:5]
            ],
            "memory_insights": insights[:3],
            "cross_project_connections": await self._get_cross_project_connections(project_id)
        }
    
    async def search_project_knowledge_with_memory(
        self,
        project_id: str,
        query: str,
        include_related_projects: bool = True
    ) -> Dict[str, Any]:
        """Search combining project knowledge and memory"""
        
        # Search project memories
        memory_results = await self.memory_manager.search_memories(
            query=f"project:{project_id} {query}",
            user_id=self.active_project_contexts.get(project_id, {}).get('user_id'),
            limit=10
        )
        
        # Search related project memories if requested
        related_memories = []
        if include_related_projects:
            related_project_ids = await self._find_related_projects(project_id, query)
            
            for related_id in related_project_ids[:3]:
                related_results = await self.memory_manager.search_memories(
                    query=f"project:{related_id} {query}",
                    limit=5
                )
                related_memories.extend(related_results)
        
        return {
            "query": query,
            "project_memories": memory_results,
            "related_project_memories": related_memories,
            "memory_insights": await self._generate_search_insights(query, memory_results)
        }
    
    async def generate_project_memory_insights(self, project_id: str) -> List[Dict[str, Any]]:
        """Generate insights from project memory patterns"""
        
        # Get all project memories
        all_memories = await self.retrieve_project_memories(
            project_id=project_id,
            limit=100
        )
        
        insights = []
        
        # Analyze memory patterns
        memory_types = {}
        for memory in all_memories:
            mem_type = memory.get("metadata", {}).get("memory_type", "unknown")
            memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
        
        # Generate activity pattern insights
        if memory_types:
            most_common = max(memory_types.items(), key=lambda x: x[1])
            insights.append({
                "type": "activity_pattern",
                "title": "Primary Project Activity",
                "description": f"This project shows high activity in {most_common[0]} with {most_common[1]} recorded events",
                "confidence": 0.9,
                "data": memory_types
            })
        
        # Analyze collaboration patterns
        collaboration_memories = [m for m in all_memories if "collaboration" in m.get("metadata", {}).get("memory_type", "")]
        if collaboration_memories:
            insights.append({
                "type": "collaboration_pattern",
                "title": "Collaboration Activity",
                "description": f"Project has {len(collaboration_memories)} collaboration sessions with multi-agent teams",
                "confidence": 0.8,
                "data": {"collaboration_sessions": len(collaboration_memories)}
            })
        
        # Analyze knowledge growth patterns
        knowledge_memories = [m for m in all_memories if "knowledge" in m.get("metadata", {}).get("memory_type", "")]
        if knowledge_memories:
            insights.append({
                "type": "knowledge_growth",
                "title": "Knowledge Development",
                "description": f"Project knowledge base has grown through {len(knowledge_memories)} discovery events",
                "confidence": 0.7,
                "data": {"knowledge_discoveries": len(knowledge_memories)}
            })
        
        # Cache insights
        self.memory_insights_cache[project_id] = insights
        
        return insights
    
    async def _update_project_stats(
        self,
        project_id: str,
        memory_type: ProjectMemoryType,
        importance: MemoryImportance
    ):
        """Update project memory statistics"""
        
        if project_id not in self.project_memory_stats:
            self.project_memory_stats[project_id] = {
                "total_memories": 0,
                "memories_by_type": {},
                "importance_distribution": {},
                "last_activity": datetime.now(timezone.utc)
            }
        
        stats = self.project_memory_stats[project_id]
        
        # Update counts
        stats["total_memories"] += 1
        stats["memories_by_type"][memory_type.value] = stats["memories_by_type"].get(memory_type.value, 0) + 1
        stats["importance_distribution"][importance.value] = stats["importance_distribution"].get(importance.value, 0) + 1
        stats["last_activity"] = datetime.now(timezone.utc)
    
    async def _analyze_memory_patterns(self, project_id: str):
        """Analyze memory patterns for insights"""
        
        # Get recent memories for pattern analysis
        recent_memories = await self.retrieve_project_memories(
            project_id=project_id,
            limit=50
        )
        
        # Analyze for patterns (simplified analysis)
        if len(recent_memories) >= 5:
            # Check for frequent activity patterns
            activity_intervals = []
            for i in range(1, len(recent_memories)):
                if i < len(recent_memories):
                    # Calculate time between activities
                    prev_time = datetime.fromisoformat(recent_memories[i-1].get("created_at", ""))
                    curr_time = datetime.fromisoformat(recent_memories[i].get("created_at", ""))
                    interval = (prev_time - curr_time).total_seconds()
                    activity_intervals.append(interval)
            
            # Store pattern data for future insights
            if project_id not in self.cross_project_patterns:
                self.cross_project_patterns[project_id] = {}
            
            self.cross_project_patterns[project_id]["activity_intervals"] = activity_intervals
    
    async def _get_cross_project_connections(self, project_id: str) -> List[Dict[str, Any]]:
        """Find connections between this project and others"""
        
        connections = []
        
        # Get memories from this project
        project_memories = await self.retrieve_project_memories(project_id, limit=20)
        
        # Extract key concepts and tags
        project_concepts = set()
        for memory in project_memories:
            # Extract concepts from content (simplified)
            content = memory.get("content", "").lower()
            tags = memory.get("tags", [])
            project_concepts.update(tags)
        
        # Find other projects with similar concepts
        for other_project_id, other_patterns in self.cross_project_patterns.items():
            if other_project_id != project_id:
                # Calculate conceptual similarity (simplified)
                similarity_score = len(project_concepts) / max(len(project_concepts), 1) * 0.5
                
                if similarity_score > 0.3:
                    connections.append({
                        "project_id": other_project_id,
                        "similarity_score": similarity_score,
                        "connection_type": "conceptual_similarity",
                        "shared_concepts": list(project_concepts)[:5]
                    })
        
        return connections[:3]  # Return top 3 connections
    
    async def _find_related_projects(self, project_id: str, query: str) -> List[str]:
        """Find projects related to the current project based on query"""
        
        related_projects = []
        
        # Search for memories containing the query across all projects
        all_memories = await self.memory_manager.search_memories(
            query=query,
            limit=50
        )
        
        # Extract project IDs from results
        project_ids = set()
        for memory in all_memories:
            mem_project_id = memory.get("metadata", {}).get("project_id")
            if mem_project_id and mem_project_id != project_id:
                project_ids.add(mem_project_id)
        
        return list(project_ids)[:5]  # Return top 5 related projects
    
    async def _generate_search_insights(
        self,
        query: str,
        memory_results: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate insights from memory search results"""
        
        insights = []
        
        if memory_results:
            # Analyze result patterns
            memory_types = {}
            for result in memory_results:
                mem_type = result.get("metadata", {}).get("memory_type", "unknown")
                memory_types[mem_type] = memory_types.get(mem_type, 0) + 1
            
            if memory_types:
                most_relevant_type = max(memory_types.items(), key=lambda x: x[1])
                insights.append({
                    "type": "search_pattern",
                    "description": f"Most relevant results are from {most_relevant_type[0]} activities",
                    "confidence": 0.8,
                    "data": memory_types
                })
        
        return insights
    
    async def cleanup_project_memory(self, project_id: str):
        """Cleanup project memory context"""
        
        # Remove from active contexts
        if project_id in self.active_project_contexts:
            del self.active_project_contexts[project_id]
        
        # Archive project memory stats
        if project_id in self.project_memory_stats:
            stats = self.project_memory_stats[project_id]
            # Could archive to file or database here
            del self.project_memory_stats[project_id]
        
        # Clear cached insights
        if project_id in self.memory_insights_cache:
            del self.memory_insights_cache[project_id]
        
        logger.info(f"Cleaned up memory context for project {project_id}")