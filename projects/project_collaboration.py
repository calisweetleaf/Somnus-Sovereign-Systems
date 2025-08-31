"""
MORPHEUS CHAT - Project Collaboration System
Multi-agent collaboration specifically for project environments

Revolutionary Features:
- Project-specific AI agent teams
- Automatic task delegation based on project context
- Collaborative artifact creation and editing
- Persistent agent knowledge within projects
- Agent specialization based on project content
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

from ..agent_collaboration_core import (
    AgentCapability, AgentProfile, CollaborativeAgent, 
    MessageBus, AgentMessage, MessageType
)
from ..agent_communication_protocol import AgentCommunicationMessage, CommunicationPriority

logger = logging.getLogger(__name__)


class ProjectRole(str, Enum):
    """Project-specific agent roles"""
    PROJECT_MANAGER = "project_manager"
    RESEARCHER = "researcher"
    ANALYST = "analyst"
    DEVELOPER = "developer"
    WRITER = "writer"
    DESIGNER = "designer"
    REVIEWER = "reviewer"
    SPECIALIST = "specialist"


@dataclass
class ProjectTask:
    """Project-specific task for agent collaboration"""
    task_id: UUID = field(default_factory=uuid4)
    title: str = ""
    description: str = ""
    task_type: str = "general"
    priority: str = "medium"
    
    # Assignment details
    assigned_agents: List[UUID] = field(default_factory=list)
    required_capabilities: List[AgentCapability] = field(default_factory=list)
    estimated_duration: int = 60  # minutes
    
    # Status tracking
    status: str = "pending"  # pending, active, completed, failed
    progress: float = 0.0
    
    # Project context
    project_files: List[str] = field(default_factory=list)
    knowledge_refs: List[str] = field(default_factory=list)
    artifact_refs: List[str] = field(default_factory=list)
    
    # Results
    output_artifacts: List[str] = field(default_factory=list)
    agent_contributions: Dict[str, str] = field(default_factory=dict)
    
    # Timestamps
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None


class ProjectAgent(CollaborativeAgent):
    """
    Enhanced collaborative agent with project-specific capabilities
    
    Extends base agent with:
    - Project context awareness
    - File system access within project VM
    - Knowledge base integration
    - Artifact creation capabilities
    """
    
    def __init__(
        self,
        profile: AgentProfile,
        project_id: str,
        project_vm,
        project_knowledge,
        *args,
        **kwargs
    ):
        super().__init__(profile, *args, **kwargs)
        
        # Project-specific components
        self.project_id = project_id
        self.project_vm = project_vm
        self.project_knowledge = project_knowledge
        
        # Project context
        self.project_understanding: Dict[str, Any] = {}
        self.available_files: List[str] = []
        self.created_artifacts: List[str] = []
        
        # Agent workspace in project VM
        self.agent_workspace = f"/project/collaboration/agent_{self.profile.agent_id}"
        
        # Specialization based on project content
        self.specialization_data: Dict[str, float] = {}
        
        logger.info(f"Project agent {profile.name} initialized for project {project_id}")
    
    async def initialize_project_context(self):
        """Initialize agent with project-specific context"""
        
        try:
            # Create agent workspace in VM
            await self.project_vm.execute_command(f"mkdir -p {self.agent_workspace}")
            
            # Load project knowledge summary
            knowledge_summary = await self.project_knowledge.get_knowledge_summary()
            self.project_understanding.update(knowledge_summary)
            
            # Analyze project files for specialization
            await self._analyze_project_for_specialization()
            
            # Setup agent tools in VM
            await self._setup_agent_tools()
            
            logger.info(f"Agent {self.profile.name} initialized with project context")
            
        except Exception as e:
            logger.error(f"Failed to initialize project context for agent {self.profile.name}: {e}")
    
    async def _analyze_project_for_specialization(self):
        """Analyze project content to determine agent specialization"""
        
        # Get file categories from project intelligence
        file_categories = self.project_understanding.get('items_by_type', {})
        top_concepts = self.project_understanding.get('top_concepts', {})
        
        # Calculate specialization scores based on project content
        if 'code' in file_categories and file_categories['code'] > 5:
            self.specialization_data['coding'] = 0.9
            if AgentCapability.CODE_GENERATION not in self.profile.capabilities:
                self.profile.capabilities.append(AgentCapability.CODE_GENERATION)
        
        if 'research' in file_categories or 'documentation' in file_categories:
            self.specialization_data['research'] = 0.8
            if AgentCapability.RESEARCH_SYNTHESIS not in self.profile.capabilities:
                self.profile.capabilities.append(AgentCapability.RESEARCH_SYNTHESIS)
        
        if 'data' in file_categories:
            self.specialization_data['data_analysis'] = 0.7
            if AgentCapability.DATA_PROCESSING not in self.profile.capabilities:
                self.profile.capabilities.append(AgentCapability.DATA_PROCESSING)
        
        # Analyze concepts for domain expertise
        concept_domains = {
            'ai': ['neural', 'model', 'training', 'machine', 'learning'],
            'web': ['html', 'css', 'javascript', 'react', 'api'],
            'data': ['analysis', 'dataset', 'visualization', 'statistics'],
            'business': ['strategy', 'market', 'customer', 'revenue', 'plan']
        }
        
        for domain, keywords in concept_domains.items():
            domain_score = sum(top_concepts.get(keyword, 0) for keyword in keywords)
            if domain_score > 10:
                self.specialization_data[domain] = min(domain_score / 50.0, 1.0)
    
    async def _setup_agent_tools(self):
        """Setup agent-specific tools in project VM"""
        
        # Create agent toolkit script
        agent_toolkit = f'''#!/usr/bin/env python3
"""
Project Agent Toolkit - {self.profile.name}
Agent ID: {self.profile.agent_id}
"""

import json
import os
import subprocess
from pathlib import Path

class AgentToolkit:
    def __init__(self):
        self.agent_id = "{self.profile.agent_id}"
        self.agent_name = "{self.profile.name}"
        self.workspace = Path("{self.agent_workspace}")
        self.project_root = Path("/project")
        
    def access_project_files(self, pattern="*"):
        """Access project files matching pattern"""
        files = list(self.project_root.glob(f"files/**/{pattern}"))
        return [str(f) for f in files]
    
    def read_file(self, file_path):
        """Read project file"""
        try:
            with open(file_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    def create_artifact(self, name, content, artifact_type="text"):
        """Create artifact in project"""
        artifact_path = self.project_root / "artifacts" / f"{name}_{self.agent_id}"
        artifact_path.parent.mkdir(exist_ok=True)
        
        with open(artifact_path, 'w') as f:
            f.write(content)
        
        return str(artifact_path)
    
    def search_knowledge(self, query):
        """Search project knowledge base"""
        # This would call the knowledge base API
        return f"Knowledge search for: {query}"
    
    def collaborate_with_agent(self, agent_id, message):
        """Send message to another agent"""
        # This would use the message bus
        return f"Message sent to {agent_id}: {message}"
    
    def log_activity(self, activity):
        """Log agent activity"""
        log_file = self.workspace / "activity.log"
        with open(log_file, 'a') as f:
            f.write(f"{datetime.now()}: {activity}\\n")

# Global toolkit instance
toolkit = AgentToolkit()
        '''
        
        await self.project_vm.write_file(
            f"{self.agent_workspace}/toolkit.py",
            agent_toolkit
        )
        
        await self.project_vm.execute_command(f"chmod +x {self.agent_workspace}/toolkit.py")
    
    async def execute_project_task(self, task: ProjectTask) -> Dict[str, Any]:
        """Execute project-specific task using project context"""
        
        try:
            task.started_at = datetime.now(timezone.utc)
            task.status = "active"
            
            # Gather project context for task
            context = await self._gather_task_context(task)
            
            # Execute task based on type and agent capabilities
            if task.task_type == "research":
                result = await self._execute_research_task(task, context)
            elif task.task_type == "development":
                result = await self._execute_development_task(task, context)
            elif task.task_type == "analysis":
                result = await self._execute_analysis_task(task, context)
            elif task.task_type == "documentation":
                result = await self._execute_documentation_task(task, context)
            else:
                result = await self._execute_general_task(task, context)
            
            # Store task results
            task.agent_contributions[str(self.profile.agent_id)] = result.get('output', '')
            task.output_artifacts.extend(result.get('artifacts', []))
            task.progress = 1.0
            task.status = "completed"
            task.completed_at = datetime.now(timezone.utc)
            
            return result
            
        except Exception as e:
            task.status = "failed"
            task.completed_at = datetime.now(timezone.utc)
            logger.error(f"Task execution failed for agent {self.profile.name}: {e}")
            raise
    
    async def _gather_task_context(self, task: ProjectTask) -> Dict[str, Any]:
        """Gather relevant context for task execution"""
        
        context = {
            'task': task,
            'project_files': [],
            'knowledge_items': [],
            'related_artifacts': []
        }
        
        # Load relevant project files
        if task.project_files:
            for file_path in task.project_files:
                try:
                    file_content = await self.project_vm.read_file_from_vm(
                        self.project_vm, f"/project/files/{file_path}"
                    )
                    context['project_files'].append({
                        'path': file_path,
                        'content': file_content[:5000]  # Limit content size
                    })
                except Exception as e:
                    logger.warning(f"Could not load file {file_path}: {e}")
        
        # Search knowledge base for relevant information
        if task.description:
            knowledge_results = await self.project_knowledge.search_knowledge(
                task.description, limit=5
            )
            context['knowledge_items'] = knowledge_results
        
        return context
    
    async def _execute_research_task(self, task: ProjectTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research-specific task"""
        
        # Analyze available information
        research_summary = []
        
        # Synthesize knowledge base information
        if context['knowledge_items']:
            research_summary.append("## Knowledge Base Findings")
            for item in context['knowledge_items']:
                research_summary.append(f"- {item['content'][:200]}...")
        
        # Analyze project files
        if context['project_files']:
            research_summary.append("## File Analysis")
            for file_info in context['project_files']:
                research_summary.append(f"### {file_info['path']}")
                # Perform simple analysis
                content = file_info['content']
                word_count = len(content.split())
                research_summary.append(f"Word count: {word_count}")
        
        # Create research artifact
        research_content = "\n".join(research_summary)
        artifact_path = await self._create_task_artifact(
            task, "research_findings.md", research_content
        )
        
        return {
            'output': research_content,
            'artifacts': [artifact_path],
            'agent_id': str(self.profile.agent_id),
            'task_type': 'research'
        }
    
    async def _execute_development_task(self, task: ProjectTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute development-specific task"""
        
        # Analyze code files in context
        code_analysis = []
        
        for file_info in context['project_files']:
            if file_info['path'].endswith(('.py', '.js', '.html', '.css')):
                content = file_info['content']
                lines = len(content.split('\n'))
                
                code_analysis.append(f"## {file_info['path']}")
                code_analysis.append(f"Lines of code: {lines}")
                
                # Simple code analysis
                if '.py' in file_info['path']:
                    functions = content.count('def ')
                    classes = content.count('class ')
                    code_analysis.append(f"Functions: {functions}, Classes: {classes}")
        
        # Create development recommendations
        recommendations = []
        if code_analysis:
            recommendations.append("## Development Recommendations")
            recommendations.append("- Code structure analysis completed")
            recommendations.append("- Consider adding documentation")
            recommendations.append("- Implement unit tests")
        
        dev_content = "\n".join(code_analysis + recommendations)
        artifact_path = await self._create_task_artifact(
            task, "development_analysis.md", dev_content
        )
        
        return {
            'output': dev_content,
            'artifacts': [artifact_path],
            'agent_id': str(self.profile.agent_id),
            'task_type': 'development'
        }
    
    async def _execute_analysis_task(self, task: ProjectTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute analysis-specific task"""
        
        analysis_results = []
        
        # Analyze project knowledge
        knowledge_summary = self.project_understanding
        
        analysis_results.append("## Project Analysis")
        analysis_results.append(f"Total knowledge items: {knowledge_summary.get('total_items', 0)}")
        analysis_results.append(f"Knowledge types: {list(knowledge_summary.get('items_by_type', {}).keys())}")
        
        # File distribution analysis
        if 'items_by_type' in knowledge_summary:
            analysis_results.append("## File Distribution")
            for file_type, count in knowledge_summary['items_by_type'].items():
                analysis_results.append(f"- {file_type}: {count} files")
        
        # Concept analysis
        if 'top_concepts' in knowledge_summary:
            analysis_results.append("## Key Concepts")
            top_concepts = list(knowledge_summary['top_concepts'].items())[:10]
            for concept, count in top_concepts:
                analysis_results.append(f"- {concept}: {count} occurrences")
        
        analysis_content = "\n".join(analysis_results)
        artifact_path = await self._create_task_artifact(
            task, "project_analysis.md", analysis_content
        )
        
        return {
            'output': analysis_content,
            'artifacts': [artifact_path],
            'agent_id': str(self.profile.agent_id),
            'task_type': 'analysis'
        }
    
    async def _execute_documentation_task(self, task: ProjectTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute documentation-specific task"""
        
        documentation = []
        
        # Create project overview
        documentation.append(f"# Project Documentation")
        documentation.append(f"Generated by: {self.profile.name}")
        documentation.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        documentation.append("")
        
        # Document project structure
        documentation.append("## Project Structure")
        if context['project_files']:
            for file_info in context['project_files']:
                documentation.append(f"- `{file_info['path']}`")
        
        # Document key concepts
        if 'top_concepts' in self.project_understanding:
            documentation.append("## Key Concepts")
            concepts = list(self.project_understanding['top_concepts'].items())[:5]
            for concept, count in concepts:
                documentation.append(f"- **{concept}**: Used {count} times across project")
        
        # Add insights from knowledge base
        if context['knowledge_items']:
            documentation.append("## Key Insights")
            for item in context['knowledge_items'][:3]:
                if item['item_type'] == 'insight':
                    documentation.append(f"- {item['content']}")
        
        doc_content = "\n".join(documentation)
        artifact_path = await self._create_task_artifact(
            task, "project_documentation.md", doc_content
        )
        
        return {
            'output': doc_content,
            'artifacts': [artifact_path],
            'agent_id': str(self.profile.agent_id),
            'task_type': 'documentation'
        }
    
    async def _execute_general_task(self, task: ProjectTask, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute general task"""
        
        # General task execution based on available context
        output = f"Task '{task.title}' executed by {self.profile.name}\n"
        output += f"Description: {task.description}\n"
        output += f"Available context: {len(context['project_files'])} files, {len(context['knowledge_items'])} knowledge items"
        
        return {
            'output': output,
            'artifacts': [],
            'agent_id': str(self.profile.agent_id),
            'task_type': 'general'
        }
    
    async def _create_task_artifact(self, task: ProjectTask, filename: str, content: str) -> str:
        """Create artifact for task output"""
        
        artifact_dir = f"/project/artifacts/tasks/{task.task_id}"
        await self.project_vm.execute_command(f"mkdir -p {artifact_dir}")
        
        artifact_path = f"{artifact_dir}/{filename}"
        await self.project_vm.write_file_to_vm(self.project_vm, artifact_path, content)
        
        return artifact_path


class ProjectCollaborationManager:
    """
    Manages multi-agent collaboration within projects
    
    Features:
    - Project-specific agent teams
    - Intelligent task delegation
    - Collaborative artifact creation
    - Agent specialization based on project content
    """
    
    def __init__(self, project_id: str, project_vm, project_knowledge, model_loader, memory_manager):
        self.project_id = project_id
        self.project_vm = project_vm
        self.project_knowledge = project_knowledge
        self.model_loader = model_loader
        self.memory_manager = memory_manager
        
        # Collaboration components
        self.project_agents: Dict[UUID, ProjectAgent] = {}
        self.message_bus = MessageBus()
        self.active_tasks: Dict[UUID, ProjectTask] = {}
        
        # Team composition
        self.team_composition: Dict[ProjectRole, UUID] = {}
        self.collaboration_enabled = False
        
        logger.info(f"Project collaboration manager initialized for {project_id}")
    
    async def enable_collaboration(self, team_size: int = 3) -> Dict[str, Any]:
        """Enable multi-agent collaboration for the project"""
        
        try:
            # Create project-specific agent team
            await self._create_project_team(team_size)
            
            # Initialize agents with project context
            await self._initialize_agent_contexts()
            
            # Setup collaboration workspace
            await self._setup_collaboration_workspace()
            
            self.collaboration_enabled = True
            
            return {
                'enabled': True,
                'team_size': len(self.project_agents),
                'agents': [
                    {
                        'id': str(agent.profile.agent_id),
                        'name': agent.profile.name,
                        'capabilities': [cap.value for cap in agent.profile.capabilities],
                        'specialization': agent.specialization_data
                    }
                    for agent in self.project_agents.values()
                ]
            }
            
        except Exception as e:
            logger.error(f"Failed to enable collaboration: {e}")
            return {'enabled': False, 'error': str(e)}
    
    async def _create_project_team(self, team_size: int):
        """Create optimal agent team for project"""
        
        # Analyze project to determine optimal team composition
        knowledge_summary = await self.project_knowledge.get_knowledge_summary()
        
        # Determine required roles based on project content
        required_roles = []
        
        file_types = knowledge_summary.get('items_by_type', {})
        
        if file_types.get('code', 0) > 0:
            required_roles.append(ProjectRole.DEVELOPER)
        
        if file_types.get('research', 0) > 0 or file_types.get('documentation', 0) > 0:
            required_roles.append(ProjectRole.RESEARCHER)
        
        if file_types.get('data', 0) > 0:
            required_roles.append(ProjectRole.ANALYST)
        
        # Always include a project manager for coordination
        if ProjectRole.PROJECT_MANAGER not in required_roles:
            required_roles.append(ProjectRole.PROJECT_MANAGER)
        
        # Add writer for documentation
        if len(required_roles) < team_size:
            required_roles.append(ProjectRole.WRITER)
        
        # Create agents for each required role
        for i, role in enumerate(required_roles[:team_size]):
            agent_profile = await self._create_agent_profile_for_role(role, i)
            
            project_agent = ProjectAgent(
                profile=agent_profile,
                project_id=self.project_id,
                project_vm=self.project_vm,
                project_knowledge=self.project_knowledge,
                model_loader=self.model_loader,
                memory_manager=self.memory_manager,
                message_bus=self.message_bus
            )
            
            self.project_agents[agent_profile.agent_id] = project_agent
            self.team_composition[role] = agent_profile.agent_id
            
            # Register with message bus
            await self.message_bus.register_agent(project_agent)
    
    async def _create_agent_profile_for_role(self, role: ProjectRole, index: int) -> AgentProfile:
        """Create agent profile optimized for specific project role"""
        
        role_configs = {
            ProjectRole.PROJECT_MANAGER: {
                'name': f"ProjectManager_{index}",
                'capabilities': [
                    AgentCapability.PROBLEM_SOLVING,
                    AgentCapability.LOGICAL_REASONING,
                    AgentCapability.TEXT_ANALYSIS
                ],
                'specialization': {
                    AgentCapability.PROBLEM_SOLVING: 0.9,
                    AgentCapability.LOGICAL_REASONING: 0.8
                }
            },
            ProjectRole.RESEARCHER: {
                'name': f"Researcher_{index}",
                'capabilities': [
                    AgentCapability.RESEARCH_SYNTHESIS,
                    AgentCapability.TEXT_ANALYSIS,
                    AgentCapability.DATA_PROCESSING
                ],
                'specialization': {
                    AgentCapability.RESEARCH_SYNTHESIS: 0.95,
                    AgentCapability.TEXT_ANALYSIS: 0.85
                }
            },
            ProjectRole.DEVELOPER: {
                'name': f"Developer_{index}",
                'capabilities': [
                    AgentCapability.CODE_GENERATION,
                    AgentCapability.PROBLEM_SOLVING,
                    AgentCapability.TECHNICAL_DOCUMENTATION
                ],
                'specialization': {
                    AgentCapability.CODE_GENERATION: 0.9,
                    AgentCapability.TECHNICAL_DOCUMENTATION: 0.7
                }
            },
            ProjectRole.ANALYST: {
                'name': f"Analyst_{index}",
                'capabilities': [
                    AgentCapability.DATA_PROCESSING,
                    AgentCapability.LOGICAL_REASONING,
                    AgentCapability.TEXT_ANALYSIS
                ],
                'specialization': {
                    AgentCapability.DATA_PROCESSING: 0.9,
                    AgentCapability.LOGICAL_REASONING: 0.8
                }
            },
            ProjectRole.WRITER: {
                'name': f"Writer_{index}",
                'capabilities': [
                    AgentCapability.CREATIVE_WRITING,
                    AgentCapability.TEXT_ANALYSIS,
                    AgentCapability.TECHNICAL_DOCUMENTATION
                ],
                'specialization': {
                    AgentCapability.CREATIVE_WRITING: 0.9,
                    AgentCapability.TECHNICAL_DOCUMENTATION: 0.8
                }
            }
        }
        
        config = role_configs.get(role, role_configs[ProjectRole.PROJECT_MANAGER])
        
        return AgentProfile(
            agent_id=uuid4(),
            name=config['name'],
            model_id="default",  # Will be loaded in project VM
            capabilities=config['capabilities'],
            specialization_score=config['specialization'],
            collaboration_preference=0.8
        )
    
    async def _initialize_agent_contexts(self):
        """Initialize all agents with project context"""
        
        for agent in self.project_agents.values():
            await agent.initialize_project_context()
    
    async def _setup_collaboration_workspace(self):
        """Setup collaboration workspace in project VM"""
        
        # Create collaboration directory structure
        await self.project_vm.execute_command('''
mkdir -p /project/collaboration/{shared,tasks,artifacts,communication}
        ''')
        
        # Create collaboration configuration
        collab_config = {
            'project_id': self.project_id,
            'collaboration_enabled': True,
            'team_composition': {role.value: str(agent_id) for role, agent_id in self.team_composition.items()},
            'agents': {
                str(agent.profile.agent_id): {
                    'name': agent.profile.name,
                    'capabilities': [cap.value for cap in agent.profile.capabilities],
                    'workspace': agent.agent_workspace
                }
                for agent in self.project_agents.values()
            },
            'initialized_at': datetime.now(timezone.utc).isoformat()
        }
        
        await self.project_vm.write_file_to_vm(
            self.project_vm,
            "/project/collaboration/config.json",
            json.dumps(collab_config, indent=2)
        )
    
    async def execute_collaborative_task(
        self,
        title: str,
        description: str,
        task_type: str = "general",
        project_files: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Execute task using collaborative agent team"""
        
        if not self.collaboration_enabled:
            return {'error': 'Collaboration not enabled for this project'}
        
        # Create project task
        task = ProjectTask(
            title=title,
            description=description,
            task_type=task_type,
            project_files=project_files or []
        )
        
        # Determine best agents for task
        suitable_agents = await self._select_agents_for_task(task)
        
        if not suitable_agents:
            return {'error': 'No suitable agents available for task'}
        
        # Execute task collaboratively
        task.assigned_agents = [agent.profile.agent_id for agent in suitable_agents]
        self.active_tasks[task.task_id] = task
        
        # Execute task with multiple agents
        agent_results = []
        for agent in suitable_agents:
            try:
                result = await agent.execute_project_task(task)
                agent_results.append(result)
            except Exception as e:
                logger.error(f"Agent {agent.profile.name} failed on task: {e}")
        
        # Synthesize results
        final_result = await self._synthesize_task_results(task, agent_results)
        
        return final_result
    
    async def _select_agents_for_task(self, task: ProjectTask) -> List[ProjectAgent]:
        """Select best agents for specific task"""
        
        suitable_agents = []
        
        # Task type to role mapping
        role_preferences = {
            'research': [ProjectRole.RESEARCHER, ProjectRole.ANALYST],
            'development': [ProjectRole.DEVELOPER],
            'analysis': [ProjectRole.ANALYST, ProjectRole.RESEARCHER],
            'documentation': [ProjectRole.WRITER, ProjectRole.RESEARCHER],
            'general': [ProjectRole.PROJECT_MANAGER, ProjectRole.RESEARCHER]
        }
        
        preferred_roles = role_preferences.get(task.task_type, [ProjectRole.PROJECT_MANAGER])
        
        # Select agents based on role and capabilities
        for role in preferred_roles:
            if role in self.team_composition:
                agent_id = self.team_composition[role]
                if agent_id in self.project_agents:
                    suitable_agents.append(self.project_agents[agent_id])
        
        # If no specific role agents, use project manager
        if not suitable_agents and ProjectRole.PROJECT_MANAGER in self.team_composition:
            manager_id = self.team_composition[ProjectRole.PROJECT_MANAGER]
            if manager_id in self.project_agents:
                suitable_agents.append(self.project_agents[manager_id])
        
        return suitable_agents[:2]  # Limit to 2 agents per task
    
    async def _synthesize_task_results(
        self,
        task: ProjectTask,
        agent_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Synthesize results from multiple agents"""
        
        if not agent_results:
            return {'error': 'No agent results to synthesize'}
        
        # Combine outputs
        combined_output = []
        all_artifacts = []
        
        for i, result in enumerate(agent_results):
            agent_id = result.get('agent_id', f'agent_{i}')
            output = result.get('output', '')
            artifacts = result.get('artifacts', [])
            
            combined_output.append(f"## Contribution from Agent {agent_id}")
            combined_output.append(output)
            combined_output.append("")
            
            all_artifacts.extend(artifacts)
        
        # Create synthesis artifact
        synthesis_content = "\n".join(combined_output)
        synthesis_artifact = await self._create_synthesis_artifact(task, synthesis_content)
        
        if synthesis_artifact:
            all_artifacts.append(synthesis_artifact)
        
        return {
            'task_id': str(task.task_id),
            'title': task.title,
            'status': 'completed',
            'participating_agents': len(agent_results),
            'output': synthesis_content,
            'artifacts': all_artifacts,
            'execution_time': (
                task.completed_at - task.started_at
            ).total_seconds() if task.completed_at and task.started_at else 0
        }
    
    async def _create_synthesis_artifact(self, task: ProjectTask, content: str) -> Optional[str]:
        """Create synthesis artifact combining all agent contributions"""
        
        try:
            synthesis_dir = f"/project/artifacts/collaborative/{task.task_id}"
            await self.project_vm.execute_command(f"mkdir -p {synthesis_dir}")
            
            synthesis_path = f"{synthesis_dir}/synthesis.md"
            
            # Add header to synthesis
            header = f"""# Collaborative Task Results
Task: {task.title}
Type: {task.task_type}
Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Participating Agents: {len(task.assigned_agents)}

---

"""
            
            full_content = header + content
            
            await self.project_vm.write_file_to_vm(
                self.project_vm,
                synthesis_path,
                full_content
            )
            
            return synthesis_path
            
        except Exception as e:
            logger.error(f"Failed to create synthesis artifact: {e}")
            return None
    
    async def get_collaboration_status(self) -> Dict[str, Any]:
        """Get current collaboration status"""
        
        return {
            'project_id': self.project_id,
            'collaboration_enabled': self.collaboration_enabled,
            'team_size': len(self.project_agents),
            'active_tasks': len(self.active_tasks),
            'agents': [
                {
                    'id': str(agent.profile.agent_id),
                    'name': agent.profile.name,
                    'role': next(
                        (role.value for role, agent_id in self.team_composition.items() 
                         if agent_id == agent.profile.agent_id), 
                        'unknown'
                    ),
                    'specialization': agent.specialization_data
                }
                for agent in self.project_agents.values()
            ],
            'recent_tasks': [
                {
                    'id': str(task.task_id),
                    'title': task.title,
                    'status': task.status,
                    'type': task.task_type
                }
                for task in list(self.active_tasks.values())[-5:]
            ]
        }
    
    async def disable_collaboration(self) -> bool:
        """Disable collaboration and cleanup agents"""
        
        try:
            # Cleanup agents
            for agent in self.project_agents.values():
                await self.message_bus.unregister_agent(agent.profile.agent_id)
            
            self.project_agents.clear()
            self.team_composition.clear()
            self.active_tasks.clear()
            self.collaboration_enabled = False
            
            logger.info(f"Collaboration disabled for project {self.project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to disable collaboration: {e}")
            return False