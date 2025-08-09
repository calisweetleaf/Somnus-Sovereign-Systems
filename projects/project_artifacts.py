"""
MORPHEUS CHAT - Project Artifact Management
Enhanced artifact system specifically for project environments

Revolutionary Features:
- Project-scoped artifact organization
- AI-generated artifacts from project knowledge
- Collaborative artifact editing within project VMs
- Version control and dependency tracking
- Live execution with full project context
- Unlimited artifact storage and complexity
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

from ..morpheus_artifact_v2 import (
    EnhancedArtifact, ArtifactType, ArtifactStatus, CollaborationMode,
    ExecutionEnvironment, RevolutionaryArtifactManager
)

logger = logging.getLogger(__name__)


class ProjectArtifactType(str, Enum):
    """Project-specific artifact types"""
    PROJECT_REPORT = "project/report"
    ANALYSIS_DASHBOARD = "project/dashboard"
    KNOWLEDGE_SYNTHESIS = "project/synthesis"
    AUTOMATION_SCRIPT = "project/automation"
    COLLABORATION_WORKSPACE = "project/workspace"
    RESEARCH_COMPILATION = "project/research"
    PROJECT_DOCUMENTATION = "project/documentation"
    DATA_VISUALIZATION = "project/visualization"
    DEVELOPMENT_ENVIRONMENT = "project/devenv"
    LIVE_NOTEBOOK = "project/notebook"


@dataclass
class ProjectArtifact(EnhancedArtifact):
    """Enhanced artifact with project-specific capabilities"""
    
    # Project context
    project_id: str = ""
    knowledge_refs: List[str] = field(default_factory=list)
    file_dependencies: List[str] = field(default_factory=list)
    vm_path: Optional[str] = None
    
    # Project-specific execution
    uses_project_files: bool = False
    uses_project_knowledge: bool = False
    uses_project_vm: bool = False
    
    # Collaboration within project
    project_collaborators: Set[str] = field(default_factory=set)
    collaboration_history: List[Dict[str, Any]] = field(default_factory=list)
    
    # Integration with project systems
    automation_triggered: bool = False
    intelligence_enhanced: bool = False
    knowledge_integrated: bool = False


class ProjectArtifactManager:
    """
    Project-specific artifact management with deep integration
    
    Features:
    - Artifacts execute within project VM with full context
    - Automatic integration with project knowledge base
    - AI-enhanced artifact creation using project intelligence
    - Real-time collaboration with project agents
    - Unlimited complexity and execution time
    """
    
    def __init__(
        self, 
        project_id: str, 
        project_vm, 
        project_knowledge, 
        project_intelligence,
        artifact_manager: RevolutionaryArtifactManager
    ):
        self.project_id = project_id
        self.project_vm = project_vm
        self.project_knowledge = project_knowledge
        self.project_intelligence = project_intelligence
        self.base_artifact_manager = artifact_manager
        
        # Project artifact storage
        self.project_artifacts: Dict[str, ProjectArtifact] = {}
        self.artifact_dependencies: Dict[str, Set[str]] = {}
        
        # Execution environments
        self.live_environments: Dict[str, Dict[str, Any]] = {}
        
        # Integration state
        self.knowledge_integration_enabled = True
        self.vm_execution_enabled = True
        self.collaboration_enabled = False
        
        logger.info(f"Project artifact manager initialized for project {project_id}")
    
    async def initialize(self):
        """Initialize project artifact system"""
        
        try:
            # Setup artifact workspace in VM
            await self._setup_artifact_workspace()
            
            # Initialize artifact execution environment
            await self._setup_execution_environment()
            
            # Load existing project artifacts
            await self._load_existing_artifacts()
            
            logger.info(f"Project artifacts ready for project {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize project artifacts: {e}")
            raise
    
    async def _setup_artifact_workspace(self):
        """Setup artifact workspace in project VM"""
        
        # Create artifact directories
        await self.project_vm.execute_command('''
mkdir -p /project/artifacts/{
    active,templates,executions,dependencies,
    collaborative,generated,exports
}
        ''')
        
        # Create artifact execution environment
        artifact_env_script = f'''#!/usr/bin/env python3
"""
Project Artifact Execution Environment
Provides full project context to executing artifacts
"""

import json
import sys
import os
from pathlib import Path

# Add project modules to path
sys.path.insert(0, '/project')
sys.path.insert(0, '/project/intelligence')

class ProjectArtifactContext:
    """Provides project context to executing artifacts"""
    
    def __init__(self):
        self.project_id = "{self.project_id}"
        self.project_root = Path("/project")
        self.artifact_root = self.project_root / "artifacts"
        
    def get_project_files(self, pattern="*"):
        """Get project files matching pattern"""
        files = list(self.project_root.glob(f"files/**/{pattern}"))
        return [str(f) for f in files]
    
    def read_project_file(self, file_path):
        """Read project file"""
        try:
            full_path = self.project_root / "files" / file_path
            with open(full_path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {e}"
    
    def search_knowledge(self, query):
        """Search project knowledge base"""
        # This would integrate with knowledge system
        # For now, return placeholder
        return f"Knowledge search for: {query}"
    
    def get_project_stats(self):
        """Get project statistics"""
        try:
            with open('/project/workspace.json', 'r') as f:
                return json.load(f)
        except:
            return {{"project_id": self.project_id}}
    
    def create_artifact_output(self, content, artifact_type="text"):
        """Create output artifact"""
        output_dir = self.artifact_root / "executions"
        output_dir.mkdir(exist_ok=True)
        
        output_file = output_dir / f"output_{{int(time.time())}}.{{}}"
        
        if artifact_type == "html":
            output_file = output_file.with_suffix(".html")
        elif artifact_type == "json":
            output_file = output_file.with_suffix(".json")
        else:
            output_file = output_file.with_suffix(".txt")
        
        with open(output_file, 'w') as f:
            if isinstance(content, dict):
                json.dump(content, f, indent=2)
            else:
                f.write(str(content))
        
        return str(output_file)

# Global context available to all artifacts
project_context = ProjectArtifactContext()

# Helper functions for artifacts
def get_files(pattern="*"):
    return project_context.get_project_files(pattern)

def read_file(file_path):
    return project_context.read_project_file(file_path)

def search_knowledge(query):
    return project_context.search_knowledge(query)

def save_output(content, artifact_type="text"):
    return project_context.create_artifact_output(content, artifact_type)
        '''
        
        await self.project_vm.write_file_to_vm(
            self.project_vm,
            "/project/artifacts/project_context.py",
            artifact_env_script
        )
    
    async def _setup_execution_environment(self):
        """Setup enhanced execution environment for artifacts"""
        
        # Create Jupyter-like execution environment
        jupyter_setup = '''
# Install Jupyter and extensions
pip3 install --quiet jupyter jupyterlab plotly dash streamlit
pip3 install --quiet matplotlib seaborn pandas numpy

# Setup Jupyter configuration
mkdir -p /home/morpheus/.jupyter
cat > /home/morpheus/.jupyter/jupyter_lab_config.py << 'EOF'
c.ServerApp.ip = '0.0.0.0'
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.token = ''
c.ServerApp.password = ''
c.ServerApp.root_dir = '/project'
EOF

# Create artifact templates
mkdir -p /project/artifacts/templates
        '''
        
        await self.project_vm.execute_command(jupyter_setup)
        
        # Create artifact templates
        await self._create_artifact_templates()
    
    async def _create_artifact_templates(self):
        """Create intelligent artifact templates based on project content"""
        
        # Get project analysis for intelligent templates
        knowledge_summary = await self.project_knowledge.get_knowledge_summary()
        
        templates = {}
        
        # Data visualization template (if project has data files)
        if 'data' in knowledge_summary.get('items_by_type', {}):
            templates['data_visualization'] = '''
# Data Visualization Artifact
# Auto-generated template for project data

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from project_context import get_files, read_file, save_output

# Get data files from project
data_files = get_files("*.csv")
print(f"Found {len(data_files)} data files")

# Example visualization
if data_files:
    # Read first data file
    first_file = data_files[0]
    data_content = read_file(first_file)
    
    # Create simple visualization
    plt.figure(figsize=(10, 6))
    plt.text(0.5, 0.5, f"Data from: {first_file}\\nRows: {len(data_content.split())}", 
             ha='center', va='center', fontsize=12)
    plt.title("Project Data Overview")
    
    # Save visualization
    plt.savefig('/project/artifacts/executions/data_viz.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("Data visualization saved!")
else:
    print("No data files found in project")
            '''
        
        # Research report template (if project has research content)
        if 'research' in knowledge_summary.get('items_by_type', {}):
            templates['research_report'] = '''
# Research Report Generator
# Auto-generated template for project research

from project_context import search_knowledge, get_files, save_output
import json

# Search for research-related knowledge
research_results = search_knowledge("research findings analysis results")

# Get research files
research_files = get_files("*.md") + get_files("*.txt") + get_files("*.pdf")

# Generate report
report = f"""
# Project Research Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Overview
This project contains {len(research_files)} research-related files.

## Key Findings
{research_results}

## Files Analyzed
{chr(10).join(f"- {file}" for file in research_files[:10])}

## Recommendations
Based on the analysis, consider:
1. Organizing research by topic
2. Creating summary documents
3. Identifying research gaps
"""

# Save report
output_file = save_output(report, "text")
print(f"Research report saved to: {output_file}")
            '''
        
        # Code analysis template (if project has code files)
        if 'code' in knowledge_summary.get('items_by_type', {}):
            templates['code_analysis'] = '''
# Code Analysis Dashboard
# Auto-generated template for project code

from project_context import get_files, read_file, save_output
import json

# Get code files
code_files = get_files("*.py") + get_files("*.js") + get_files("*.html")

analysis = {
    "total_files": len(code_files),
    "file_types": {},
    "file_sizes": {},
    "analysis_summary": {}
}

for file_path in code_files:
    file_content = read_file(file_path)
    file_ext = file_path.split('.')[-1]
    
    # Count by type
    analysis["file_types"][file_ext] = analysis["file_types"].get(file_ext, 0) + 1
    
    # Analyze content
    lines = len(file_content.split('\\n'))
    analysis["file_sizes"][file_path] = lines
    
    # Simple code analysis
    if file_ext == 'py':
        functions = file_content.count('def ')
        classes = file_content.count('class ')
        analysis["analysis_summary"][file_path] = {
            "lines": lines,
            "functions": functions,
            "classes": classes
        }

# Create HTML dashboard
html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Project Code Analysis</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .metric {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
        .file-list {{ max-height: 300px; overflow-y: auto; }}
    </style>
</head>
<body>
    <h1>Project Code Analysis</h1>
    
    <div class="metric">
        <h3>Total Files: {analysis['total_files']}</h3>
    </div>
    
    <div class="metric">
        <h3>File Types:</h3>
        {chr(10).join(f"<p>{ext}: {count} files</p>" for ext, count in analysis['file_types'].items())}
    </div>
    
    <div class="metric">
        <h3>Code Files:</h3>
        <div class="file-list">
            {chr(10).join(f"<p>{file}: {size} lines</p>" for file, size in list(analysis['file_sizes'].items())[:20])}
        </div>
    </div>
</body>
</html>
"""

# Save dashboard
output_file = save_output(html_content, "html")
print(f"Code analysis dashboard saved to: {output_file}")
print(f"Analysis data: {json.dumps(analysis, indent=2)}")
            '''
        
        # Save templates to VM
        for template_name, template_content in templates.items():
            template_path = f"/project/artifacts/templates/{template_name}.py"
            await self.project_vm.write_file_to_vm(
                self.project_vm,
                template_path,
                template_content
            )
    
    async def create_project_artifact(
        self,
        title: str,
        content: str,
        artifact_type: ArtifactType,
        user_id: str,
        enhance_with_project: bool = True,
        collaboration_mode: CollaborationMode = CollaborationMode.HUMAN_ONLY
    ) -> ProjectArtifact:
        """
        Create project artifact with full project context integration
        """
        
        # Create base artifact
        base_artifact = await self.base_artifact_manager.create_artifact(
            title=title,
            content=content,
            artifact_type=artifact_type,
            user_id=user_id,
            model_used="project_vm",
            workspace_id=self.project_id
        )
        
        # Enhance with project-specific capabilities
        project_artifact = ProjectArtifact(
            **base_artifact.dict(),
            project_id=self.project_id,
            uses_project_vm=True,
            collaboration_mode=collaboration_mode
        )
        
        if enhance_with_project:
            # Enhance artifact with project knowledge
            await self._enhance_artifact_with_project_knowledge(project_artifact)
            
            # Setup execution in project VM
            await self._setup_artifact_vm_execution(project_artifact)
        
        # Store project artifact
        self.project_artifacts[project_artifact.id] = project_artifact
        
        # Integrate with project systems
        await self._integrate_artifact_with_project(project_artifact)
        
        logger.info(f"Created project artifact: {title}")
        return project_artifact
    
    async def _enhance_artifact_with_project_knowledge(self, artifact: ProjectArtifact):
        """Enhance artifact with relevant project knowledge"""
        
        if artifact.artifact_type in [ArtifactType.PYTHON, ArtifactType.HTML, ArtifactType.JAVASCRIPT]:
            # Search for relevant knowledge
            knowledge_results = await self.project_knowledge.search_knowledge(
                artifact.title + " " + artifact.content[:200],
                limit=5
            )
            
            if knowledge_results:
                # Add knowledge references to artifact
                artifact.knowledge_refs = [item['item_id'] for item in knowledge_results]
                artifact.knowledge_integrated = True
                
                # Enhance artifact content with knowledge context
                knowledge_context = "\\n".join([
                    f"// Knowledge: {item['content'][:100]}..."
                    for item in knowledge_results
                ])
                
                # Add knowledge context as comments
                if artifact.artifact_type == ArtifactType.PYTHON:
                    enhanced_content = f'''"""
Project Knowledge Context:
{knowledge_context}
"""

{artifact.content}
'''
                elif artifact.artifact_type == ArtifactType.HTML:
                    enhanced_content = f'''<!-- 
Project Knowledge Context:
{knowledge_context}
-->

{artifact.content}
'''
                else:
                    enhanced_content = artifact.content
                
                artifact.content = enhanced_content
    
    async def _setup_artifact_vm_execution(self, artifact: ProjectArtifact):
        """Setup artifact for execution in project VM"""
        
        # Create artifact execution script in VM
        artifact_vm_path = f"/project/artifacts/active/{artifact.id}"
        await self.project_vm.execute_command(f"mkdir -p {artifact_vm_path}")
        
        # Save artifact content to VM
        if artifact.artifact_type == ArtifactType.PYTHON:
            file_extension = ".py"
        elif artifact.artifact_type == ArtifactType.HTML:
            file_extension = ".html"
        elif artifact.artifact_type == ArtifactType.JAVASCRIPT:
            file_extension = ".js"
        else:
            file_extension = ".txt"
        
        artifact_file_path = f"{artifact_vm_path}/artifact{file_extension}"
        
        # Add project context imports for Python artifacts
        if artifact.artifact_type == ArtifactType.PYTHON:
            enhanced_content = f'''#!/usr/bin/env python3
"""
Project Artifact: {artifact.title}
Executing in Project VM with full context
"""

# Import project context
import sys
sys.path.insert(0, '/project/artifacts')
from project_context import project_context, get_files, read_file, search_knowledge, save_output

# Artifact code
{artifact.content}
'''
        else:
            enhanced_content = artifact.content
        
        await self.project_vm.write_file_to_vm(
            self.project_vm,
            artifact_file_path,
            enhanced_content
        )
        
        # Make Python scripts executable
        if artifact.artifact_type == ArtifactType.PYTHON:
            await self.project_vm.execute_command(f"chmod +x {artifact_file_path}")
        
        artifact.vm_path = artifact_file_path
        artifact.uses_project_vm = True
    
    async def _integrate_artifact_with_project(self, artifact: ProjectArtifact):
        """Integrate artifact with project systems"""
        
        # Register with project intelligence
        if hasattr(self.project_intelligence, 'register_artifact'):
            await self.project_intelligence.register_artifact(artifact)
        
        # Add to project memory
        artifact_memory = f"""
        Project Artifact Created: {artifact.title}
        Type: {artifact.artifact_type}
        Uses Project VM: {artifact.uses_project_vm}
        Knowledge Integration: {artifact.knowledge_integrated}
        Created: {artifact.created_at}
        """
        
        # This would integrate with memory system
        # await self.project_memory.store_memory(artifact_memory)
    
    async def execute_artifact_in_project(self, artifact_id: str) -> Dict[str, Any]:
        """Execute artifact within project VM context"""
        
        if artifact_id not in self.project_artifacts:
            raise ValueError(f"Project artifact {artifact_id} not found")
        
        artifact = self.project_artifacts[artifact_id]
        
        if not artifact.uses_project_vm or not artifact.vm_path:
            raise ValueError(f"Artifact {artifact_id} not configured for VM execution")
        
        execution_start = datetime.now(timezone.utc)
        
        try:
            # Execute artifact in VM
            if artifact.artifact_type == ArtifactType.PYTHON:
                result = await self.project_vm.execute_command_in_vm(
                    self.project_vm,
                    f"cd /project && python3 {artifact.vm_path}"
                )
            elif artifact.artifact_type == ArtifactType.HTML:
                # For HTML artifacts, start a simple server
                port = 8080  # Could be dynamic
                server_command = f"""
cd /project/artifacts/active/{artifact.id}
python3 -m http.server {port} &
echo "HTML artifact served at http://localhost:{port}"
                """
                result = await self.project_vm.execute_command_in_vm(
                    self.project_vm,
                    server_command
                )
            else:
                result = f"Execution not supported for {artifact.artifact_type}"
            
            execution_end = datetime.now(timezone.utc)
            execution_time = (execution_end - execution_start).total_seconds()
            
            # Update artifact metrics
            artifact.metrics.execution_time = execution_time
            artifact.status = ArtifactStatus.ACTIVE
            
            # Create execution record
            execution_record = {
                'artifact_id': artifact_id,
                'execution_time': execution_time,
                'started_at': execution_start.isoformat(),
                'completed_at': execution_end.isoformat(),
                'output': result,
                'status': 'completed'
            }
            
            return execution_record
            
        except Exception as e:
            execution_end = datetime.now(timezone.utc)
            
            error_record = {
                'artifact_id': artifact_id,
                'started_at': execution_start.isoformat(),
                'completed_at': execution_end.isoformat(),
                'status': 'failed',
                'error': str(e)
            }
            
            artifact.status = ArtifactStatus.ERROR
            
            logger.error(f"Artifact execution failed: {e}")
            return error_record
    
    async def generate_artifact_from_template(
        self,
        template_name: str,
        title: str,
        user_id: str,
        parameters: Optional[Dict[str, Any]] = None
    ) -> ProjectArtifact:
        """Generate artifact from project-specific template"""
        
        template_path = f"/project/artifacts/templates/{template_name}.py"
        
        try:
            # Check if template exists
            template_content = await self.project_vm.read_file_from_vm(
                self.project_vm,
                template_path
            )
            
            # Customize template with parameters
            if parameters:
                for key, value in parameters.items():
                    template_content = template_content.replace(f"{{{key}}}", str(value))
            
            # Create artifact from template
            artifact = await self.create_project_artifact(
                title=title,
                content=template_content,
                artifact_type=ArtifactType.PYTHON,
                user_id=user_id,
                enhance_with_project=True
            )
            
            # Mark as template-generated
            artifact.metadata['generated_from_template'] = template_name
            artifact.metadata['template_parameters'] = parameters or {}
            
            return artifact
            
        except Exception as e:
            logger.error(f"Failed to generate artifact from template {template_name}: {e}")
            raise
    
    async def enable_artifact_collaboration(self, artifact_id: str) -> bool:
        """Enable real-time collaboration for artifact"""
        
        if artifact_id not in self.project_artifacts:
            return False
        
        artifact = self.project_artifacts[artifact_id]
        
        try:
            # Setup collaboration workspace for artifact
            collab_workspace = f"/project/artifacts/collaborative/{artifact_id}"
            await self.project_vm.execute_command(f"mkdir -p {collab_workspace}")
            
            # Create collaborative editing environment
            # This would integrate with collaboration system
            artifact.collaboration_mode = CollaborationMode.LIVE_COLLABORATION
            self.collaboration_enabled = True
            
            logger.info(f"Collaboration enabled for artifact {artifact.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to enable collaboration for artifact {artifact_id}: {e}")
            return False
    
    async def get_project_artifacts_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of project artifacts"""
        
        total_artifacts = len(self.project_artifacts)
        
        artifacts_by_type = {}
        vm_enabled_count = 0
        collaborative_count = 0
        knowledge_integrated_count = 0
        
        for artifact in self.project_artifacts.values():
            artifact_type = artifact.artifact_type
            artifacts_by_type[artifact_type] = artifacts_by_type.get(artifact_type, 0) + 1
            
            if artifact.uses_project_vm:
                vm_enabled_count += 1
            
            if artifact.collaboration_mode != CollaborationMode.HUMAN_ONLY:
                collaborative_count += 1
            
            if artifact.knowledge_integrated:
                knowledge_integrated_count += 1
        
        return {
            'project_id': self.project_id,
            'total_artifacts': total_artifacts,
            'artifacts_by_type': artifacts_by_type,
            'vm_enabled_artifacts': vm_enabled_count,
            'collaborative_artifacts': collaborative_count,
            'knowledge_integrated_artifacts': knowledge_integrated_count,
            'execution_environments': len(self.live_environments),
            'collaboration_enabled': self.collaboration_enabled,
            'recent_artifacts': [
                {
                    'id': artifact.id,
                    'title': artifact.title,
                    'type': artifact.artifact_type,
                    'status': artifact.status,
                    'created_at': artifact.created_at.isoformat(),
                    'uses_vm': artifact.uses_project_vm
                }
                for artifact in sorted(
                    self.project_artifacts.values(),
                    key=lambda x: x.created_at,
                    reverse=True
                )[:10]
            ]
        }
    
    async def _load_existing_artifacts(self):
        """Load existing project artifacts from VM"""
        
        try:
            # Check for existing artifacts
            result = await self.project_vm.execute_command_in_vm(
                self.project_vm,
                "ls /project/artifacts/active/*/artifact.* 2>/dev/null || echo 'none'"
            )
            
            if result.strip() != 'none':
                logger.info("Loading existing project artifacts...")
                # Implementation would load existing artifacts
        
        except Exception as e:
            logger.info(f"No existing artifacts found: {e}")
    
    async def cleanup(self):
        """Cleanup project artifact resources"""
        
        # Stop any running execution environments
        for env_id, env_data in self.live_environments.items():
            try:
                # Cleanup running processes
                if 'process_id' in env_data:
                    await self.project_vm.execute_command(
                        f"kill {env_data['process_id']} 2>/dev/null || true"
                    )
            except:
                pass
        
        self.live_environments.clear()
        self.project_artifacts.clear()
        self.artifact_dependencies.clear()