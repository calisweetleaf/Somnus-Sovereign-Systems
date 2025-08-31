"""
MORPHEUS CHAT - Project Core Management
Each Project = Virtual Machine + Loaded Model + Autonomous Intelligence

Revolutionary Architecture:
- Projects ARE virtual machines, not chat + files
- Each project has its own loaded model
- Autonomous file management and knowledge synthesis
- Unlimited growth, zero constraints
- Complete SaaS destruction
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

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ProjectType(str, Enum):
    """Auto-detected project types"""
    RESEARCH = "research"
    DEVELOPMENT = "development"
    BUSINESS = "business"
    CREATIVE = "creative"
    DATA_ANALYSIS = "data_analysis"
    EDUCATION = "education"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class ProjectStatus(str, Enum):
    """Project lifecycle status"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    PROCESSING = "processing"
    SUSPENDED = "suspended"
    ARCHIVED = "archived"
    ERROR = "error"


@dataclass
class ProjectSpecs:
    """Project VM specifications"""
    vcpus: int = 4
    memory_gb: int = 8
    storage_gb: int = 200
    gpu_enabled: bool = True
    model_id: str = "default"
    enable_networking: bool = True
    enable_collaboration: bool = False


class ProjectMetadata(BaseModel):
    """Core project metadata and configuration"""
    project_id: UUID = Field(default_factory=uuid4)
    name: str = Field(description="Human-readable project name")
    description: Optional[str] = None
    project_type: ProjectType = ProjectType.UNKNOWN
    status: ProjectStatus = ProjectStatus.INITIALIZING
    
    # VM and model configuration
    specs: ProjectSpecs = Field(default_factory=ProjectSpecs)
    vm_id: Optional[UUID] = None
    loaded_model: Optional[str] = None
    
    # Autonomous intelligence state
    files_processed: int = 0
    knowledge_items: int = 0
    auto_insights: List[str] = Field(default_factory=list)
    project_understanding: Dict[str, Any] = Field(default_factory=dict)
    
    # Usage tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_active: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    total_interactions: int = 0
    
    # Optional capabilities
    collaboration_enabled: bool = False
    automation_enabled: bool = False
    web_access_enabled: bool = False
    
    # File organization state
    file_categories: Dict[str, List[str]] = Field(default_factory=dict)
    knowledge_graph: Dict[str, Any] = Field(default_factory=dict)
    
    # User preferences learned over time
    user_patterns: Dict[str, Any] = Field(default_factory=dict)


class ProjectManager:
    """
    Core project management system
    
    Orchestrates project VMs, model loading, and autonomous intelligence
    """
    
    def __init__(self, projects_dir: Path, vm_manager, model_loader, memory_manager):
        self.projects_dir = projects_dir
        self.projects_dir.mkdir(parents=True, exist_ok=True)
        
        # Core system integrations
        self.vm_manager = vm_manager
        self.model_loader = model_loader
        self.memory_manager = memory_manager
        
        # Project tracking
        self.active_projects: Dict[UUID, ProjectMetadata] = {}
        self.project_vms: Dict[UUID, Any] = {}
        
        # Background intelligence tasks
        self.intelligence_tasks: Set[asyncio.Task] = set()
        
        logger.info("Project Manager initialized")
    
    async def create_project(
        self,
        name: str,
        description: Optional[str] = None,
        specs: Optional[ProjectSpecs] = None,
        user_id: Optional[str] = None
    ) -> ProjectMetadata:
        """
        Create new project with dedicated VM and loaded model
        
        This creates a complete project environment, not just a chat room
        """
        specs = specs or ProjectSpecs()
        
        project = ProjectMetadata(
            name=name,
            description=description,
            specs=specs
        )
        
        try:
            # 1. Create dedicated VM for this project
            vm_instance = await self.vm_manager.create_project_vm(
                project_id=project.project_id,
                vm_specs=specs,
                project_name=name
            )
            
            project.vm_id = vm_instance.vm_id
            project.status = ProjectStatus.INITIALIZING
            
            # 2. Load model directly into project VM
            model_loaded = await self.model_loader.load_model_into_vm(
                vm_id=vm_instance.vm_id,
                model_id=specs.model_id,
                project_context=True
            )
            
            if model_loaded:
                project.loaded_model = specs.model_id
                project.status = ProjectStatus.ACTIVE
            else:
                project.status = ProjectStatus.ERROR
                raise Exception(f"Failed to load model {specs.model_id}")
            
            # 3. Initialize project intelligence systems
            await self._initialize_project_intelligence(project, vm_instance)
            
            # 4. Setup project workspace in VM
            await self._setup_project_workspace(project, vm_instance)
            
            # 5. Track and persist project
            self.active_projects[project.project_id] = project
            self.project_vms[project.project_id] = vm_instance
            await self._persist_project_metadata(project)
            
            # 6. Start autonomous intelligence monitoring
            intelligence_task = asyncio.create_task(
                self._run_project_intelligence(project.project_id)
            )
            self.intelligence_tasks.add(intelligence_task)
            
            logger.info(f"Created project {name} with VM {vm_instance.vm_id}")
            return project
            
        except Exception as e:
            project.status = ProjectStatus.ERROR
            logger.error(f"Failed to create project {name}: {e}")
            raise
    
    async def _initialize_project_intelligence(self, project: ProjectMetadata, vm_instance):
        """Initialize autonomous intelligence systems in project VM"""
        
        # Install project intelligence toolkit in VM
        intelligence_setup = '''
# Project Intelligence Initialization
mkdir -p /project/{
    files,workspace,knowledge,memory,automation,
    intelligence,logs,artifacts,collaboration
}

# Install analysis tools
pip install --quiet numpy pandas matplotlib seaborn plotly
pip install --quiet scikit-learn transformers sentence-transformers
pip install --quiet gitpython requests beautifulsoup4

# Setup project environment variables
echo 'export PROJECT_ID="{project_id}"' >> ~/.bashrc
echo 'export PROJECT_NAME="{project_name}"' >> ~/.bashrc
echo 'export PROJECT_TYPE="{project_type}"' >> ~/.bashrc
'''.format(
            project_id=project.project_id,
            project_name=project.name,
            project_type=project.project_type
        )
        
        await vm_instance.execute_command(intelligence_setup)
        
        # Create project intelligence script
        intelligence_script = f'''#!/usr/bin/env python3
"""
Project Intelligence System - Autonomous file and knowledge management
Running inside project VM: {project.project_id}
"""

import os
import json
import asyncio
from pathlib import Path
from datetime import datetime

class ProjectIntelligence:
    def __init__(self):
        self.project_id = "{project.project_id}"
        self.project_path = Path("/project")
        self.files_path = self.project_path / "files"
        self.knowledge_path = self.project_path / "knowledge"
        
    async def auto_process_files(self):
        """Automatically process and organize uploaded files"""
        while True:
            try:
                # Monitor for new files
                new_files = list(self.files_path.glob("**/*"))
                
                for file_path in new_files:
                    if file_path.is_file() and not self._is_processed(file_path):
                        await self._analyze_and_organize_file(file_path)
                
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Intelligence error: {{e}}")
                await asyncio.sleep(10)
    
    def _is_processed(self, file_path: Path) -> bool:
        """Check if file has been processed"""
        marker_file = self.knowledge_path / f"{{file_path.name}}.processed"
        return marker_file.exists()
    
    async def _analyze_and_organize_file(self, file_path: Path):
        """Analyze file and add to project knowledge"""
        # This will be enhanced by project_intelligence.py
        print(f"Processing new file: {{file_path}}")
        
        # Mark as processed
        marker_file = self.knowledge_path / f"{{file_path.name}}.processed"
        marker_file.touch()

if __name__ == "__main__":
    intelligence = ProjectIntelligence()
    asyncio.run(intelligence.auto_process_files())
        '''
        
        await vm_instance.write_file("/project/intelligence/auto_processor.py", intelligence_script)
        await vm_instance.execute_command("chmod +x /project/intelligence/auto_processor.py")
    
    async def _setup_project_workspace(self, project: ProjectMetadata, vm_instance):
        """Setup complete project workspace in VM"""
        
        workspace_config = {
            "project_id": str(project.project_id),
            "project_name": project.name,
            "project_type": project.project_type,
            "created_at": project.created_at.isoformat(),
            "model_loaded": project.loaded_model,
            "workspace_version": "1.0"
        }
        
        config_json = json.dumps(workspace_config, indent=2)
        await vm_instance.write_file("/project/workspace.json", config_json)
        
        # Create project README
        readme_content = f'''# {project.name}

**Project ID**: {project.project_id}
**Type**: {project.project_type}
**Created**: {project.created_at.strftime("%Y-%m-%d %H:%M:%S")}
**Model**: {project.loaded_model}

## Project Structure

- `/project/files/` - All uploaded files (auto-organized by AI)
- `/project/knowledge/` - AI-generated knowledge base
- `/project/workspace/` - Working files and artifacts
- `/project/memory/` - Project memory and context
- `/project/automation/` - Automated tasks and scripts
- `/project/collaboration/` - Multi-agent collaboration workspace

## AI Capabilities

This project has autonomous AI that:
- Automatically organizes uploaded files
- Builds knowledge base from all content
- Learns project patterns over time
- Suggests improvements and optimizations
- Can enable collaboration with multiple AI agents

## Getting Started

1. Upload files to `/project/files/`
2. AI automatically processes and organizes them
3. Ask questions about your project - AI knows everything
4. Enable collaboration for complex tasks
5. Let AI automate repetitive work

**This is your unlimited, autonomous project environment.**
        '''
        
        await vm_instance.write_file("/project/README.md", readme_content)
    
    async def _run_project_intelligence(self, project_id: UUID):
        """Run continuous autonomous intelligence for project"""
        
        while project_id in self.active_projects:
            try:
                project = self.active_projects[project_id]
                vm_instance = self.project_vms[project_id]
                
                # Update last active
                project.last_active = datetime.now(timezone.utc)
                
                # Run intelligence checks
                await self._check_project_health(project, vm_instance)
                await self._update_project_understanding(project, vm_instance)
                
                # Sleep for 30 seconds between intelligence cycles
                await asyncio.sleep(30)
                
            except Exception as e:
                logger.error(f"Intelligence error for project {project_id}: {e}")
                await asyncio.sleep(60)  # Longer sleep on error
    
    async def _check_project_health(self, project: ProjectMetadata, vm_instance):
        """Check project VM and model health"""
        
        # Check VM status
        vm_status = await vm_instance.get_status()
        if vm_status != "running":
            logger.warning(f"Project {project.project_id} VM not running")
            return
        
        # Check model status
        model_status = await self.model_loader.check_model_status(
            vm_id=vm_instance.vm_id,
            model_id=project.loaded_model
        )
        
        if not model_status.get("healthy", False):
            logger.warning(f"Project {project.project_id} model unhealthy")
    
    async def _update_project_understanding(self, project: ProjectMetadata, vm_instance):
        """Update AI's understanding of the project"""
        
        # Get current file count
        file_count_result = await vm_instance.execute_command(
            "find /project/files -type f | wc -l"
        )
        
        try:
            current_files = int(file_count_result.strip())
            if current_files != project.files_processed:
                project.files_processed = current_files
                logger.info(f"Project {project.project_id} now has {current_files} files")
        except:
            pass
    
    async def _persist_project_metadata(self, project: ProjectMetadata):
        """Persist project metadata to disk"""
        
        project_file = self.projects_dir / f"{project.project_id}.json"
        project_data = project.dict()
        
        # Convert datetime objects to strings for JSON serialization
        project_data["created_at"] = project.created_at.isoformat()
        project_data["last_active"] = project.last_active.isoformat()
        
        with open(project_file, 'w') as f:
            json.dump(project_data, f, indent=2, default=str)
    
    async def get_project(self, project_id: UUID) -> Optional[ProjectMetadata]:
        """Get project by ID"""
        return self.active_projects.get(project_id)
    
    async def list_projects(self, user_id: Optional[str] = None) -> List[ProjectMetadata]:
        """List all projects for user"""
        # For now, return all active projects
        # Later can filter by user_id
        return list(self.active_projects.values())
    
    async def delete_project(self, project_id: UUID) -> bool:
        """Delete project and cleanup VM"""
        
        if project_id not in self.active_projects:
            return False
        
        try:
            # Stop intelligence monitoring
            intelligence_tasks_to_remove = []
            for task in self.intelligence_tasks:
                if not task.done():
                    task.cancel()
                intelligence_tasks_to_remove.append(task)
            
            for task in intelligence_tasks_to_remove:
                self.intelligence_tasks.discard(task)
            
            # Cleanup VM
            if project_id in self.project_vms:
                vm_instance = self.project_vms[project_id]
                await self.vm_manager.destroy_vm(vm_instance.vm_id)
                del self.project_vms[project_id]
            
            # Remove from active projects
            del self.active_projects[project_id]
            
            # Remove metadata file
            project_file = self.projects_dir / f"{project_id}.json"
            if project_file.exists():
                project_file.unlink()
            
            logger.info(f"Deleted project {project_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete project {project_id}: {e}")
            return False
    
    async def cleanup(self):
        """Cleanup all projects and resources"""
        
        # Cancel all intelligence tasks
        for task in self.intelligence_tasks:
            if not task.done():
                task.cancel()
        
        # Cleanup all VMs
        for vm_instance in self.project_vms.values():
            try:
                await self.vm_manager.destroy_vm(vm_instance.vm_id)
            except Exception as e:
                logger.error(f"Failed to cleanup VM {vm_instance.vm_id}: {e}")
        
        self.active_projects.clear()
        self.project_vms.clear()
        self.intelligence_tasks.clear()