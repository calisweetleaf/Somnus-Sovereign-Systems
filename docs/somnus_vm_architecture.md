# SOMNUS SOVEREIGN SYSTEMS - Virtual Machine Architecture Documentation

## Overview

The Somnus VM Architecture implements a revolutionary **persistent AI computing environment** where AI agents operate from never-reset virtual machines that accumulate capabilities, tools, and knowledge over time. This documentation covers the complete backend virtual machine system that enables unlimited AI execution and cross-session continuity.

## Architecture Philosophy

### Core Principles

1. **Persistent AI Intelligence** - VMs never reset, maintaining AI state permanently
2. **Progressive Capability Building** - Tools and skills accumulate over time  
3. **Dual-Layer Execution** - Persistent VM intelligence layer + disposable container overlays
4. **Complete Digital Sovereignty** - Local-only execution, no cloud dependencies
5. **Memory-Driven Operations** - Semantic memory system enables intelligent VM orchestration

### Revolutionary Design

**Unlike traditional AI services that reset after every session, Somnus VMs:**

- Maintain persistent file systems and installed tools
- Remember previous sessions through integrated memory core
- Learn and optimize their development environments
- Build personal libraries and automation scripts
- Collaborate with other AI VMs in real-time

## System Components

### 1. VM Supervisor (`vm_supervisor.py`)

**Purpose**: OS-level supervisor managing fleets of persistent AI computers

**Key Features**:

- **Libvirt Integration**: Direct KVM/QEMU hypervisor control
- **Dynamic Resource Scaling**: CPU/memory adjustment during runtime
- **Snapshot Management**: Point-in-time VM state preservation
- **Intelligent Monitoring**: Real-time stats collection from in-VM agents
- **Security Isolation**: Network-isolated execution environments

**Resource Profiles**:

```python
profiles = {
    "idle": ResourceProfile(vcpus=1, memory_gb=4),
    "coding": ResourceProfile(vcpus=4, memory_gb=8), 
    "research": ResourceProfile(vcpus=2, memory_gb=6),
    "media_creation": ResourceProfile(vcpus=6, memory_gb=16, gpu_enabled=True)
}
```

**VM Lifecycle Management**:

- Creation from base OS images
- Dynamic hardware reconfiguration
- Snapshot-based capability preservation
- Graceful shutdown and cleanup

### 2. In-VM Somnus Agent (`somnus_agent.py`)

**Purpose**: Lightweight Flask agent running inside each AI VM for host communication

**Capabilities**:

- **Health Monitoring**: Process status and resource usage
- **Intelligent Error Analysis**: ML-powered error categorization using embeddings
- **Soft Reboot Management**: Graceful AI process restart without VM shutdown
- **Stats Collection**: Detailed runtime metrics for optimization

**API Endpoints**:

- `/status` - Health check and monitored process status
- `/stats/basic` - Raw system resource usage
- `/stats/intelligent` - AI-analyzed error patterns and faults
- `/soft-reboot` - Graceful AI process restart
- `/config/reload` - Dynamic configuration updates

**Error Intelligence**:

```python
ERROR_CATEGORIES = {
    "CUDA_ERROR": "CUDA out of memory, GPU error, nvidia-smi failed",
    "NETWORK_FAILURE": "Connection timed out, failed to resolve host",
    "DEPENDENCY_MISSING": "ModuleNotFoundError, ImportError, package not found",
    "RUNTIME_ERROR": "TypeError, ValueError, IndexError"
}
```

### 3. AI Action Orchestrator (`ai_action_orchestrator.py`)

**Purpose**: Primary interface for AI agents to interact with their sovereign environment

**Core Functions**:

- **Artifact Management**: Create and execute code in isolated containers
- **Memory Integration**: Store and retrieve execution results
- **Session Logging**: Track AI actions across development sessions
- **API Communication**: Secure communication with host Morpheus application

**Workflow Example**:

```python
# AI creates and executes artifacts
artifact_info = orchestrator.create_and_setup_artifact(
    title="Data Analysis Script",
    content=python_code,
    artifact_type="text/python"
)

# Execute in unlimited container environment
result = orchestrator.execute_code_in_artifact(
    artifact_id, "python3 analyze_data.py"
)

# Store results in persistent memory
orchestrator.remember_result(command, result.output)
```

### 4. Advanced AI Shell (`advanced_ai_shell.py`)

**Purpose**: Unified command execution interface supporting multiple execution contexts

**Execution Contexts**:

- **VM_NATIVE**: Direct execution in AI's persistent VM
- **CONTAINER_OVERLAY**: Specialized containers for heavy computation
- **MULTI_AGENT**: Coordinated execution across multiple AI VMs
- **HYBRID**: VM orchestrating container operations

**Multi-Agent Collaboration**:

```python
# Initiate collaboration with other AI VMs
session_id = await collaboration_manager.initiate_collaboration(
    task_description="Complex research analysis",
    collaborator_agents=[agent_1_id, agent_2_id, agent_3_id]
)

# Collect and synthesize responses
result = await collaboration_manager.collect_collaboration_responses(session_id)
```

**Container Orchestration**:

- GPU-accelerated execution environments
- Isolated artifact execution
- Automatic cleanup and resource management
- Security boundaries through containerization

### 5. AI Personal Development Environment (`ai_personal_dev_environment.py`)

**Purpose**: Self-improving development setup that evolves with AI experience

**Capability Building**:

- **Tool Installation**: Project-specific development environments
- **Personal Libraries**: AI-generated code libraries from learned patterns
- **Automation Scripts**: Custom workflows for repetitive tasks
- **Efficiency Tracking**: Quantified productivity improvements over time

**Development Environment Types**:

```python
environments = {
    "web_development": {
        "tools": ["nodejs", "typescript", "vue-cli"],
        "browsers": "firefox-developer-edition",
        "databases": "postgresql redis-server"
    },
    "ai_research": {
        "tools": ["jupyter", "torch", "transformers"],
        "gpu_tools": "nvidia-smi",
        "notebooks": "jupyter lab"
    },
    "data_analysis": {
        "tools": ["pandas", "plotly", "streamlit"],
        "visualization": "graphviz"
    }
}
```

**Efficiency Evolution**:

- Month 1: 1.0x baseline efficiency
- Month 3: 2.5x with installed tools and libraries
- Month 6: 5.0x with full automation and personal toolkit

### 6. AI Browser Research System (`ai_browser_research_system.py`)

**Purpose**: Persistent browser-based research capabilities with visual interaction

**Research Capabilities**:

- **Visual Web Research**: Browser automation with screenshot capture
- **Form Interaction**: Complex workflow execution
- **Content Extraction**: Intelligent article parsing and summarization
- **Fact Checking**: Cross-referencing information across sources
- **Document Collection**: PDF download and analysis

**Research Workflow Automation**:

```python
research_workflows = {
    'academic_search': 'Multi-database academic paper discovery',
    'fact_checking': 'Cross-source verification with credibility scoring',
    'deep_dive': 'Comprehensive topic investigation with synthesis'
}
```

### 7. Sovereign AI Orchestrator (`ai_orchestrator.py`)

**Purpose**: Central conductor integrating all system components for environment provisioning

**Capability Pack Installation**:

```python
CAPABILITY_PACKS = {
    "base_tools": ["git", "curl", "wget", "build-essential"],
    "web_development": ["nodejs", "npm", "typescript", "vue-cli"],
    "ai_research": ["torch", "transformers", "jupyter", "accelerate"],
    "data_analysis": ["pandas", "scikit-learn", "plotly", "r-base"]
}
```

**Environment Provisioning Workflow**:

1. Security validation and user authorization
2. VM provisioning with hardware specifications
3. DevSession creation and VM linking
4. Capability pack installation with snapshotting
5. Configuration persistence and monitoring setup

## Memory System Integration

### Memory Core (`memory_core.py`)

**Purpose**: Persistent cross-session memory with semantic indexing and encryption

**Memory Architecture**:

- **Vector Storage**: ChromaDB with sentence-transformer embeddings
- **Multi-Modal Support**: Text, code, files, images
- **User-Scoped Encryption**: Fernet encryption with PBKDF2 key derivation
- **Importance-Based Retention**: Automatic memory lifecycle management
- **Cross-Session Continuity**: Context reconstruction across sessions

**Memory Types**:

```python
class MemoryType(str, Enum):
    CORE_FACT = "core_fact"           # Persistent user facts
    CONVERSATION = "conversation"      # Chat exchanges and context
    DOCUMENT = "document"             # Uploaded files and analysis
    CODE_SNIPPET = "code_snippet"    # Generated/executed code
    TOOL_RESULT = "tool_result"       # Plugin/tool outputs
    CUSTOM_INSTRUCTION = "custom_instruction"
    SYSTEM_EVENT = "system_event"     # Technical events and errors
```

### Memory Integration (`memory_integration.py`)

**Purpose**: Seamless integration of persistent memory with session management

**Session Memory Context**:

- Automatic memory storage during conversations
- Context window enhancement with relevant memories
- Cross-session state management
- Privacy-preserving memory access

**Memory-Enhanced Session Management**:

```python
# Enhanced session with memory context
session, memory_context = await enhanced_session_manager.create_session_with_memory(request)

# Automatic fact extraction and storage
await memory_context.store_extracted_fact(
    fact="User prefers Python for data analysis",
    importance=MemoryImportance.HIGH
)

# Context enhancement for new sessions
relevant_memories = await memory_context.enhance_context_with_query(user_query)
```

### System Cache (`system_cache.py`)

**Purpose**: High-performance runtime caching complementary to persistent memory

**Cache Architecture**:

- **Session-Aware Namespacing**: Automatic cleanup and organization
- **LRU Eviction**: Intelligent priority scoring for cache management
- **Background Persistence**: Automatic cache-to-disk serialization
- **Dependency Tracking**: Cache invalidation on data changes
- **Performance Metrics**: Comprehensive hit/miss ratio tracking

**Cache Namespaces**:

```python
class CacheNamespace(str, Enum):
    GLOBAL = "global"
    SESSION = "session"
    VM = "vm"
    ARTIFACT = "artifact"
    MODEL = "model"
    RESEARCH = "research"
    SYSTEM = "system"
```

## VM Settings and Configuration

### VM Settings Manager (`vm_settings.py`)

**Purpose**: Comprehensive configuration management for persistent AI computers

**Instance Configuration**:

```python
class VMInstanceSettings(BaseModel):
    instance_id: UUID
    instance_name: str
    user_id: str
    hardware_spec: VMHardwareSpec
    resource_policy: ResourcePolicy
    personality: VMPersonality
    auto_suspend_minutes: int = 30
    backup_schedule: BackupSchedule = BackupSchedule.DAILY
    installed_tools: List[str]
    learned_capabilities: List[str]
```

**Resource Policies**:

- **CONSERVATIVE**: Minimal resources, slow scale-up
- **BALANCED**: Moderate resources, adaptive scaling  
- **PERFORMANCE**: High resources, fast response
- **UNLIMITED**: No artificial limits
- **CUSTOM**: User-defined allocation

## Security and Isolation

### Network Security

- **Localhost-Only Communication**: All VM-host communication via 127.0.0.1
- **Isolated Networks**: VMs operate on separate virtual networks
- **No External Dependencies**: Complete air-gapped operation possible
- **Secure SSH**: Key-based authentication for VM access

### Container Security

- **Process Isolation**: Each artifact runs in dedicated container
- **Resource Limits**: CPU/memory constraints prevent resource exhaustion
- **Network Restrictions**: Optional network access control
- **Temporary Execution**: Containers destroyed after execution

### Data Security

- **Encrypted Memory**: User-scoped encryption for all stored data
- **Local Storage**: No cloud synchronization or external storage
- **Audit Logging**: Comprehensive execution and access logging
- **Secure Communication**: TLS for all inter-component communication

## Hot Swapping and Dynamic Reconfiguration

### Memory-Driven VM Orchestration

The memory + cache system enables intelligent VM hot swapping between subsystems:

**Context-Aware Resource Allocation**:

```python
# Memory system tracks user patterns
if memory_core.get_user_pattern(user_id) == "research_heavy":
    vm_supervisor.apply_resource_profile(vm_id, "research")
elif cache.get_recent_activity(session_id) == "gpu_intensive":
    vm_supervisor.apply_resource_profile(vm_id, "media_creation")
```

**Seamless Subsystem Migration**:

- **Research Mode**: Browser research system activation
- **Development Mode**: Full IDE and development tool suite
- **Collaboration Mode**: Multi-agent communication protocols
- **Analysis Mode**: Data processing and visualization tools

**State Preservation During Transitions**:

```python
# Create snapshot before subsystem change
vm_supervisor.create_snapshot(vm_id, "Before research mode activation")

# Apply new configuration
await ai_orchestrator.install_capability_pack(vm_id, "ai_research")

# Update memory context
memory_context.store_memory(
    "subsystem_change", 
    f"Switched to research mode for {task_description}"
)
```

## Performance and Optimization

### Resource Monitoring

- Real-time CPU, memory, and GPU usage tracking
- Process-level monitoring for AI applications
- Intelligent error pattern recognition
- Automatic scaling based on workload demands

### Efficiency Metrics

- **Capability Accumulation**: Track tool installation over time
- **Automation Development**: Measure script and library creation
- **Execution Optimization**: Performance improvements through experience
- **Collaboration Effectiveness**: Multi-agent task completion rates

### Caching Strategy

- **Hot Data**: Frequently accessed information in memory cache
- **Session Context**: Recent conversation and execution history
- **Model Results**: Cached LLM outputs for identical queries
- **Artifact Cache**: Reusable code snippets and configurations

## Deployment and Scaling

### Single User Deployment

- Minimum: 16GB RAM, 4 CPU cores, 500GB storage
- Recommended: 32GB RAM, 8 CPU cores, 1TB NVMe SSD
- Optimal: 64GB RAM, 16 CPU cores, 2TB NVMe SSD, GPU

### Multi-User Enterprise Deployment

- **VM Pool Management**: Shared resource allocation
- **User Isolation**: Complete separation of user data and VMs
- **Load Balancing**: Intelligent VM distribution across hardware
- **Backup Strategies**: Automated snapshot and backup scheduling

### Container Registry

- **Local Artifact Registry**: No external dependencies
- **Custom Base Images**: Optimized containers for specific tasks
- **Capability-Specific Images**: Pre-configured environments
- **Automatic Image Building**: Dynamic container creation

## Integration Points

### DevSession Integration

- VMs linked to development sessions for context preservation
- Session event logging for audit trails
- Automatic session state restoration
- Cross-session memory continuity

### Artifact System Integration

- Unlimited execution environments for artifact containers
- Persistent storage for artifact results
- Version control for artifact evolution
- Collaborative artifact development

### Plugin System Integration

- Dynamic plugin installation in persistent VMs
- Plugin state preservation across sessions
- Custom plugin development environments
- Plugin marketplace integration

## Future Enhancements

### Planned Features

- **GPU Clustering**: Multi-GPU coordination for large models
- **VM Migration**: Live migration between physical hosts
- **Federated Learning**: Cross-VM model training coordination
- **Advanced Collaboration**: Real-time screen sharing between AI VMs

### Research Directions

- **Autonomous Capability Discovery**: AI agents finding and installing new tools
- **Self-Healing Systems**: Automatic error recovery and optimization
- **Predictive Resource Allocation**: ML-driven hardware scaling
- **Cross-Agent Knowledge Transfer**: Shared learning between AI instances

## Conclusion

The Somnus VM Architecture represents a fundamental shift from ephemeral AI services to persistent, evolving AI computing environments. By combining never-reset VMs, integrated memory systems, and intelligent orchestration, it enables AI agents to accumulate capabilities, build personal toolkits, and collaborate effectively while maintaining complete user sovereignty and privacy.

This architecture forms the foundation for truly autonomous AI development environments that grow more capable and efficient over time, representing the future of AI-human collaboration in software development and research.

import asyncio
import json
import logging
import psutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

import aiofiles
from pydantic import BaseModel, Field, validator

logger = logging.getLogger(**name**)

# ============================================================================

# VM CONFIGURATION MODELS

# ============================================================================

class VMState(str, Enum):
    """Virtual machine lifecycle states"""
    CREATING = "creating"
    RUNNING = "running"
    SUSPENDED = "suspended"
    SAVING = "saving"
    RESTORING = "restoring"
    BACKING_UP = "backing_up"
    ERROR = "error"
    ARCHIVED = "archived"

class BackupSchedule(str, Enum):
    """VM backup frequency options"""
    DISABLED = "disabled"
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    ON_CAPABILITY_CHANGE = "on_capability_change"
    CUSTOM = "custom"

class ResourcePolicy(str, Enum):
    """VM resource allocation policies"""
    CONSERVATIVE = "conservative"    # Minimal resources, slow scale-up
    BALANCED = "balanced"           # Moderate resources, adaptive scaling
    PERFORMANCE = "performance"     # High resources, fast response
    UNLIMITED = "unlimited"         # No artificial limits
    CUSTOM = "custom"              # User-defined allocation

@dataclass
class VMHardwareSpec:
    """VM hardware specifications"""
    vcpus: int = 4
    memory_gb: int = 8
    storage_gb: int = 100
    gpu_enabled: bool = False
    gpu_memory_gb: Optional[int] = None

    # Network configuration
    network_enabled: bool = True
    ssh_port: int = 2222
    vnc_port: int = 5900
    
    # Performance tuning
    cpu_priority: str = "normal"  # low, normal, high
    memory_ballooning: bool = True
    disk_cache_mode: str = "writethrough"  # none, writethrough, writeback

@dataclass
class VMPersonality:
    """AI personality configuration for VM behavior"""
    agent_name: str = "AI_Assistant"
    specialization: str = "general"  # general, research, coding, analysis
    creativity_level: float = 0.7
    research_methodology: str = "systematic"  # systematic, exploratory, hybrid

    # Workspace preferences
    preferred_ide: str = "vscode"
    preferred_shell: str = "bash"
    preferred_browser: str = "firefox"
    
    # Capability preferences
    auto_install_tools: bool = True
    capability_learning_enabled: bool = True
    cross_session_memory: bool = True

class VMInstanceSettings(BaseModel):
    """Configuration for individual VM instances"""
    # Instance identification
    instance_id: UUID = Field(default_factory=uuid4)
    instance_name: str = Field(..., min_length=1, max_length=100)
    user_id: str = Field(..., description="Owner user ID")

    # Hardware configuration
    hardware_spec: VMHardwareSpec = Field(default_factory=VMHardwareSpec)
    resource_policy: ResourcePolicy = ResourcePolicy.BALANCED
    
    # AI personality and behavior
    personality: VMPersonality = Field(default_factory=VMPersonality)
    
    # Lifecycle management
    auto_suspend_minutes: int = Field(default=30, ge=5, le=1440)
    max_idle_hours: int = Field(default=24, ge=1, le=168)
    backup_schedule: BackupSchedule = BackupSchedule.DAILY
    retention_days: int = Field(default=90, ge=7, le=365)
    
    # Storage and persistence
    vm_disk_path: Optional[str] = None
    snapshot_path: Optional[str] = None
    backup_path: Optional[str] = None
    
    # Capability tracking
    installed_tools: List[str] = Field(default_factory=list)
    learned_capabilities: List[str] = Field(default_factory=list)
    research_bookmarks: List[str] = Field(default_factory=list)
    custom_workflows: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance metrics
    total_uptime_hours: float = 0.0
    efficiency_rating: float = 1.0
    capability_acquisition_rate: float = 0.0
    
    # Creation and modification tracking
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_backup: Optional[datetime] = None

class VMPoolSettings(BaseModel):
    """Global VM pool management configuration"""
    # System identification
    system_id: UUID = Field(default_factory=uuid4)
    version: str = "1.0.0"

    # Pool management
    max_concurrent_vms: int = Field(default=10, ge=1, le=100)
    max_vms_per_user: int = Field(default=5, ge=1, le=20)
    auto_scaling_enabled: bool = True
    vm_warmup_enabled: bool = False  # Keep dormant for efficiency
    
    # Resource management
    global_resource_limits: Dict[str, Any] = Field(default_factory=lambda: {
        "total_cpu_cores": psutil.cpu_count(),
        "total_memory_gb": int(psutil.virtual_memory().total / (1024**3)),
        "total_storage_gb": 1000,  # Default 1TB allocation
        "reserved_cpu_cores": 2,    # Reserve for host system
        "reserved_memory_gb": 4     # Reserve for host system
    })
    
    # VM templates and base images
    base_image_path: str = "/data/vm_templates/base_ai_computer.qcow2"
    template_directory: str = "/data/vm_templates"
    instance_directory: str = "/data/vm_instances"
    backup_directory: str = "/data/vm_backups"
    
    # Default specifications for new VMs
    default_hardware_spec: VMHardwareSpec = Field(default_factory=VMHardwareSpec)
    default_personality: VMPersonality = Field(default_factory=VMPersonality)
    
    # Automation settings
    auto_backup_enabled: bool = True
    auto_cleanup_enabled: bool = True
    capability_sync_enabled: bool = True
    cross_vm_learning: bool = False  # Future feature
    
    # Performance optimization
    vm_balancing_enabled: bool = True
    resource_monitoring_interval: int = 30
    health_check_interval: int = 60
    
    # Security settings
    ssh_key_management: bool = True
    network_isolation: bool = True
    snapshot_encryption: bool = False  # Future feature
    
    # User preferences
    user_vm_preferences: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    vm_sharing_enabled: bool = False  # Future collaboration feature

# ============================================================================

# VM SETTINGS MANAGER

# ============================================================================

class VMSettingsManager:
    """Manages VM system configuration and on-demand lifecycle"""

    def __init__(
        self,
        pool_settings: VMPoolSettings,
        config_path: str = "data/vm_settings.json"
    ):
        self.pool_settings = pool_settings
        self.config_path = Path(config_path)
        
        # Runtime state tracking
        self.active_vms: Dict[UUID, VMInstanceSettings] = {}
        self.vm_metrics: Dict[UUID, Dict[str, Any]] = {}
        self.resource_usage: Dict[str, float] = {}
        
        # Background tasks
        self._monitoring_tasks: Dict[str, asyncio.Task] = {}
        self._resource_monitor_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        
        logger.info("VM Settings Manager initialized")
    
    async def initialize(self) -> None:
        """Initialize the VM settings management system"""
        # Load configuration
        await self._load_settings()
        
        # Initialize directories
        await self._initialize_directories()
        
        # Start background monitoring
        await self._start_monitoring_tasks()
        
        logger.info(f"VM Settings Manager initialized with capacity for {self.pool_settings.max_concurrent_vms} VMs")
    
    async def create_vm_instance(
        self,
        user_id: str,
        instance_name: str,
        hardware_spec: Optional[VMHardwareSpec] = None,
        personality: Optional[VMPersonality] = None
    ) -> VMInstanceSettings:
        """Create new VM instance configuration"""
        
        # Check resource availability
        if not await self._check_resource_availability(hardware_spec):
            raise ValueError("Insufficient resources for VM creation")
        
        # Check user limits
        user_vm_count = sum(1 for vm in self.active_vms.values() if vm.user_id == user_id)
        if user_vm_count >= self.pool_settings.max_vms_per_user:
            raise ValueError(f"User VM limit reached ({self.pool_settings.max_vms_per_user})")
        
        # Create VM configuration
        vm_settings = VMInstanceSettings(
            instance_name=instance_name,
            user_id=user_id,
            hardware_spec=hardware_spec or self.pool_settings.default_hardware_spec,
            personality=personality or self.pool_settings.default_personality
        )
        
        # Set up storage paths
        vm_settings.vm_disk_path = str(
            Path(self.pool_settings.instance_directory) / f"{vm_settings.instance_id}.qcow2"
        )
        vm_settings.snapshot_path = str(
            Path(self.pool_settings.instance_directory) / f"{vm_settings.instance_id}_snapshots"
        )
        vm_settings.backup_path = str(
            Path(self.pool_settings.backup_directory) / f"{vm_settings.instance_id}"
        )
        
        # Register VM
        self.active_vms[vm_settings.instance_id] = vm_settings
        
        # Initialize metrics tracking
        self.vm_metrics[vm_settings.instance_id] = {
            "creation_time": datetime.now(timezone.utc),
            "last_heartbeat": None,
            "resource_usage": {"cpu": 0.0, "memory": 0.0, "disk": 0.0},
            "capability_changes": []
        }
        
        # Save configuration
        await self._save_settings()
        
        logger.info(f"Created VM instance {vm_settings.instance_name} for user {user_id}")
        return vm_settings
    
    async def update_vm_settings(
        self,
        instance_id: UUID,
        updates: Dict[str, Any]
    ) -> bool:
        """Update VM instance settings"""
        
        if instance_id not in self.active_vms:
            return False
        
        vm_settings = self.active_vms[instance_id]
        
        # Apply updates with validation
        for key, value in updates.items():
            if hasattr(vm_settings, key):
                # Special handling for complex objects
                if key == "hardware_spec" and isinstance(value, dict):
                    for spec_key, spec_value in value.items():
                        if hasattr(vm_settings.hardware_spec, spec_key):
                            setattr(vm_settings.hardware_spec, spec_key, spec_value)
                elif key == "personality" and isinstance(value, dict):
                    for pers_key, pers_value in value.items():
                        if hasattr(vm_settings.personality, pers_key):
                            setattr(vm_settings.personality, pers_key, pers_value)
                else:
                    setattr(vm_settings, key, value)
        
        vm_settings.last_modified = datetime.now(timezone.utc)
        
        # Save updated configuration
        await self._save_settings()
        
        logger.info(f"Updated VM settings for instance {instance_id}")
        return True
    
    async def track_capability_acquisition(
        self,
        instance_id: UUID,
        capability: str,
        capability_type: str = "tool"
    ) -> bool:
        """Track when AI acquires new capabilities"""
        
        if instance_id not in self.active_vms:
            return False
        
        vm_settings = self.active_vms[instance_id]
        
        # Add to appropriate tracking list
        if capability_type == "tool" and capability not in vm_settings.installed_tools:
            vm_settings.installed_tools.append(capability)
        elif capability_type == "skill" and capability not in vm_settings.learned_capabilities:
            vm_settings.learned_capabilities.append(capability)
        
        # Update metrics
        if instance_id in self.vm_metrics:
            self.vm_metrics[instance_id]["capability_changes"].append({
                "capability": capability,
                "type": capability_type,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "total_capabilities": len(vm_settings.installed_tools) + len(vm_settings.learned_capabilities)
            })
        
        # Calculate efficiency improvement
        total_capabilities = len(vm_settings.installed_tools) + len(vm_settings.learned_capabilities)
        vm_settings.efficiency_rating = 1.0 + (total_capabilities * 0.1)
        
        # Update acquisition rate
        if vm_settings.total_uptime_hours > 0:
            vm_settings.capability_acquisition_rate = total_capabilities / vm_settings.total_uptime_hours
        
        await self._save_settings()
        
        logger.info(f"VM {instance_id} acquired capability: {capability} ({capability_type})")
        return True
    
    async def get_vm_status(self, instance_id: Optional[UUID] = None) -> Dict[str, Any]:
        """Get status of VM instances"""
        
        if instance_id:
            if instance_id in self.active_vms:
                vm_settings = self.active_vms[instance_id]
                metrics = self.vm_metrics.get(instance_id, {})
                
                return {
                    "instance_id": str(instance_id),
                    "instance_name": vm_settings.instance_name,
                    "user_id": vm_settings.user_id,
                    "resource_policy": vm_settings.resource_policy,
                    "hardware_spec": vm_settings.hardware_spec.__dict__,
                    "personality": vm_settings.personality.__dict__,
                    "capabilities": {
                        "installed_tools": vm_settings.installed_tools,
                        "learned_capabilities": vm_settings.learned_capabilities,
                        "efficiency_rating": vm_settings.efficiency_rating,
                        "acquisition_rate": vm_settings.capability_acquisition_rate
                    },
                    "metrics": metrics,
                    "uptime_hours": vm_settings.total_uptime_hours,
                    "created_at": vm_settings.created_at.isoformat(),
                    "last_modified": vm_settings.last_modified.isoformat()
                }
            else:
                return {"error": f"VM instance {instance_id} not found"}
        
        # Return pool status
        return {
            "pool_status": {
                "active_vms": len(self.active_vms),
                "max_concurrent": self.pool_settings.max_concurrent_vms,
                "total_capacity": self.pool_settings.global_resource_limits,
                "current_usage": self.resource_usage
            },
            "active_instances": [
                {
                    "instance_id": str(vm_id),
                    "instance_name": vm_settings.instance_name,
                    "user_id": vm_settings.user_id,
                    "efficiency_rating": vm_settings.efficiency_rating,
                    "total_capabilities": len(vm_settings.installed_tools) + len(vm_settings.learned_capabilities)
                }
                for vm_id, vm_settings in self.active_vms.items()
            ]
        }
    
    async def configure_vm_pool(self, pool_updates: Dict[str, Any]) -> bool:
        """Update VM pool configuration"""
        
        # Apply updates to pool settings
        for key, value in pool_updates.items():
            if hasattr(self.pool_settings, key):
                if key == "global_resource_limits" and isinstance(value, dict):
                    self.pool_settings.global_resource_limits.update(value)
                elif key == "default_hardware_spec" and isinstance(value, dict):
                    for spec_key, spec_value in value.items():
                        if hasattr(self.pool_settings.default_hardware_spec, spec_key):
                            setattr(self.pool_settings.default_hardware_spec, spec_key, spec_value)
                else:
                    setattr(self.pool_settings, key, value)
        
        # Save updated configuration
        await self._save_settings()
        
        logger.info("VM pool configuration updated")
        return True
    
    async def _check_resource_availability(self, hardware_spec: Optional[VMHardwareSpec]) -> bool:
        """Check if sufficient resources are available for VM creation"""
        
        if not hardware_spec:
            hardware_spec = self.pool_settings.default_hardware_spec
        
        # Calculate current resource usage
        current_cpu = sum(
            vm.hardware_spec.vcpus for vm in self.active_vms.values()
        )
        current_memory = sum(
            vm.hardware_spec.memory_gb for vm in self.active_vms.values()
        )
        current_storage = sum(
            vm.hardware_spec.storage_gb for vm in self.active_vms.values()
        )
        
        # Check against limits
        available_cpu = (
            self.pool_settings.global_resource_limits["total_cpu_cores"] - 
            self.pool_settings.global_resource_limits["reserved_cpu_cores"] - 
            current_cpu
        )
        available_memory = (
            self.pool_settings.global_resource_limits["total_memory_gb"] - 
            self.pool_settings.global_resource_limits["reserved_memory_gb"] - 
            current_memory
        )
        available_storage = (
            self.pool_settings.global_resource_limits["total_storage_gb"] - 
            current_storage
        )
        
        return (
            hardware_spec.vcpus <= available_cpu and
            hardware_spec.memory_gb <= available_memory and
            hardware_spec.storage_gb <= available_storage
        )
    
    async def _initialize_directories(self):
        """Initialize VM storage directories"""
        directories = [
            self.pool_settings.template_directory,
            self.pool_settings.instance_directory,
            self.pool_settings.backup_directory
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    async def _start_monitoring_tasks(self):
        """Start background monitoring tasks"""
        if self.pool_settings.resource_monitoring_interval > 0:
            self._resource_monitor_task = asyncio.create_task(
                self._resource_monitoring_loop()
            )
        
        if self.pool_settings.health_check_interval > 0:
            self._health_check_task = asyncio.create_task(
                self._health_check_loop()
            )
    
    async def _resource_monitoring_loop(self):
        """Background resource monitoring"""
        while True:
            try:
                # Update resource usage metrics
                self.resource_usage = {
                    "cpu_usage_percent": sum(vm.hardware_spec.vcpus for vm in self.active_vms.values()) / 
                                       self.pool_settings.global_resource_limits["total_cpu_cores"] * 100,
                    "memory_usage_percent": sum(vm.hardware_spec.memory_gb for vm in self.active_vms.values()) / 
                                          self.pool_settings.global_resource_limits["total_memory_gb"] * 100,
                    "storage_usage_percent": sum(vm.hardware_spec.storage_gb for vm in self.active_vms.values()) / 
                                           self.pool_settings.global_resource_limits["total_storage_gb"] * 100
                }
                
                await asyncio.sleep(self.pool_settings.resource_monitoring_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in resource monitoring: {e}")
                await asyncio.sleep(30)
    
    async def _health_check_loop(self):
        """Background health monitoring for VMs"""
        while True:
            try:
                for vm_id, vm_settings in self.active_vms.items():
                    # Update uptime tracking
                    if vm_id in self.vm_metrics:
                        creation_time = self.vm_metrics[vm_id]["creation_time"]
                        current_uptime = (datetime.now(timezone.utc) - creation_time).total_seconds() / 3600
                        vm_settings.total_uptime_hours = current_uptime
                
                await asyncio.sleep(self.pool_settings.health_check_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health monitoring: {e}")
                await asyncio.sleep(60)
    
    async def _load_settings(self) -> None:
        """Load settings from configuration file"""
        if self.config_path.exists():
            try:
                async with aiofiles.open(self.config_path, 'r') as f:
                    settings_data = json.loads(await f.read())
                
                # Update pool settings
                if "pool_settings" in settings_data:
                    for key, value in settings_data["pool_settings"].items():
                        if hasattr(self.pool_settings, key):
                            setattr(self.pool_settings, key, value)
                
                # Load active VMs
                if "active_vms" in settings_data:
                    for vm_data in settings_data["active_vms"]:
                        vm_settings = VMInstanceSettings(**vm_data)
                        self.active_vms[vm_settings.instance_id] = vm_settings
                
                logger.info(f"VM settings loaded from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load VM settings: {e}")
    
    async def _save_settings(self) -> None:
        """Save current settings to configuration file"""
        try:
            self.config_path.parent.mkdir(parents=True, exist_ok=True)
            
            settings_data = {
                "pool_settings": self.pool_settings.dict(),
                "active_vms": [vm.dict() for vm in self.active_vms.values()],
                "last_updated": datetime.now(timezone.utc).isoformat()
            }
            
            async with aiofiles.open(self.config_path, 'w') as f:
                await f.write(json.dumps(settings_data, indent=2, default=str))
            
            logger.debug(f"VM settings saved to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save VM settings: {e}")
    
    async def shutdown(self) -> None:
        """Graceful shutdown of VM settings management"""
        logger.info("Shutting down VM settings management...")
        
        # Cancel background tasks
        for task_name, task in self._monitoring_tasks.items():
            logger.debug(f"Cancelling task: {task_name}")
            task.cancel()
        
        if self._resource_monitor_task:
            self._resource_monitor_task.cancel()
        if self._health_check_task:
            self._health_check_task.cancel()
        
        # Save final settings
        await self._save_settings()
        
        logger.info("VM settings management shutdown complete")

# ============================================================================

# FACTORY AND INITIALIZATION

# ============================================================================

async def create_vm_settings_manager(
    max_concurrent_vms: int = 10,
    base_image_path: str = "/data/vm_templates/base_ai_computer.qcow2"
) -> VMSettingsManager:
    """Create and initialize VM settings manager"""

    pool_settings = VMPoolSettings(
        max_concurrent_vms=max_concurrent_vms,
        base_image_path=base_image_path
    )
    
    manager = VMSettingsManager(pool_settings)
    await manager.initialize()
    
    logger.info("VM settings manager created and initialized")
    return manager

if **name** == "**main**":
    # Example usage and testing
    async def main():
        import logging
        logging.basicConfig(level=logging.INFO)

        # Create VM settings manager
        manager = await create_vm_settings_manager()
        
        try:
            # Create test VM instance
            vm_settings = await manager.create_vm_instance(
                user_id="test_user",
                instance_name="Test AI Assistant",
                hardware_spec=VMHardwareSpec(vcpus=2, memory_gb=4, storage_gb=50)
            )
            print(f"Created VM instance: {vm_settings.instance_name}")
            
            # Track capability acquisition
            await manager.track_capability_acquisition(
                vm_settings.instance_id,
                "python_development",
                "skill"
            )
            
            # Get status
            status = await manager.get_vm_status(vm_settings.instance_id)
            print(f"VM Status: {status}")
            
        finally:
            # Cleanup
            await manager.shutdown()
    
    asyncio.run(main())
