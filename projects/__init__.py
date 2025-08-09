"""
MORPHEUS CHAT - Projects System Package
Complete project management system for AI sovereignty

The ultimate SaaS-killer project system that provides:
- Persistent VMs for each project (never reset)
- Autonomous file organization and knowledge management
- Multi-agent collaboration capabilities
- Intelligent automation and task scheduling
- Revolutionary artifact system with unlimited execution
- Cross-session memory integration
- Complete local sovereignty with no limits

This package destroys paid SaaS AI services by providing:
✅ Unlimited projects vs. limited plans
✅ Persistent VMs vs. ephemeral containers  
✅ Autonomous intelligence vs. manual organization
✅ Multi-agent teams vs. single AI
✅ Unlimited execution vs. timeout limits
✅ Complete sovereignty vs. vendor lock-in
✅ Zero monthly fees vs. $20-200/month subscriptions
"""

__version__ = "1.0.0"
__author__ = "Morpheus Chat Development Team"
__description__ = "Revolutionary project management system for complete AI sovereignty"

# Core project management components
from .project_core import (
    ProjectManager,
    ProjectMetadata,
    ProjectSpecs,
    ProjectType,
    ProjectStatus
)

# Virtual machine management for projects
from .project_vm_manager import (
    ProjectVMManager,
    ProjectVMInstance,
    ProjectVMState,
    ProjectPortManager
)

# Autonomous intelligence and file management
from .project_intelligence import (
    ProjectIntelligenceEngine,
    FileAnalysis,
    ProjectInsight
)

# Dynamic knowledge base management
from .project_knowledge import (
    ProjectKnowledgeBase,
    KnowledgeItem,
    KnowledgeGraph
)

# Multi-agent collaboration system
from .project_collaboration import (
    ProjectCollaborationManager