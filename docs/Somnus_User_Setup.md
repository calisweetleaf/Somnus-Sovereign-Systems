# Somnus Sovereign Systems - User Registry Documentation v2.0

## Overview

The Somnus User Registry is a privacy-first, local-only user profile system with intelligent capability projection for AI collaboration. It transforms personal information into abstract capabilities without exposing sensitive details, maintaining complete user sovereignty while enabling sophisticated AI personalization.

## Core Philosophy

### Data Sovereignty Principles

- **Local-Only Operation**: No data ever leaves the user's machine
- **Capability-Based Projection**: AI sees capabilities, not raw personal data
- **Granular Context Control**: Different projections for different AI tasks/roles
- **Explicit Consent**: All AI access requires explicit user permission
- **Provenance Tracking**: Complete audit trail of data sources and access

### Revolutionary Projection Layer

Unlike traditional systems that expose raw user data, Somnus uses a **Projection DSL (Domain Specific Language)** that maps personal information to abstract capabilities:

```
Raw Data: "RTX 4090, 32GB RAM, VS Code, Docker"
↓ Projection DSL ↓
AI Sees: {
  "capabilities": {
    "gpu_acceleration": true,
    "high_memory_available": true,
    "containerization": true,
    "code_editing": true
  }
}
```

## System Architecture

### Profile Component Structure

#### 1. Identity Profile
Core user identification with projection mapping:
- **Display name** → `who.name`
- **Professional role** → `who.roles[]`
- **Expertise areas** → `capabilities.domain_knowledge[]`
- **Timezone** → `context.timezone`

#### 2. Technical Profile  
Hardware/software environment with capability mapping:
- **GPU specs** → `capabilities.gpu_acceleration`, `capabilities.ml_inference`
- **Development tools** → `capabilities.containerization`, `capabilities.version_control`
- **Network context** → `capabilities.cloud_access`, `capabilities.offline_capable`

#### 3. Workflow Profile
Work patterns with behavioral indicators:
- **Communication style** → `prefs.communication_style`
- **Documentation preferences** → `prefs.doc_format`
- **Peak hours** → `context.availability_patterns`

#### 4. Learning Profile
Knowledge acquisition preferences:
- **Learning styles** → `prefs.explanation_style`
- **Error tolerance** → `prefs.feedback_approach`
- **Code commenting** → `prefs.code_verbosity`

### Projection DSL System

#### Field Metadata with Projection Mapping
```python
class FieldMeta:
    expose_to_llm: bool = False
    scope: str = "session"  # session|always|agent:<id>|task:<tag>
    redaction: str = "coarse"  # none|coarse|masked|hidden
    project_as: Optional[str] = None  # DSL target: "capabilities.gpu_acceleration"
    expires_at: Optional[datetime] = None
    provenance: str = "user_input"  # user_input|import|ai_observation
```

#### Capability Mapping Examples
```python
# Hardware projections
"gpu_model": FieldMeta(
    expose_to_llm=True,
    project_as="capabilities.gpu_acceleration",
    redaction="hidden"  # Raw model hidden, only capability exposed
)

# Software projections  
"development_tools": FieldMeta(
    expose_to_llm=True,
    project_as="capabilities.containerization,capabilities.version_control",
    redaction="coarse"
)

# Context projections
"timezone": FieldMeta(
    expose_to_llm=True,
    project_as="context.timezone",
    redaction="none"
)
```

### Role and Task-Aware Projections

Different AI agents/tasks see different capability sets:

#### Research Agent Context
```python
# Scope: "task:research"
{
  "capabilities": {
    "web_scraping": true,
    "data_analysis": true,
    "gpu_acceleration": true
  },
  "prefs": {
    "depth": "comprehensive",
    "sources": "academic_preferred"
  }
}
```

#### Coding Agent Context
```python
# Scope: "task:coding"
{
  "capabilities": {
    "containerization": true,
    "version_control": true,
    "gpu_development": true
  },
  "prefs": {
    "code_style": "pythonic",
    "commenting": "balanced"
  }
}
```

### Dual Storage Architecture

#### Default: Python Module Store
```python
# profiles/alex_developer.py
from somnus_user_profile.types import UserProfile, Identity, Technical

profile = UserProfile(
    username="alex_developer",
    identity=Identity(
        display_name="Alex",
        role="Senior Developer",
        expertise=["Python", "AI", "DevOps"]
    ),
    technical=Technical(
        primary_machine={"gpu": "RTX_4090", "ram_gb": 32},
        tools=["VS Code", "Docker", "Git"]
    )
)
```

#### Optional: Encrypted SQLite Store
```python
# For users wanting search/indexing capabilities
SOMNUS_PROFILE_BACKEND=sqlite
```

### Provenance and Consent System

#### Data Source Tracking
```python
provenance_types = {
    "user_input": "User directly entered",
    "import": "Imported from external system", 
    "ai_observation": "AI detected/suggested",
    "system_detected": "Auto-detected by system"
}
```

#### Promotion Workflow
```python
# AI makes observation
ai_suggestion = {
    "field": "technical.tools",
    "value": "Kubernetes",
    "confidence": 0.85,
    "evidence": "User mentioned kubectl commands"
}

# User must explicitly promote
user_action_required = "Promote AI observation to profile?"
# Only on user approval does it become permanent
```

## Security Model

### Projection-First Security
- **No Raw Data Exposure**: AI never sees actual hardware specs, file paths, or personal details
- **Capability Abstraction**: All data transformed into abstract capabilities
- **Context Isolation**: Different tasks see different capability subsets
- **Audit Trail**: Complete log of what was projected to which agent when

### Encryption and Authentication
- **Key Derivation**: Argon2id from username + password + salt
- **Profile Encryption**: All stored data encrypted at rest
- **Session Security**: Temporary decryption with automatic cleanup
- **Memory Protection**: Sensitive data cleared on logout

### Audit and Replay System
```python
audit_entry = {
    "timestamp": "2025-08-09T14:30:00Z",
    "action": "projection_generated",
    "agent_id": "coding_assistant_v1",
    "scope": "task:coding",
    "projected_fields": ["capabilities.containerization", "prefs.code_style"],
    "session_id": "sess_abc123"
}
```

## Integration with Somnus Systems

### VM Settings Integration
```python
def map_projection_to_vm(vm_settings, projection):
    """Map user capabilities to VM configuration"""
    caps = projection.get("capabilities", {})
    prefs = projection.get("prefs", {})
    
    if caps.get("gpu_acceleration"):
        vm_settings.hardware_spec.gpu_enabled = True
    
    if caps.get("containerization"):
        vm_settings.installed_tools.append("docker")
    
    if prefs.get("communication_style") == "technical":
        vm_settings.personality.technical_depth = "advanced"
```

### Memory System Integration
```python
memory_config = {
    "retention_days": projection.get("prefs", {}).get("memory_retention", 90),
    "summarization": projection.get("prefs", {}).get("auto_summarize", True),
    "privacy_level": projection.get("context", {}).get("privacy_level", "balanced")
}
```

## User Experience Flow

### Initial Setup
1. **Profile Creation**: Username, password, basic identity
2. **Privacy-First Defaults**: All fields `expose_to_llm=False` initially
3. **Capability Mapping**: User chooses which capabilities to expose
4. **Projection Preview**: "Here's exactly what the AI will see"
5. **Task Configuration**: Different projections for different AI tasks

### Projection Management
```python
# User toggles specific capabilities
user.toggle_capability("gpu_acceleration", enabled=True, scope="task:coding")
user.toggle_capability("location_context", enabled=False, scope="always")

# Live preview
preview = user.get_projection_preview(scope="task:research")
```

### AI Interaction Flow
1. **Task Identification**: System identifies AI task (coding, research, etc.)
2. **Scope Selection**: Appropriate projection scope selected
3. **Capability Projection**: Generate filtered context for AI
4. **Audit Logging**: Record what was projected when
5. **Session Cleanup**: Clear sensitive data after interaction

## Canonical Capability Flags

### Hardware Capabilities
- `gpu_acceleration`: GPU available for compute
- `high_memory_available`: >16GB RAM available
- `ssd_storage`: Fast storage available
- `multiple_monitors`: Multi-display setup
- `high_bandwidth`: Fast internet connection

### Software Capabilities
- `containerization`: Docker/container tools
- `version_control`: Git and related tools
- `cloud_platforms`: AWS/GCP/Azure access
- `ml_frameworks`: TensorFlow/PyTorch available
- `code_editing`: Advanced IDE/editor setup

### Development Capabilities
- `web_development`: HTML/CSS/JS tools
- `mobile_development`: iOS/Android tools
- `data_science`: Pandas/NumPy/Jupyter setup
- `devops`: CI/CD and deployment tools
- `security_tools`: Pentesting/security software

### Domain Knowledge
- `ai_expertise`: Machine learning knowledge
- `web3_knowledge`: Blockchain/crypto understanding
- `cybersecurity`: Information security expertise
- `data_analysis`: Statistics and analytics knowledge
- `system_administration`: Server/infrastructure knowledge

## Implementation Guidelines

### Minimal API Contract
```python
# Core types
class UserProfile(BaseModel):
    username: str
    identity: Identity = Field(default_factory=Identity)
    technical: Technical = Field(default_factory=Technical)
    workflow: Workflow = Field(default_factory=Workflow)
    learning: Learning = Field(default_factory=Learning)

# Projection system
def project(profile: UserProfile, scope: str = "session") -> Dict[str, Any]:
    """Generate capability projection for AI consumption"""

# Storage backends
class ProfileStore(Protocol):
    def load(self, username: str) -> Optional[UserProfile]: ...
    def save(self, profile: UserProfile) -> None: ...
```

### Backend Selection
```python
# Environment variable controls backend
SOMNUS_PROFILE_BACKEND=python  # Default: Python modules
SOMNUS_PROFILE_BACKEND=sqlite  # Optional: Encrypted SQLite
```

### Projection DSL Examples
```python
# Simple capability mapping
"gpu_model" → "capabilities.gpu_acceleration"

# Multiple capability mapping  
"development_tools" → "capabilities.containerization,capabilities.version_control"

# Conditional mapping
"ram_gb" → "capabilities.high_memory_available" (if > 16GB)

# Context mapping
"timezone" → "context.timezone"
"work_hours" → "context.availability_patterns"
```

## Security Considerations

### Threat Model Protection
- **Data Extraction**: Capabilities don't leak personal details
- **Inference Attacks**: Abstract capabilities prevent reconstruction
- **Memory Analysis**: Encrypted storage with secure cleanup
- **Side Channel**: Audit logs detect unusual access patterns

### Privacy Guarantees
- **Capability Abstraction**: No reverse engineering from capabilities to raw data
- **Scope Isolation**: Task-specific projections limit exposure
- **Explicit Consent**: All AI access requires user approval
- **Audit Transparency**: Users see exactly what AI accessed when

## Future Enhancements

### Advanced Projection Features
- **Dynamic Capability Detection**: Auto-detect new software installations
- **Capability Versioning**: Track capability changes over time
- **Cross-Profile Learning**: Learn optimal projection patterns (with consent)
- **Federated Capabilities**: Share abstract capabilities across Somnus installations

### AI Integration Expansion
- **Capability Optimization**: AI suggests optimal capability configurations
- **Context Prediction**: Anticipate needed capabilities for tasks
- **Privacy Coaching**: AI helps users optimize privacy settings
- **Workflow Intelligence**: Capability-driven workflow optimization

---

## Conclusion

The Somnus User Registry v2.0 represents a breakthrough in privacy-preserving AI personalization. By abstracting personal data into capabilities and providing granular projection controls, it enables unprecedented AI collaboration while maintaining absolute user sovereignty.

The Projection DSL system ensures that AI agents receive meaningful context without compromising user privacy. Combined with provenance tracking, audit logging, and role-based projections, it creates a new standard for sovereign AI interactions.

This system serves as the foundation for all AI personalization within the Somnus ecosystem, ensuring users remain in complete control while enabling sophisticated, context-aware AI assistance.