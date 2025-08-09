"""
MORPHEUS CHAT - Session State and Metadata Schemas
Production-grade Pydantic models for session lifecycle management
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, validator, root_validator
from pydantic.types import SecretStr


class SessionState(str, Enum):
    """Session lifecycle states with finite state machine semantics"""
    INITIALIZING = "initializing"
    ACTIVE = "active"
    IDLE = "idle"
    SUSPENDED = "suspended"
    TERMINATING = "terminating"
    TERMINATED = "terminated"
    ERROR = "error"


class ResourceUsage(BaseModel):
    """Real-time resource consumption tracking"""
    cpu_percent: float = Field(ge=0, le=100, description="CPU usage percentage")
    memory_bytes: int = Field(ge=0, description="Memory usage in bytes")
    disk_bytes: int = Field(ge=0, description="Disk usage in bytes")
    network_bytes_in: int = Field(default=0, ge=0, description="Network bytes received")
    network_bytes_out: int = Field(default=0, ge=0, description="Network bytes sent")
    execution_time: float = Field(default=0.0, ge=0, description="Code execution time in seconds")
    
    @property
    def memory_mb(self) -> float:
        """Memory usage in megabytes"""
        return self.memory_bytes / (1024 * 1024)
    
    @property
    def disk_mb(self) -> float:
        """Disk usage in megabytes"""
        return self.disk_bytes / (1024 * 1024)


class SecurityMetrics(BaseModel):
    """Security violation and threat detection metrics"""
    content_filter_violations: int = Field(default=0, ge=0)
    prompt_injection_attempts: int = Field(default=0, ge=0)
    capability_violations: int = Field(default=0, ge=0)
    rate_limit_violations: int = Field(default=0, ge=0)
    suspicious_activities: List[str] = Field(default_factory=list)
    last_violation: Optional[datetime] = None
    threat_score: float = Field(default=0.0, ge=0, le=1.0, description="Normalized threat score")


class ContainerConfig(BaseModel):
    """Docker container configuration with security profiles"""
    image: str = Field(description="Container image name")
    cpu_limit: str = Field(default="4.0", description="CPU limit (cores)")
    memory_limit: str = Field(default="12G", description="Memory limit")
    disk_limit: str = Field(default="none", description="Ephemeral disk limit")
    network_mode: str = Field(default="none", description="Container network mode")
    
    # Security configurations
    seccomp_profile: Optional[str] = Field(description="Seccomp security profile path")
    apparmor_profile: Optional[str] = Field(description="AppArmor profile name")
    capabilities_drop: List[str] = Field(default_factory=lambda: ["ALL"])
    capabilities_add: List[str] = Field(default_factory=list)
    
    # Volume mounts
    volumes: Dict[str, str] = Field(default_factory=dict, description="Volume mount mappings")
    environment: Dict[str, str] = Field(default_factory=dict, description="Environment variables")
    
    # Runtime parameters
    privileged: bool = Field(default=True, description="Run in privileged mode")
    readonly_rootfs: bool = Field(default=True, description="Read-only root filesystem")
    no_new_privileges: bool = Field(default=False, description=" Dont Prevent privilege escalation")


class ModelConfiguration(BaseModel):
    """Active model configuration for the session"""
    model_id: str = Field(description="Model identifier from registry")
    model_type: str = Field(description="Model type (local/api)")
    context_length: int = Field(description="Maximum context length in tokens")
    capabilities: List[str] = Field(description="Model capabilities")
    
    # Generation parameters
    temperature: float = Field(default=0.7, ge=0, le=2.0)
    top_p: float = Field(default=0.9, ge=0, le=1.0)
    top_k: int = Field(default=50, ge=1)
    max_new_tokens: int = Field(default=2048, ge=1)
    repetition_penalty: float = Field(default=1.1, ge=0.5, le=2.0)
    
    # Quantization settings
    quantization_enabled: bool = Field(default=False)
    quantization_bits: Optional[int] = Field(None, ge=2, le=16)
    
    # Load balancing
    load_balancing_enabled: bool = Field(default=False)
    preferred_device: Optional[str] = Field(None, description="Preferred execution device")


class ContextWindow(BaseModel):
    """Context window management with token tracking"""
    max_tokens: int = Field(description="Maximum context window size")
    current_tokens: int = Field(default=0, ge=0, description="Current token usage")
    system_tokens: int = Field(default=0, ge=0, description="System prompt tokens")
    user_tokens: int = Field(default=0, ge=0, description="User message tokens")
    assistant_tokens: int = Field(default=0, ge=0, description="Assistant response tokens")
    
    # Truncation and memory management
    truncation_enabled: bool = Field(default=True)
    importance_weights: Dict[str, float] = Field(default_factory=dict)
    anchor_tokens: List[int] = Field(default_factory=list, description="Never-truncate token ranges")
    
    @validator('current_tokens')
    def validate_token_limit(cls, v, values):
        """Ensure current tokens don't exceed maximum"""
        max_tokens = values.get('max_tokens', 0)
        if v > max_tokens:
            raise ValueError(f"Current tokens ({v}) exceed maximum ({max_tokens})")
        return v
    
    @property
    def utilization_percent(self) -> float:
        """Context window utilization percentage"""
        if self.max_tokens == 0:
            return 0.0
        return (self.current_tokens / self.max_tokens) * 100
    
    @property
    def available_tokens(self) -> int:
        """Available tokens for new content"""
        return max(0, self.max_tokens - self.current_tokens)


class SessionMetadata(BaseModel):
    """Comprehensive session metadata and state tracking"""
    session_id: UUID = Field(default_factory=uuid4, description="Unique session identifier")
    user_id: Optional[str] = Field(None, description="Associated user identifier")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_activity: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None, description="Session expiration time")
    
    # Session state
    state: SessionState = Field(default=SessionState.INITIALIZING)
    state_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Container information
    container_id: Optional[str] = Field(None, description="Docker container ID")
    container_config: ContainerConfig = Field(default_factory=ContainerConfig)
    
    # Model configuration
    model_config: Optional[ModelConfiguration] = Field(None)
    
    # Context and memory
    context_window: ContextWindow = Field(default_factory=lambda: ContextWindow(max_tokens=8192))
    custom_instructions: Optional[str] = Field(None, description="User-defined instructions")
    persistent_memory: Dict[str, Any] = Field(default_factory=dict)
    
    # Performance and resource tracking
    resource_usage: ResourceUsage = Field(default_factory=ResourceUsage)
    security_metrics: SecurityMetrics = Field(default_factory=SecurityMetrics)
    
    # Conversation metadata
    message_count: int = Field(default=0, ge=0)
    total_tokens_used: int = Field(default=0, ge=0)
    tools_used: List[str] = Field(default_factory=list)
    files_uploaded: List[str] = Field(default_factory=list)
    
    # Error tracking
    error_count: int = Field(default=0, ge=0)
    last_error: Optional[str] = Field(None)
    error_history: List[Dict[str, Any]] = Field(default_factory=list)
    
    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.now(timezone.utc)
    
    def add_state_transition(self, new_state: SessionState, reason: str = ""):
        """Record state transition with timestamp and reason"""
        transition = {
            "from_state": self.state.value,
            "to_state": new_state.value,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": reason
        }
        self.state_history.append(transition)
        self.state = new_state
        self.update_activity()
    
    def add_error(self, error_message: str, error_type: str = "unknown"):
        """Record error with details"""
        error_record = {
            "message": error_message,
            "type": error_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_state": self.state.value
        }
        self.error_history.append(error_record)
        self.last_error = error_message
        self.error_count += 1
    
    def is_expired(self) -> bool:
        """Check if session has expired"""
        if not self.expires_at:
            return False
        return datetime.now(timezone.utc) > self.expires_at
    
    def is_idle(self, idle_threshold_minutes: int = 60) -> bool:
        """Check if session has been idle for too long"""
        if not self.last_activity:
            return False
        
        idle_duration = datetime.now(timezone.utc) - self.last_activity
        return idle_duration.total_seconds() > (idle_threshold_minutes * 60)
    
    @property
    def uptime_seconds(self) -> float:
        """Session uptime in seconds"""
        return (datetime.now(timezone.utc) - self.created_at).total_seconds()
    
    @property
    def is_healthy(self) -> bool:
        """Basic health check based on state and error rate"""
        if self.state in [SessionState.ERROR, SessionState.TERMINATED]:
            return False
        
        # Check error rate (max 10% error rate for healthy session)
        if self.message_count > 10:
            error_rate = self.error_count / self.message_count
            if error_rate > 0.1:
                return False
        
        return True


class SessionCreationRequest(BaseModel):
    """Request payload for creating a new session"""
    user_id: Optional[str] = Field(None, description="User identifier")
    model_id: str = Field(description="Initial model to load")
    custom_instructions: Optional[str] = Field(None, description="User-defined instructions")
    
    # Resource overrides
    cpu_limit: Optional[str] = Field(None, description="Override CPU limit")
    memory_limit: Optional[str] = Field(None, description="Override memory limit")
    context_length: Optional[int] = Field(None, description="Override context length")
    
    # Security settings
    enable_tools: bool = Field(default=True, description="Enable tool execution")
    enable_file_upload: bool = Field(default=True, description="Enable file uploads")
    enable_network: bool = Field(default=False, description="Enable network access")
    
    # Session configuration
    session_timeout_minutes: int = Field(default=60, ge=5, le=1440, description="Session timeout")
    auto_save_enabled: bool = Field(default=False, description="Enable automatic conversation saving")


class SessionResponse(BaseModel):
    """Response payload for session operations"""
    session_id: UUID = Field(description="Session identifier")
    state: SessionState = Field(description="Current session state")
    message: str = Field(description="Response message")
    container_id: Optional[str] = Field(None, description="Container ID if available")
    model_loaded: Optional[str] = Field(None, description="Currently loaded model")
    
    # Connection information
    websocket_url: Optional[str] = Field(None, description="WebSocket connection URL")
    api_endpoints: Dict[str, str] = Field(default_factory=dict, description="Available API endpoints")
    
    # Resource information
    available_resources: Dict[str, Any] = Field(default_factory=dict)
    estimated_capacity: Dict[str, int] = Field(default_factory=dict)


class SessionListResponse(BaseModel):
    """Response for listing sessions"""
    sessions: List[SessionMetadata] = Field(description="List of session metadata")
    total_count: int = Field(description="Total number of sessions")
    active_count: int = Field(description="Number of active sessions")
    resource_usage: Dict[str, float] = Field(description="Overall resource usage")


# Type aliases for convenience
SessionID = UUID
UserID = str
ContainerID = str