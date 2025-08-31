"""
SOMNUS SYSTEMS - Capability Broker
Cross-Subsystem Capability Negotiation and Resource Allocation

ARCHITECTURE PHILOSOPHY:
- Capabilities are abstract resources that map to concrete subsystem features
- User projections control which capabilities are available to AI agents
- Intelligent routing based on capability type and current system state
- Resource allocation with conflict resolution and priority management
- Complete audit trail for capability usage and denial reasons

This broker ensures AI agents can request abstract capabilities (like "gpu_acceleration")
while the system intelligently routes these to the appropriate subsystem and validates
permissions through the user's projection settings.
"""

import asyncio
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from abc import ABC, abstractmethod
import contextlib
import random

from events import (
    get_event_bus, publish_capability_requested, CapabilityEvent,
    EventPriority, EventMetadata
)
# Fix: do not import get_event_bus from projection_registry; import only what is needed.
from projection_registry import CapabilityFlag, generate_projection_from_registry
# Integrate with user registry store to load user profiles for projections
from user_registry import SecurePythonModuleStore

logger = logging.getLogger(__name__)


# ============================================================================
# CAPABILITY DEFINITIONS AND TYPES
# ============================================================================

class CapabilityType(str, Enum):
    """Types of capabilities that can be requested"""
    HARDWARE = "hardware"       # Physical hardware resources
    SOFTWARE = "software"       # Software tools and frameworks  
    NETWORK = "network"         # Network and connectivity
    COMPUTE = "compute"         # Computing resources
    STORAGE = "storage"         # Storage and data access
    DOMAIN = "domain"           # Domain-specific knowledge/tools

class CapabilityStatus(str, Enum):
    """Status of capability requests"""
    PENDING = "pending"         # Request being processed
    GRANTED = "granted"         # Capability granted and available
    DENIED = "denied"           # Request denied
    EXPIRED = "expired"         # Grant expired
    REVOKED = "revoked"         # Grant revoked
    ERROR = "error"             # Error in processing

class ResourcePriority(str, Enum):
    """Priority levels for resource allocation"""
    CRITICAL = "critical"       # System-critical operations
    HIGH = "high"              # User-initiated operations
    NORMAL = "normal"          # Standard background operations
    LOW = "low"                # Non-essential operations

@dataclass
class CapabilityDefinition:
    """Definition of a capability and its requirements"""
    name: str
    capability_type: CapabilityType
    description: str
    required_subsystems: List[str] = field(default_factory=list)
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    conflicts_with: Set[str] = field(default_factory=set)
    prerequisites: Set[str] = field(default_factory=set)
    max_concurrent_grants: Optional[int] = None
    default_grant_duration_minutes: int = 60
    auto_renewal_enabled: bool = True

@dataclass
class CapabilityRequest:
    """A request for a specific capability"""
    request_id: str = field(default_factory=lambda: str(uuid4()))
    capability_name: str = ""
    requester_id: str = ""
    requester_type: str = ""  # "agent", "user", "subsystem"
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    scope: str = "session"
    priority: ResourcePriority = ResourcePriority.NORMAL
    duration_minutes: Optional[int] = None
    context: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
@dataclass
class CapabilityGrant:
    """A granted capability with tracking information"""
    grant_id: str = field(default_factory=lambda: str(uuid4()))
    request: CapabilityRequest = field(default_factory=CapabilityRequest)
    status: CapabilityStatus = CapabilityStatus.PENDING
    granted_at: Optional[datetime] = None
    expires_at: Optional[datetime] = None
    subsystem_allocations: Dict[str, Any] = field(default_factory=dict)
    resource_handles: Dict[str, Any] = field(default_factory=dict)
    usage_stats: Dict[str, Any] = field(default_factory=dict)
    denial_reason: Optional[str] = None
    last_accessed: Optional[datetime] = None
    access_count: int = 0


# ============================================================================
# CAPABILITY REGISTRY
# ============================================================================

CAPABILITY_REGISTRY: Dict[str, CapabilityDefinition] = {
    
    # Hardware Capabilities
    CapabilityFlag.GPU_ACCELERATION: CapabilityDefinition(
        name=CapabilityFlag.GPU_ACCELERATION,
        capability_type=CapabilityType.HARDWARE,
        description="GPU acceleration for compute tasks",
        required_subsystems=["vm_manager"],
        resource_requirements={"gpu_memory_mb": 1024, "gpu_compute_capability": "6.0+"},
        max_concurrent_grants=5,
        default_grant_duration_minutes=120
    ),
    
    CapabilityFlag.HIGH_MEMORY_AVAILABLE: CapabilityDefinition(
        name=CapabilityFlag.HIGH_MEMORY_AVAILABLE,
        capability_type=CapabilityType.HARDWARE,
        description="High memory allocation (16GB+)",
        required_subsystems=["vm_manager"],
        resource_requirements={"memory_gb": 16},
        max_concurrent_grants=3
    ),
    
    CapabilityFlag.SSD_STORAGE: CapabilityDefinition(
        name=CapabilityFlag.SSD_STORAGE,
        capability_type=CapabilityType.STORAGE,
        description="Fast SSD storage access",
        required_subsystems=["vm_manager"],
        resource_requirements={"storage_iops": 10000}
    ),
    
    # Software Capabilities
    CapabilityFlag.CONTAINERIZATION: CapabilityDefinition(
        name=CapabilityFlag.CONTAINERIZATION,
        capability_type=CapabilityType.SOFTWARE,
        description="Container orchestration capabilities",
        required_subsystems=["artifact_system"],
        resource_requirements={"docker_api": True},
        conflicts_with={"vm_isolation"}
    ),
    
    CapabilityFlag.VERSION_CONTROL: CapabilityDefinition(
        name=CapabilityFlag.VERSION_CONTROL,
        capability_type=CapabilityType.SOFTWARE,
        description="Version control system access",
        required_subsystems=["artifact_system", "vm_manager"],
        resource_requirements={"git_client": True}
    ),
    
    CapabilityFlag.ML_FRAMEWORKS: CapabilityDefinition(
        name=CapabilityFlag.ML_FRAMEWORKS,
        capability_type=CapabilityType.SOFTWARE,
        description="Machine learning framework access",
        required_subsystems=["artifact_system", "vm_manager"],
        resource_requirements={"python_env": True, "ml_libraries": ["tensorflow", "pytorch"]},
        prerequisites={CapabilityFlag.GPU_ACCELERATION}
    ),
    
    CapabilityFlag.CLOUD_PLATFORMS: CapabilityDefinition(
        name=CapabilityFlag.CLOUD_PLATFORMS,
        capability_type=CapabilityType.NETWORK,
        description="Cloud platform API access",
        required_subsystems=["artifact_system"],
        resource_requirements={"internet_access": True, "api_credentials": True}
    ),
    
    # Development Capabilities
    CapabilityFlag.WEB_DEVELOPMENT: CapabilityDefinition(
        name=CapabilityFlag.WEB_DEVELOPMENT,
        capability_type=CapabilityType.SOFTWARE,
        description="Web development tools and frameworks",
        required_subsystems=["artifact_system"],
        resource_requirements={"node_runtime": True, "web_server": True}
    ),
    
    CapabilityFlag.DATA_SCIENCE: CapabilityDefinition(
        name=CapabilityFlag.DATA_SCIENCE,
        capability_type=CapabilityType.SOFTWARE,
        description="Data science and analytics tools",
        required_subsystems=["artifact_system", "memory_system"],
        resource_requirements={"jupyter_env": True, "data_libraries": ["pandas", "numpy"]},
        prerequisites={CapabilityFlag.HIGH_MEMORY_AVAILABLE}
    ),
    
    CapabilityFlag.DEVOPS: CapabilityDefinition(
        name=CapabilityFlag.DEVOPS,
        capability_type=CapabilityType.SOFTWARE,
        description="DevOps and infrastructure tools",
        required_subsystems=["artifact_system"],
        resource_requirements={"infrastructure_apis": True},
        prerequisites={CapabilityFlag.CONTAINERIZATION, CapabilityFlag.CLOUD_PLATFORMS}
    ),
    
    # Domain Knowledge Capabilities
    CapabilityFlag.AI_EXPERTISE: CapabilityDefinition(
        name=CapabilityFlag.AI_EXPERTISE,
        capability_type=CapabilityType.DOMAIN,
        description="AI and machine learning domain expertise",
        required_subsystems=["research_system"],
        resource_requirements={"research_databases": ["arxiv", "papers_with_code"]}
    ),
    
    CapabilityFlag.CYBERSECURITY: CapabilityDefinition(
        name=CapabilityFlag.CYBERSECURITY,
        capability_type=CapabilityType.DOMAIN,
        description="Cybersecurity domain expertise and tools",
        required_subsystems=["research_system", "artifact_system"],
        resource_requirements={"security_databases": ["cve", "mitre"], "security_tools": True}
    ),
    
    CapabilityFlag.DATA_ANALYSIS: CapabilityDefinition(
        name=CapabilityFlag.DATA_ANALYSIS,
        capability_type=CapabilityType.DOMAIN,
        description="Data analysis and statistics expertise",
        required_subsystems=["research_system", "artifact_system"],
        prerequisites={CapabilityFlag.DATA_SCIENCE}
    )
}


# ============================================================================
# SUBSYSTEM INTERFACE DEFINITIONS
# ============================================================================

class SubsystemError(Exception):
    """Standardized subsystem operation error."""

class SubsystemInterface(ABC):
    """Base interface for subsystem capability allocation with production-grade scaffolding.
    
    This class provides:
    - Concurrency control (semaphore-based)
    - Timeouts and retries with exponential backoff + jitter
    - Structured logging and lightweight metrics
    - Stable public API methods that call protected abstract hooks to be implemented by subsystems
    """

    def __init__(
        self,
        name: str,
        *,
        max_concurrency: int = 10,
        operation_timeout: float = 30.0,
        retry_attempts: int = 2,
        retry_backoff_base: float = 0.2,
    ) -> None:
        self.name = name
        self.max_concurrency = max(1, int(max_concurrency))
        self.operation_timeout = float(operation_timeout)
        self.retry_attempts = max(0, int(retry_attempts))
        self.retry_backoff_base = max(0.0, float(retry_backoff_base))

        self._sem = asyncio.Semaphore(self.max_concurrency)
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}.{name}")
        self.metrics: Dict[str, Any] = {
            "allocate_success": 0,
            "allocate_fail": 0,
            "deallocate_success": 0,
            "deallocate_fail": 0,
            "can_provide_calls": 0,
            "status_calls": 0,
            "timeouts": 0,
            "retries": 0,
            "last_error": None,
            "last_updated": None,
        }

    # -------------------------
    # Public API (used by broker)
    # -------------------------

    async def can_provide_capability(self, capability: str, requirements: Dict[str, Any]) -> bool:
        """Check if subsystem can provide the requested capability in a safe, non-throwing manner."""
        self.metrics["can_provide_calls"] += 1
        op_name = "can_provide"
        return await self._call_with_guard(
            op_name=op_name,
            strict=False,
            default=False,
            coro_factory=lambda: self._can_provide(capability, requirements),
        )

    async def allocate_capability(
        self,
        capability: str,
        requirements: Dict[str, Any],
        grant: "CapabilityGrant",
    ) -> Dict[str, Any]:
        """Allocate resources for the capability. Raises SubsystemError on failure."""
        op_name = "allocate"
        result = await self._call_with_guard(
            op_name=op_name,
            strict=True,  # Let broker catch and handle failures
            default=None,
            coro_factory=lambda: self._allocate(capability, requirements, grant),
        )
        if not isinstance(result, dict):
            self.metrics["allocate_fail"] += 1
            self._update_last_error(f"{op_name} returned non-dict result")
            raise SubsystemError(f"{self.name}: allocation returned invalid result type")
        self.metrics["allocate_success"] += 1
        self._touch()
        return result

    async def deallocate_capability(self, grant: "CapabilityGrant") -> bool:
        """Release resources for the capability. Raises SubsystemError on failure."""
        op_name = "deallocate"
        result = await self._call_with_guard(
            op_name=op_name,
            strict=True,  # Raise so broker can record the failure path
            default=False,
            coro_factory=lambda: self._deallocate(grant),
        )
        if result is not True:
            self.metrics["deallocate_fail"] += 1
            self._update_last_error(f"{op_name} reported False")
            raise SubsystemError(f"{self.name}: deallocation reported failure")
        self.metrics["deallocate_success"] += 1
        self._touch()
        return True

    async def get_capability_status(self, grant: "CapabilityGrant") -> Dict[str, Any]:
        """Get current status of allocated capability. Returns a dict with 'error' on failure."""
        self.metrics["status_calls"] += 1
        op_name = "status"
        try:
            result = await self._call_with_guard(
                op_name=op_name,
                strict=False,  # Non-throwing; provide structured error payload
                default=None,
                coro_factory=lambda: self._status(grant),
            )
            if not isinstance(result, dict):
                raise SubsystemError("status returned non-dict")
            self._touch()
            return result
        except Exception as e:
            # Return an error payload rather than raising; broker handles exceptions but will accept a payload too.
            self.logger.error("Failed to obtain status for grant %s: %s", getattr(grant, "grant_id", "unknown"), e)
            self._update_last_error(f"status error: {e!r}")
            return {"error": str(e), "status": "error", "subsystem": self.name}

    # -------------------------
    # Protected abstract hooks
    # -------------------------

    @abstractmethod
    async def _can_provide(self, capability: str, requirements: Dict[str, Any]) -> bool:
        """Subsystem-specific check for capability availability."""
        raise NotImplementedError

    @abstractmethod
    async def _allocate(
        self,
        capability: str,
        requirements: Dict[str, Any],
        grant: "CapabilityGrant",
    ) -> Dict[str, Any]:
        """Subsystem-specific allocation. Return allocation info dict."""
        raise NotImplementedError

    @abstractmethod
    async def _deallocate(self, grant: "CapabilityGrant") -> bool:
        """Subsystem-specific deallocation. Return True on success."""
        raise NotImplementedError

    @abstractmethod
    async def _status(self, grant: "CapabilityGrant") -> Dict[str, Any]:
        """Subsystem-specific status retrieval. Return a dict payload."""
        raise NotImplementedError

    # -------------------------
    # Internal helpers
    # -------------------------

    async def _call_with_guard(
        self,
        *,
        op_name: str,
        strict: bool,
        default: Any,
        coro_factory,
    ) -> Any:
        """Execute a coroutine with concurrency, timeout, retries, and error handling."""
        async with self._sem:
            try:
                return await self._run_with_retries_and_timeout(coro_factory, op_name=op_name)
            except asyncio.TimeoutError as te:
                self.metrics["timeouts"] += 1
                self._update_last_error(f"{op_name} timeout: {te!r}")
                self.logger.error("[%s] %s operation timed out after %.2fs", self.name, op_name, self.operation_timeout)
                if strict:
                    raise SubsystemError(f"{self.name}: {op_name} timed out") from te
                return default
            except asyncio.CancelledError:
                self.logger.warning("[%s] %s operation cancelled", self.name, op_name)
                raise
            except Exception as e:
                self._update_last_error(f"{op_name} error: {e!r}")
                self.logger.exception("[%s] %s operation failed", self.name, op_name)
                if strict:
                    if isinstance(e, SubsystemError):
                        raise
                    raise SubsystemError(f"{self.name}: {op_name} failed") from e
                return default

    async def _run_with_retries_and_timeout(self, coro_factory, *, op_name: str) -> Any:
        """Run a coroutine with retries and timeout applied per attempt."""
        attempt = 0
        last_exc: Optional[BaseException] = None

        while attempt <= self.retry_attempts:
            try:
                return await asyncio.wait_for(coro_factory(), timeout=self.operation_timeout)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                last_exc = e
                if attempt == self.retry_attempts:
                    break
                self.metrics["retries"] += 1
                backoff = self._compute_backoff(attempt)
                self.logger.debug("[%s] %s attempt %d failed: %r; retrying in %.3fs", self.name, op_name, attempt + 1, e, backoff)
                with contextlib.suppress(asyncio.CancelledError):
                    await asyncio.sleep(backoff)
                attempt += 1

        assert last_exc is not None
        raise last_exc

    def _compute_backoff(self, attempt: int) -> float:
        """Exponential backoff with jitter."""
        base = self.retry_backoff_base * (2 ** attempt)
        jitter = random.uniform(0.0, self.retry_backoff_base)
        return base + jitter

    def _update_last_error(self, message: str) -> None:
        self.metrics["last_error"] = message
        self._touch()

    def _touch(self) -> None:
        self.metrics["last_updated"] = datetime.now(timezone.utc).isoformat()


# ============================================================================
# CAPABILITY BROKER IMPLEMENTATION
# ============================================================================

class CapabilityBroker:
    """Central broker for cross-subsystem capability negotiation"""
    
    def __init__(self):
        # Subsystem managers
        self.subsystem_interfaces: Dict[str, SubsystemInterface] = {}
        
        # Active grants and requests
        self.active_grants: Dict[str, CapabilityGrant] = {}
        self.pending_requests: Dict[str, CapabilityRequest] = {}
        
        # Resource tracking
        self.resource_usage: Dict[str, Dict[str, Any]] = {}
        self.capability_metrics: Dict[str, Dict[str, Any]] = {}
        
        # Configuration
        self.max_concurrent_requests = 100
        self.default_grant_duration = timedelta(hours=1)
        self.cleanup_interval = timedelta(minutes=5)
        
        # Background tasks
        self._cleanup_task: Optional[asyncio.Task] = None
        self._monitoring_task: Optional[asyncio.Task] = None
        
        # Event integration
        self.event_bus = get_event_bus()
        
        logger.info("Capability Broker initialized")
    
    async def initialize(self) -> None:
        """Initialize the capability broker"""
        # Start background tasks
        self._cleanup_task = asyncio.create_task(self._cleanup_expired_grants())
        self._monitoring_task = asyncio.create_task(self._monitor_resource_usage())
        
        # Subscribe to relevant events
        self.event_bus.subscribe(
            CapabilityEvent,
            self._handle_capability_event,
            priority=EventPriority.HIGH
        )
        
        logger.info("Capability Broker initialized and monitoring started")
    
    def register_subsystem(self, name: str, interface: SubsystemInterface) -> None:
        """Register a subsystem interface for capability allocation"""
        self.subsystem_interfaces[name] = interface
        logger.info(f"Registered subsystem interface: {name}")
    
    async def request_capability(
        self,
        capability_name: str,
        requester_id: str,
        requester_type: str = "agent",
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        scope: str = "session",
        priority: ResourcePriority = ResourcePriority.NORMAL,
        duration_minutes: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Tuple[bool, str, Optional[CapabilityGrant]]:
        """Request a capability from the broker"""
        
        # Create request
        request = CapabilityRequest(
            capability_name=capability_name,
            requester_id=requester_id,
            requester_type=requester_type,
            user_id=user_id,
            session_id=session_id,
            scope=scope,
            priority=priority,
            duration_minutes=duration_minutes,
            context=context or {}
        )
        
        # Validate request
        validation_result = await self._validate_request(request)
        if not validation_result[0]:
            # Update metrics on denial
            self._update_capability_metrics(capability_name, "denied")
            return False, validation_result[1], None
        
        # Check user projection permissions
        permission_result = await self._check_projection_permissions(request)
        if not permission_result[0]:
            # Update metrics on denial
            self._update_capability_metrics(capability_name, "denied")
            return False, permission_result[1], None
        
        # Allocate capability
        allocation_result = await self._allocate_capability(request)
        if not allocation_result[0]:
            # Update metrics on denial (allocation failure)
            self._update_capability_metrics(capability_name, "denied")
            return False, allocation_result[1], None
        
        grant = allocation_result[2]
        
        # Publish capability granted event
        await publish_capability_requested(
            capability_name=capability_name,
            requester=requester_id,
            subsystem="capability_broker",
            priority=EventPriority.HIGH,
            user_id=user_id,
            session_id=session_id
        )
        
        logger.info(f"Capability granted: {capability_name} to {requester_id}")
        return True, "Capability granted successfully", grant
    
    async def release_capability(self, grant_id: str) -> Tuple[bool, str]:
        """Release a previously granted capability"""
        
        if grant_id not in self.active_grants:
            return False, "Grant not found"
        
        grant = self.active_grants[grant_id]
        
        # Deallocate from subsystems
        deallocation_success = True
        for subsystem_name in grant.subsystem_allocations:
            subsystem = self.subsystem_interfaces.get(subsystem_name)
            if subsystem:
                try:
                    await subsystem.deallocate_capability(grant)
                except Exception as e:
                    logger.error(f"Failed to deallocate from {subsystem_name}: {e}")
                    deallocation_success = False
        
        # Update grant status
        grant.status = CapabilityStatus.REVOKED
        del self.active_grants[grant_id]
        
        # Update metrics
        self._update_capability_metrics(grant.request.capability_name, "released")
        
        status_msg = "Capability released successfully"
        if not deallocation_success:
            status_msg += " (with some subsystem errors)"
        
        return True, status_msg
    
    async def get_capability_status(self, grant_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a granted capability"""
        
        if grant_id not in self.active_grants:
            return None
        
        grant = self.active_grants[grant_id]
        
        # Collect status from subsystems
        subsystem_statuses = {}
        for subsystem_name in grant.subsystem_allocations:
            subsystem = self.subsystem_interfaces.get(subsystem_name)
            if subsystem:
                try:
                    status = await subsystem.get_capability_status(grant)
                    subsystem_statuses[subsystem_name] = status
                except Exception as e:
                    subsystem_statuses[subsystem_name] = {"error": str(e)}
        
        return {
            "grant_id": grant.grant_id,
            "capability_name": grant.request.capability_name,
            "status": grant.status.value,
            "granted_at": grant.granted_at.isoformat() if grant.granted_at else None,
            "expires_at": grant.expires_at.isoformat() if grant.expires_at else None,
            "access_count": grant.access_count,
            "last_accessed": grant.last_accessed.isoformat() if grant.last_accessed else None,
            "subsystem_statuses": subsystem_statuses,
            "usage_stats": grant.usage_stats
        }
    
    async def list_available_capabilities(
        self,
        user_id: Optional[str] = None,
        scope: str = "session"
    ) -> Dict[str, Any]:
        """List capabilities available to user based on their projection"""
        
        available_capabilities = {}
        
        for capability_name, definition in CAPABILITY_REGISTRY.items():
            # Check if user projection allows this capability
            if user_id:
                permission_check = await self._check_user_capability_permission(
                    user_id, capability_name, scope
                )
                if not permission_check:
                    continue
            
            # Check if subsystems can provide this capability
            can_provide = await self._check_subsystem_availability(definition)
            
            available_capabilities[capability_name] = {
                "description": definition.description,
                "capability_type": definition.capability_type.value,
                "required_subsystems": definition.required_subsystems,
                "available": can_provide,
                "max_concurrent_grants": definition.max_concurrent_grants,
                "current_grants": len([g for g in self.active_grants.values() 
                                     if g.request.capability_name == capability_name])
            }
        
        return available_capabilities
    
    async def get_broker_statistics(self) -> Dict[str, Any]:
        """Get capability broker statistics and metrics"""
        
        total_grants = len(self.active_grants)
        grants_by_status = {}
        grants_by_capability = {}
        
        for grant in self.active_grants.values():
            # Count by status
            status = grant.status.value
            grants_by_status[status] = grants_by_status.get(status, 0) + 1
            
            # Count by capability
            capability = grant.request.capability_name
            grants_by_capability[capability] = grants_by_capability.get(capability, 0) + 1
        
        return {
            "total_active_grants": total_grants,
            "pending_requests": len(self.pending_requests),
            "registered_subsystems": list(self.subsystem_interfaces.keys()),
            "grants_by_status": grants_by_status,
            "grants_by_capability": grants_by_capability,
            "capability_metrics": self.capability_metrics,
            "resource_usage": self.resource_usage
        }
    
    async def _validate_request(self, request: CapabilityRequest) -> Tuple[bool, str]:
        """Validate capability request"""
        
        # Check if capability exists
        if request.capability_name not in CAPABILITY_REGISTRY:
            return False, f"Unknown capability: {request.capability_name}"
        
        # Check request limits
        if len(self.pending_requests) >= self.max_concurrent_requests:
            return False, "Too many concurrent requests"
        
        # Check prerequisites
        definition = CAPABILITY_REGISTRY[request.capability_name]
        for prerequisite in definition.prerequisites:
            # Check if user has prerequisite capability active
            has_prerequisite = any(
                grant.request.capability_name == prerequisite and 
                grant.request.user_id == request.user_id and
                grant.status == CapabilityStatus.GRANTED
                for grant in self.active_grants.values()
            )
            if not has_prerequisite:
                return False, f"Missing prerequisite capability: {prerequisite}"
        
        # Check concurrent grant limits
        if definition.max_concurrent_grants:
            current_grants = len([
                grant for grant in self.active_grants.values()
                if grant.request.capability_name == request.capability_name and
                grant.status == CapabilityStatus.GRANTED
            ])
            if current_grants >= definition.max_concurrent_grants:
                return False, f"Maximum concurrent grants reached for {request.capability_name}"
        
        return True, "Request valid"
    
    async def _check_projection_permissions(self, request: CapabilityRequest) -> Tuple[bool, str]:
        """Check if user's projection allows this capability"""
        
        if not request.user_id:
            # System requests don't need projection permission
            return True, "System request approved"
        
        # This would integrate with the user registry projection system
        # For now, we'll simulate the check
        allowed_capabilities = await self._get_user_allowed_capabilities(
            request.user_id, request.scope
        )
        
        if request.capability_name not in allowed_capabilities:
            return False, f"Capability {request.capability_name} not allowed in user projection for scope {request.scope}"
        
        return True, "Projection permissions valid"
    
    async def _get_user_allowed_capabilities(self, user_id: str, scope: str) -> Set[str]:
        """Resolve allowed capabilities by generating the user's projection for the requested scope.
        
        Implementation details:
        - Loads the user's profile via SecurePythonModuleStore (non-blocking).
        - Uses projection_registry.generate_projection_from_registry to compute capabilities.
        - Extracts truthy flags under projection['capabilities']['flags'] as allowed capability names.
        - Returns empty set on any failure to enforce least-privilege by default.
        """
        try:
            store = SecurePythonModuleStore()  # defaults to "profiles" directory
            profile = await store.load(user_id)
            if not profile:
                logger.warning("Projection permission check: profile not found for user_id=%s", user_id)
                return set()
            
            projection = generate_projection_from_registry(profile, scope=scope)
            flags = projection.get("capabilities", {}).get("flags", {})
            # Permit only truthy flags and normalize to known capabilities
            allowed = {name for name, enabled in flags.items() if enabled is True}
            # Ensure we only allow capabilities known to the broker registry
            known = set(CAPABILITY_REGISTRY.keys())
            return allowed & known
        except Exception as e:
            logger.error("Failed to generate projection for user_id=%s scope=%s: %s", user_id, scope, e)
            return set()
    
    async def _check_user_capability_permission(
        self, 
        user_id: str, 
        capability_name: str, 
        scope: str
    ) -> bool:
        """Check if user projection allows specific capability"""
        allowed_capabilities = await self._get_user_allowed_capabilities(user_id, scope)
        return capability_name in allowed_capabilities
    
    async def _check_subsystem_availability(self, definition: CapabilityDefinition) -> bool:
        """Check if required subsystems can provide the capability"""
        
        for subsystem_name in definition.required_subsystems:
            subsystem = self.subsystem_interfaces.get(subsystem_name)
            if not subsystem:
                return False
            
            try:
                can_provide = await subsystem.can_provide_capability(
                    definition.name, definition.resource_requirements
                )
                if not can_provide:
                    return False
            except Exception as e:
                logger.error(f"Error checking {subsystem_name} availability: {e}")
                return False
        
        return True
    
    async def _allocate_capability(self, request: CapabilityRequest) -> Tuple[bool, str, Optional[CapabilityGrant]]:
        """Allocate capability across required subsystems"""
        
        definition = CAPABILITY_REGISTRY[request.capability_name]
        
        # Create grant
        grant = CapabilityGrant(
            request=request,
            status=CapabilityStatus.PENDING
        )
        
        # Set expiration
        duration = timedelta(minutes=request.duration_minutes or definition.default_grant_duration_minutes)
        grant.expires_at = datetime.now(timezone.utc) + duration
        
        # Allocate from each required subsystem
        allocation_errors = []
        successful_allocations = []
        
        for subsystem_name in definition.required_subsystems:
            subsystem = self.subsystem_interfaces.get(subsystem_name)
            if not subsystem:
                allocation_errors.append(f"Subsystem {subsystem_name} not available")
                continue
            
            try:
                allocation_result = await subsystem.allocate_capability(
                    request.capability_name,
                    definition.resource_requirements,
                    grant
                )
                grant.subsystem_allocations[subsystem_name] = allocation_result
                successful_allocations.append(subsystem_name)
            except Exception as e:
                allocation_errors.append(f"Failed to allocate from {subsystem_name}: {e}")
        
        # Check if all allocations succeeded
        if allocation_errors:
            # Rollback successful allocations
            for subsystem_name in successful_allocations:
                subsystem = self.subsystem_interfaces[subsystem_name]
                try:
                    await subsystem.deallocate_capability(grant)
                except Exception as e:
                    logger.error(f"Failed to rollback allocation from {subsystem_name}: {e}")
            
            return False, "; ".join(allocation_errors), None
        
        # Mark as granted
        grant.status = CapabilityStatus.GRANTED
        grant.granted_at = datetime.now(timezone.utc)
        
        # Store grant
        self.active_grants[grant.grant_id] = grant
        
        # Update metrics
        self._update_capability_metrics(request.capability_name, "granted")
        
        return True, "Capability allocated successfully", grant
    
    def _update_capability_metrics(self, capability_name: str, action: str) -> None:
        """Update capability usage metrics"""
        
        if capability_name not in self.capability_metrics:
            self.capability_metrics[capability_name] = {
                "requests": 0,
                "grants": 0,
                "denials": 0,
                "releases": 0
            }
        
        if action == "granted":
            self.capability_metrics[capability_name]["grants"] += 1
        elif action == "denied":
            self.capability_metrics[capability_name]["denials"] += 1
        elif action == "released":
            self.capability_metrics[capability_name]["releases"] += 1
        
        self.capability_metrics[capability_name]["requests"] += 1
    
    async def _handle_capability_event(self, event: CapabilityEvent) -> None:
        """Handle capability-related events"""
        
        logger.debug(f"Received capability event: {event.capability_name} - {event.granted}")
        
        # Update internal state based on event
        # This could trigger rebalancing, logging, etc.
    
    async def _cleanup_expired_grants(self) -> None:
        """Background task to clean up expired grants"""
        
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                expired_grants = []
                
                for grant_id, grant in self.active_grants.items():
                    if grant.expires_at and grant.expires_at <= current_time:
                        expired_grants.append(grant_id)
                
                # Release expired grants
                for grant_id in expired_grants:
                    try:
                        await self.release_capability(grant_id)
                        logger.info(f"Released expired capability grant: {grant_id}")
                    except Exception as e:
                        logger.error(f"Failed to release expired grant {grant_id}: {e}")
                
                # Wait before next cleanup
                await asyncio.sleep(self.cleanup_interval.total_seconds())
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")
                await asyncio.sleep(60)  # Wait before retrying
    
    async def _monitor_resource_usage(self) -> None:
        """Background task to monitor resource usage"""
        
        while True:
            try:
                # Collect resource usage from subsystems
                for subsystem_name, subsystem in self.subsystem_interfaces.items():
                    try:
                        # This would call a method to get resource usage
                        # For now, we'll just track basic metrics
                        active_grants_count = len([
                            grant for grant in self.active_grants.values()
                            if subsystem_name in grant.subsystem_allocations
                        ])
                        
                        self.resource_usage[subsystem_name] = {
                            "active_grants": active_grants_count,
                            "last_updated": datetime.now(timezone.utc).isoformat()
                        }
                    except Exception as e:
                        logger.error(f"Failed to get resource usage from {subsystem_name}: {e}")
                
                # Wait before next monitoring cycle
                await asyncio.sleep(60)  # Monitor every minute
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring task: {e}")
                await asyncio.sleep(60)
    
    async def shutdown(self) -> None:
        """Graceful shutdown of capability broker"""
        
        logger.info("Shutting down capability broker...")
        
        # Cancel background tasks
        if self._cleanup_task:
            self._cleanup_task.cancel()
        if self._monitoring_task:
            self._monitoring_task.cancel()
        
        # Release all active grants
        for grant_id in list(self.active_grants.keys()):
            try:
                await self.release_capability(grant_id)
            except Exception as e:
                logger.error(f"Failed to release grant {grant_id} during shutdown: {e}")
        
        logger.info("Capability broker shutdown complete")


# ============================================================================
# GLOBAL BROKER INSTANCE
# ============================================================================

_capability_broker: Optional[CapabilityBroker] = None

def get_capability_broker() -> CapabilityBroker:
    """Get global capability broker instance"""
    global _capability_broker
    if _capability_broker is None:
        _capability_broker = CapabilityBroker()
    return _capability_broker

async def initialize_capability_broker() -> CapabilityBroker:
    """Initialize global capability broker"""
    global _capability_broker
    _capability_broker = CapabilityBroker()
    await _capability_broker.initialize()
    return _capability_broker

async def shutdown_capability_broker() -> None:
    """Shutdown global capability broker"""
    global _capability_broker
    if _capability_broker:
        await _capability_broker.shutdown()
        _capability_broker = None


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize broker
        broker = await initialize_capability_broker()
        
        # Request a capability
        success, message, grant = await broker.request_capability(
            capability_name=CapabilityFlag.GPU_ACCELERATION,
            requester_id="test_agent",
            requester_type="agent",
            user_id="test_user",
            scope="task:ml_training"
        )
        
        print(f"Capability request result: {success} - {message}")
        
        if grant:
            # Get status
            status = await broker.get_capability_status(grant.grant_id)
            print(f"Grant status: {status}")
            
            # Release capability
            success, message = await broker.release_capability(grant.grant_id)
            print(f"Release result: {success} - {message}")
        
        # Get broker statistics
        stats = await broker.get_broker_statistics()
        print(f"Broker statistics: {stats}")
        
        # Cleanup
        await shutdown_capability_broker()
    
    asyncio.run(main())