"""
SOMNUS SYSTEMS - Cross-Subsystem Event Bus
Minimal, High-Performance Event Coordination

ARCHITECTURE PHILOSOPHY:
- Lightweight in-process event coordination
- Type-safe event definitions with payload validation
- Async-first with error isolation
- Zero overhead when no subscribers
- Subsystem autonomy with loose coupling

This event bus enables subsystems to coordinate without tight coupling,
maintaining the sovereignty of each configuration domain while enabling
intelligent cross-system optimizations.
"""

import asyncio
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Any, Callable, Optional, Union, TypeVar, Generic
from uuid import UUID, uuid4
from enum import Enum
import inspect
import weakref

logger = logging.getLogger(__name__)


# ============================================================================
# EVENT TYPE DEFINITIONS
# ============================================================================

class EventPriority(str, Enum):
    """Event priority levels"""
    CRITICAL = "critical"    # System state changes
    HIGH = "high"           # User-initiated changes
    NORMAL = "normal"       # Background operations
    LOW = "low"             # Informational events

class EventScope(str, Enum):
    """Event scope and propagation rules"""
    SYSTEM = "system"       # System-wide events
    SUBSYSTEM = "subsystem" # Within subsystem only
    USER = "user"           # User-specific events
    SESSION = "session"     # Session-specific events

@dataclass
class EventMetadata:
    """Event metadata and tracking information"""
    event_id: str = field(default_factory=lambda: str(uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    priority: EventPriority = EventPriority.NORMAL
    scope: EventScope = EventScope.SYSTEM
    source_subsystem: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    correlation_id: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3

@dataclass
class BaseEvent:
    """Base event class with common metadata"""
    meta: EventMetadata = field(default_factory=EventMetadata)
    payload: Dict[str, Any] = field(default_factory=dict)

# ============================================================================
# SPECIFIC EVENT TYPES
# ============================================================================

@dataclass
class ProfileEvent(BaseEvent):
    """User profile-related events"""
    username: str = ""
    profile_id: str = ""
    changes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ProjectionEvent(BaseEvent):
    """AI projection generation events"""
    username: str = ""
    scope: str = "session"
    agent_id: Optional[str] = None
    projection_hash: str = ""
    capabilities_exposed: List[str] = field(default_factory=list)

@dataclass
class CapabilityEvent(BaseEvent):
    """Capability request/grant events"""
    capability_name: str = ""
    requester: str = ""
    subsystem: str = ""
    granted: bool = False
    reason: Optional[str] = None

@dataclass
class VMEvent(BaseEvent):
    """VM lifecycle and configuration events"""
    vm_id: str = ""
    vm_name: str = ""
    instance_id: Optional[str] = None
    hardware_changes: Dict[str, Any] = field(default_factory=dict)
    personality_changes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MemoryEvent(BaseEvent):
    """Memory system events"""
    memory_id: Optional[str] = None
    memory_type: str = ""
    importance: str = ""
    retention_changes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ResearchEvent(BaseEvent):
    """Research subsystem events"""
    research_id: Optional[str] = None
    research_mode: str = ""
    configuration_changes: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ArtifactEvent(BaseEvent):
    """Artifact system events"""
    artifact_id: Optional[str] = None
    capability_activated: Optional[str] = None
    resource_allocation: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# EVENT HANDLER TYPES
# ============================================================================

EventHandler = Union[
    Callable[[BaseEvent], None],
    Callable[[BaseEvent], asyncio.Future[None]]
]

T = TypeVar('T', bound=BaseEvent)

class EventSubscription(Generic[T]):
    """Event subscription with metadata"""
    
    def __init__(
        self,
        handler: EventHandler,
        event_type: type,
        priority: EventPriority = EventPriority.NORMAL,
        filters: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout_seconds: float = 5.0
    ):
        self.handler = handler
        self.event_type = event_type
        self.priority = priority
        self.filters = filters or {}
        self.max_retries = max_retries
        self.timeout_seconds = timeout_seconds
        self.subscription_id = str(uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.invocation_count = 0
        self.error_count = 0
        self.last_error: Optional[Exception] = None


# ============================================================================
# EVENT BUS IMPLEMENTATION
# ============================================================================

class SomnusEventBus:
    """High-performance, type-safe event bus for cross-subsystem coordination"""
    
    def __init__(self):
        # Event subscriptions by event type
        self._subscriptions: Dict[type, List[EventSubscription]] = defaultdict(list)
        
        # Event statistics and metrics
        self._event_stats: Dict[str, int] = defaultdict(int)
        self._handler_stats: Dict[str, Dict[str, Any]] = defaultdict(dict)
        
        # Error tracking
        self._failed_events: List[Dict[str, Any]] = []
        self._max_failed_events = 1000
        
        # Background task management
        self._background_tasks: set = set()
        self._shutdown_event = asyncio.Event()
        
        # Performance monitoring
        self._performance_metrics: Dict[str, List[float]] = defaultdict(list)
        self._max_metrics_history = 1000
        
        logger.info("Somnus Event Bus initialized")
    
    def subscribe(
        self,
        event_type: type,
        handler: EventHandler,
        priority: EventPriority = EventPriority.NORMAL,
        filters: Optional[Dict[str, Any]] = None,
        max_retries: int = 3,
        timeout_seconds: float = 5.0
    ) -> str:
        """Subscribe to events of a specific type"""
        
        subscription = EventSubscription(
            handler=handler,
            event_type=event_type,
            priority=priority,
            filters=filters,
            max_retries=max_retries,
            timeout_seconds=timeout_seconds
        )
        
        # Add to subscriptions
        self._subscriptions[event_type].append(subscription)
        
        # Sort by priority (critical first)
        self._subscriptions[event_type].sort(
            key=lambda s: list(EventPriority).index(s.priority)
        )
        
        logger.debug(f"Subscribed to {event_type.__name__} events: {subscription.subscription_id}")
        return subscription.subscription_id
    
    def unsubscribe(self, subscription_id: str) -> bool:
        """Unsubscribe from events by subscription ID"""
        
        for event_type, subscriptions in self._subscriptions.items():
            for i, subscription in enumerate(subscriptions):
                if subscription.subscription_id == subscription_id:
                    subscriptions.pop(i)
                    logger.debug(f"Unsubscribed: {subscription_id}")
                    return True
        
        return False
    
    async def publish(
        self,
        event: BaseEvent,
        wait_for_handlers: bool = False,
        timeout_seconds: Optional[float] = None
    ) -> Dict[str, Any]:
        """Publish event to all registered handlers"""
        
        start_time = time.time()
        event_type = type(event)
        event_type_name = event_type.__name__
        
        # Update statistics
        self._event_stats[event_type_name] += 1
        self._event_stats["total"] += 1
        
        # Get subscriptions for this event type
        subscriptions = self._subscriptions.get(event_type, [])
        
        if not subscriptions:
            logger.debug(f"No subscribers for {event_type_name}")
            return {"handlers_called": 0, "duration_ms": 0}
        
        # Filter subscriptions based on event filters
        eligible_subscriptions = []
        for subscription in subscriptions:
            if self._matches_filters(event, subscription.filters):
                eligible_subscriptions.append(subscription)
        
        if not eligible_subscriptions:
            logger.debug(f"No eligible subscribers for {event_type_name} after filtering")
            return {"handlers_called": 0, "duration_ms": 0}
        
        # Execute handlers
        results = {
            "handlers_called": len(eligible_subscriptions),
            "successful": 0,
            "failed": 0,
            "errors": []
        }
        
        if wait_for_handlers:
            # Wait for all handlers to complete
            await self._execute_handlers_sync(event, eligible_subscriptions, results)
        else:
            # Fire and forget - don't wait for handlers
            task = asyncio.create_task(
                self._execute_handlers_async(event, eligible_subscriptions, results)
            )
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
        
        # Record performance metrics
        duration_ms = (time.time() - start_time) * 1000
        results["duration_ms"] = duration_ms
        self._performance_metrics[event_type_name].append(duration_ms)
        
        # Limit metrics history
        if len(self._performance_metrics[event_type_name]) > self._max_metrics_history:
            self._performance_metrics[event_type_name] = \
                self._performance_metrics[event_type_name][-self._max_metrics_history:]
        
        return results
    
    async def _execute_handlers_sync(
        self,
        event: BaseEvent,
        subscriptions: List[EventSubscription],
        results: Dict[str, Any]
    ) -> None:
        """Execute handlers synchronously (wait for completion)"""
        
        for subscription in subscriptions:
            try:
                await self._execute_single_handler(event, subscription)
                results["successful"] += 1
            except Exception as e:
                results["failed"] += 1
                results["errors"].append(str(e))
                logger.error(f"Handler failed: {subscription.subscription_id} - {e}")
    
    async def _execute_handlers_async(
        self,
        event: BaseEvent,
        subscriptions: List[EventSubscription],
        results: Dict[str, Any]
    ) -> None:
        """Execute handlers asynchronously (fire and forget)"""
        
        tasks = []
        for subscription in subscriptions:
            task = asyncio.create_task(self._execute_single_handler(event, subscription))
            tasks.append(task)
        
        # Wait for all tasks to complete
        completed_tasks = await asyncio.gather(*tasks, return_exceptions=True)
        
        for i, result in enumerate(completed_tasks):
            if isinstance(result, Exception):
                results["failed"] += 1
                results["errors"].append(str(result))
                logger.error(f"Handler failed: {subscriptions[i].subscription_id} - {result}")
            else:
                results["successful"] += 1
    
    async def _execute_single_handler(
        self,
        event: BaseEvent,
        subscription: EventSubscription
    ) -> None:
        """Execute a single event handler with retry logic"""
        
        subscription.invocation_count += 1
        
        for attempt in range(subscription.max_retries + 1):
            try:
                # Check if handler is async
                if inspect.iscoroutinefunction(subscription.handler):
                    await asyncio.wait_for(
                        subscription.handler(event),
                        timeout=subscription.timeout_seconds
                    )
                else:
                    # Run sync handler in thread pool
                    loop = asyncio.get_event_loop()
                    await loop.run_in_executor(None, subscription.handler, event)
                
                # Success - break retry loop
                break
                
            except asyncio.TimeoutError:
                error_msg = f"Handler timeout: {subscription.subscription_id}"
                logger.warning(error_msg)
                if attempt == subscription.max_retries:
                    subscription.error_count += 1
                    subscription.last_error = TimeoutError(error_msg)
                    raise
            
            except Exception as e:
                error_msg = f"Handler error: {subscription.subscription_id} - {e}"
                logger.warning(error_msg)
                if attempt == subscription.max_retries:
                    subscription.error_count += 1
                    subscription.last_error = e
                    self._record_failed_event(event, subscription, e)
                    raise
                else:
                    # Wait before retry
                    await asyncio.sleep(0.1 * (attempt + 1))
    
    def _matches_filters(self, event: BaseEvent, filters: Dict[str, Any]) -> bool:
        """Check if event matches subscription filters"""
        
        if not filters:
            return True
        
        for filter_key, filter_value in filters.items():
            # Check nested attributes using dot notation
            event_value = self._get_nested_attr(event, filter_key)
            
            if event_value != filter_value:
                return False
        
        return True
    
    def _get_nested_attr(self, obj: Any, attr_path: str) -> Any:
        """Get nested attribute using dot notation"""
        
        try:
            parts = attr_path.split('.')
            current = obj
            
            for part in parts:
                if hasattr(current, part):
                    current = getattr(current, part)
                else:
                    return None
            
            return current
        except Exception:
            return None
    
    def _record_failed_event(
        self,
        event: BaseEvent,
        subscription: EventSubscription,
        error: Exception
    ) -> None:
        """Record failed event for debugging"""
        
        failed_event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "event_type": type(event).__name__,
            "event_id": event.meta.event_id,
            "subscription_id": subscription.subscription_id,
            "error": str(error),
            "retry_count": subscription.max_retries
        }
        
        self._failed_events.append(failed_event)
        
        # Limit failed events history
        if len(self._failed_events) > self._max_failed_events:
            self._failed_events = self._failed_events[-self._max_failed_events:]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get event bus performance statistics"""
        
        total_subscriptions = sum(len(subs) for subs in self._subscriptions.values())
        
        # Calculate average processing times
        avg_processing_times = {}
        for event_type, times in self._performance_metrics.items():
            if times:
                avg_processing_times[event_type] = sum(times) / len(times)
        
        return {
            "total_subscriptions": total_subscriptions,
            "subscriptions_by_type": {
                event_type.__name__: len(subs) 
                for event_type, subs in self._subscriptions.items()
            },
            "event_stats": dict(self._event_stats),
            "failed_events_count": len(self._failed_events),
            "background_tasks": len(self._background_tasks),
            "average_processing_times_ms": avg_processing_times
        }
    
    def get_failed_events(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent failed events for debugging"""
        return self._failed_events[-limit:]
    
    async def shutdown(self) -> None:
        """Graceful shutdown of event bus"""
        
        logger.info("Shutting down event bus...")
        
        # Set shutdown flag
        self._shutdown_event.set()
        
        # Wait for background tasks to complete
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)
        
        # Clear subscriptions
        self._subscriptions.clear()
        
        logger.info("Event bus shutdown complete")


# ============================================================================
# GLOBAL EVENT BUS INSTANCE
# ============================================================================

# Global event bus instance for easy access across subsystems
_event_bus: Optional[SomnusEventBus] = None

def get_event_bus() -> SomnusEventBus:
    """Get global event bus instance"""
    global _event_bus
    if _event_bus is None:
        _event_bus = SomnusEventBus()
    return _event_bus

def initialize_event_bus() -> SomnusEventBus:
    """Initialize global event bus"""
    global _event_bus
    _event_bus = SomnusEventBus()
    return _event_bus

async def shutdown_event_bus() -> None:
    """Shutdown global event bus"""
    global _event_bus
    if _event_bus:
        await _event_bus.shutdown()
        _event_bus = None


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def subscribe_to_profile_events(handler: EventHandler, **kwargs) -> str:
    """Subscribe to profile-related events"""
    return get_event_bus().subscribe(ProfileEvent, handler, **kwargs)

def subscribe_to_projection_events(handler: EventHandler, **kwargs) -> str:
    """Subscribe to projection generation events"""
    return get_event_bus().subscribe(ProjectionEvent, handler, **kwargs)

def subscribe_to_capability_events(handler: EventHandler, **kwargs) -> str:
    """Subscribe to capability request/grant events"""
    return get_event_bus().subscribe(CapabilityEvent, handler, **kwargs)

def subscribe_to_vm_events(handler: EventHandler, **kwargs) -> str:
    """Subscribe to VM lifecycle events"""
    return get_event_bus().subscribe(VMEvent, handler, **kwargs)

def subscribe_to_memory_events(handler: EventHandler, **kwargs) -> str:
    """Subscribe to memory system events"""
    return get_event_bus().subscribe(MemoryEvent, handler, **kwargs)

def subscribe_to_research_events(handler: EventHandler, **kwargs) -> str:
    """Subscribe to research system events"""
    return get_event_bus().subscribe(ResearchEvent, handler, **kwargs)

def subscribe_to_artifact_events(handler: EventHandler, **kwargs) -> str:
    """Subscribe to artifact system events"""
    return get_event_bus().subscribe(ArtifactEvent, handler, **kwargs)

async def publish_profile_updated(
    username: str,
    profile_id: str,
    changes: Dict[str, Any],
    **kwargs
) -> Dict[str, Any]:
    """Publish profile update event"""
    event = ProfileEvent(
        username=username,
        profile_id=profile_id,
        changes=changes,
        meta=EventMetadata(source_subsystem="user_registry", **kwargs)
    )
    return await get_event_bus().publish(event)

async def publish_projection_generated(
    username: str,
    scope: str,
    projection_hash: str,
    capabilities_exposed: List[str],
    agent_id: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """Publish projection generation event"""
    event = ProjectionEvent(
        username=username,
        scope=scope,
        agent_id=agent_id,
        projection_hash=projection_hash,
        capabilities_exposed=capabilities_exposed,
        meta=EventMetadata(source_subsystem="user_registry", **kwargs)
    )
    return await get_event_bus().publish(event)

async def publish_capability_requested(
    capability_name: str,
    requester: str,
    subsystem: str,
    **kwargs
) -> Dict[str, Any]:
    """Publish capability request event"""
    event = CapabilityEvent(
        capability_name=capability_name,
        requester=requester,
        subsystem=subsystem,
        granted=False,
        meta=EventMetadata(source_subsystem=subsystem, **kwargs)
    )
    return await get_event_bus().publish(event)

async def publish_vm_configuration_changed(
    vm_id: str,
    vm_name: str,
    hardware_changes: Optional[Dict[str, Any]] = None,
    personality_changes: Optional[Dict[str, Any]] = None,
    **kwargs
) -> Dict[str, Any]:
    """Publish VM configuration change event"""
    event = VMEvent(
        vm_id=vm_id,
        vm_name=vm_name,
        hardware_changes=hardware_changes or {},
        personality_changes=personality_changes or {},
        meta=EventMetadata(source_subsystem="vm_manager", **kwargs)
    )
    return await get_event_bus().publish(event)


# ============================================================================
# EXAMPLE EVENT HANDLERS
# ============================================================================

class EventHandlerExamples:
    """Example event handlers for reference"""
    
    @staticmethod
    async def on_profile_updated(event: ProfileEvent) -> None:
        """Handle profile update events"""
        logger.info(f"Profile updated: {event.username} - {list(event.changes.keys())}")
        
        # Example: Invalidate projection cache
        # Example: Update VM personality if identity changed
        # Example: Adjust memory retention if preferences changed
    
    @staticmethod
    async def on_projection_generated(event: ProjectionEvent) -> None:
        """Handle projection generation events"""
        logger.info(f"Projection generated: {event.username} - scope: {event.scope}")
        
        # Example: Update VM configuration based on capabilities
        # Example: Adjust artifact system based on exposed capabilities
        # Example: Cache projection for quick retrieval
    
    @staticmethod
    async def on_capability_requested(event: CapabilityEvent) -> None:
        """Handle capability request events"""
        logger.info(f"Capability requested: {event.capability_name} by {event.requester}")
        
        # Example: Check user projection allows this capability
        # Example: Allocate resources in appropriate subsystem
        # Example: Log capability usage for analytics
    
    @staticmethod
    def on_vm_config_changed(event: VMEvent) -> None:
        """Handle VM configuration changes (sync example)"""
        logger.info(f"VM config changed: {event.vm_name}")
        
        # Example: Update related subsystem configurations
        # Example: Notify user of hardware allocation changes
        # Example: Trigger memory system optimization


if __name__ == "__main__":
    # Example usage and testing
    async def main():
        import logging
        logging.basicConfig(level=logging.INFO)
        
        # Initialize event bus
        bus = initialize_event_bus()
        
        # Subscribe to events
        profile_sub = subscribe_to_profile_events(
            EventHandlerExamples.on_profile_updated,
            priority=EventPriority.HIGH
        )
        
        projection_sub = subscribe_to_projection_events(
            EventHandlerExamples.on_projection_generated
        )
        
        capability_sub = subscribe_to_capability_events(
            EventHandlerExamples.on_capability_requested
        )
        
        vm_sub = subscribe_to_vm_events(
            EventHandlerExamples.on_vm_config_changed
        )
        
        # Publish test events
        await publish_profile_updated(
            username="test_user",
            profile_id="profile_123",
            changes={"identity.display_name": "New Name"}
        )
        
        await publish_projection_generated(
            username="test_user",
            scope="task:coding",
            projection_hash="abc123",
            capabilities_exposed=["gpu_acceleration", "containerization"]
        )
        
        await publish_capability_requested(
            capability_name="gpu_acceleration",
            requester="coding_agent",
            subsystem="artifact_system"
        )
        
        # Wait a moment for async handlers
        await asyncio.sleep(0.1)
        
        # Get statistics
        stats = bus.get_statistics()
        print(f"Event bus statistics: {stats}")
        
        # Cleanup
        await shutdown_event_bus()
    
    asyncio.run(main())