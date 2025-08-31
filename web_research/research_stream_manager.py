"""
MORPHEUS RESEARCH - Real-time Research Stream Manager
Production-grade WebSocket streaming system for live research sessions.

Features:
- Multi-user concurrent research streaming
- Real-time progress updates and source quality scoring
- Dynamic plan evolution visualization
- User intervention points and live collaboration
- Connection resilience and automatic reconnection
- Rate limiting and resource management
- Event-driven architecture with proper async handling
"""

import asyncio
import json
import logging
import time
import weakref
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, List, Optional, Set, Any, Callable, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, asdict
from collections import defaultdict, deque

import os
import base64
import zlib
from pydantic import BaseModel, Field, validator
import redis.asyncio as redis
from fastapi import WebSocket, WebSocketDisconnect
try:
    import jwt  # PyJWT for JWT validation
except Exception:
    jwt = None

logger = logging.getLogger(__name__)


class StreamEventType(str, Enum):
    """Research stream event types"""
    # Session lifecycle
    SESSION_START = "session_start"
    SESSION_PAUSE = "session_pause"
    SESSION_RESUME = "session_resume"
    SESSION_COMPLETE = "session_complete"
    SESSION_ERROR = "session_error"
    
    # Research progress
    PLAN_GENERATED = "plan_generated"
    PLAN_MODIFIED = "plan_modified"
    SEARCH_START = "search_start"
    SEARCH_PROGRESS = "search_progress"
    SEARCH_COMPLETE = "search_complete"
    SOURCE_FOUND = "source_found"
    SOURCE_ANALYZED = "source_analyzed"
    CONTRADICTION_DETECTED = "contradiction_detected"
    
    # User interactions
    USER_INTERVENTION = "user_intervention"
    PLAN_FEEDBACK = "plan_feedback"
    SOURCE_FEEDBACK = "source_feedback"
    
    # System events
    CONNECTION_STATUS = "connection_status"
    RATE_LIMIT_WARNING = "rate_limit_warning"
    SYSTEM_MESSAGE = "system_message"
    
    # Collaboration
    USER_JOINED = "user_joined"
    USER_LEFT = "user_left"
    COLLABORATIVE_EDIT = "collaborative_edit"


class StreamPriority(int, Enum):
    """Event priority levels for stream management"""
    CRITICAL = 1
    HIGH = 2
    NORMAL = 3
    LOW = 4


@dataclass
class StreamEvent:
    """Individual stream event with metadata"""
    event_type: StreamEventType
    session_id: str
    user_id: str
    timestamp: datetime
    data: Dict[str, Any]
    priority: StreamPriority = StreamPriority.NORMAL
    event_id: str = None
    
    def __post_init__(self):
        if self.event_id is None:
            self.event_id = str(uuid4())
        if not isinstance(self.timestamp, datetime):
            self.timestamp = datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to WebSocket-safe dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type.value,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority.value,
            "data": self.data
        }


class ConnectionState(str, Enum):
    """WebSocket connection states"""
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    DISCONNECTING = "disconnecting"
    DISCONNECTED = "disconnected"
    ERROR = "error"


@dataclass
class ResearchConnection:
    """Manages individual research WebSocket connection"""
    websocket: WebSocket
    user_id: str
    session_id: str
    connection_id: str
    state: ConnectionState
    subscribed_events: Set[StreamEventType]
    last_heartbeat: datetime
    rate_limit_tokens: int
    rate_limit_reset: datetime
    filters: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not hasattr(self, 'connection_id') or not self.connection_id:
            self.connection_id = str(uuid4())
        if not hasattr(self, 'last_heartbeat'):
            self.last_heartbeat = datetime.now(timezone.utc)
        if not hasattr(self, 'rate_limit_tokens'):
            self.rate_limit_tokens = 100  # Default rate limit
        if not hasattr(self, 'rate_limit_reset'):
            self.rate_limit_reset = datetime.now(timezone.utc)


class ResearchStreamConfig:
    """Configuration for research streaming"""
    def __init__(self):
        self.max_connections_per_user = 5
        self.max_events_per_second = 50
        self.heartbeat_interval = 30  # seconds
        self.connection_timeout = 300  # seconds
        self.event_buffer_size = 1000
        self.rate_limit_window = 60  # seconds
        self.rate_limit_tokens = 100
        self.redis_url = "redis://localhost:6379"
        self.enable_persistence = True
        self.enable_collaboration = True
        # Security/auth
        self.jwt_secret_env = "STREAM_JWT_SECRET"
        self.jwt_algorithm = "HS256"
        # Payload handling
        self.compression_enabled = True  # zlib+base64
        # Pub/Sub
        self.enable_pubsub = True


class ResearchStreamManager:
    """Production-grade research streaming engine"""
    
    def __init__(self, config: Optional[ResearchStreamConfig] = None):
        self.config = config or ResearchStreamConfig()
        
        # Connection management
        self.connections: Dict[str, ResearchConnection] = {}
        self.session_connections: Dict[str, Set[str]] = defaultdict(set)
        self.user_connections: Dict[str, Set[str]] = defaultdict(set)
        
        # Event management
        self.event_queue: deque = deque(maxlen=self.config.event_buffer_size)
        self.event_handlers: Dict[StreamEventType, List[Callable]] = defaultdict(list)
        
        # State management
        self.active_sessions: Dict[str, Dict[str, Any]] = {}
        self.session_states: Dict[str, Dict[str, Any]] = {}
        
        # Background tasks
        self.background_tasks: Set[asyncio.Task] = set()
        self.is_running = False
        
        # Redis for persistence and scaling
        self.redis_client: Optional[redis.Redis] = None
        
        # Metrics and monitoring
        self.metrics = {
            "total_connections": 0,
            "active_connections": 0,
            "events_sent": 0,
            "events_failed": 0,
            "sessions_active": 0,
            "bytes_sent": 0
        }
    
    async def initialize(self):
        """Initialize the stream manager"""
        logger.info("🌊 Initializing Research Stream Manager...")
        
        # Initialize Redis connection
        if self.config.enable_persistence:
            try:
                self.redis_client = redis.from_url(
                    self.config.redis_url,
                    decode_responses=True,
                    socket_connect_timeout=5,
                    socket_keepalive=True,
                    socket_keepalive_options={}
                )
                await self.redis_client.ping()
                logger.info("✅ Redis connection established")
            except Exception as e:
                logger.warning(f"⚠️ Redis connection failed: {e}")
                self.redis_client = None
        
        # Start background tasks
        self.is_running = True
        self._start_background_tasks()
        
        logger.info("🚀 Research Stream Manager initialized")
    
    async def shutdown(self):
        """Gracefully shutdown the stream manager"""
        logger.info("🛑 Shutting down Research Stream Manager...")
        
        self.is_running = False
        
        # Cancel background tasks
        for task in self.background_tasks:
            task.cancel()
        
        # Close all connections
        for connection in list(self.connections.values()):
            await self._close_connection(connection.connection_id, "server_shutdown")
        
        # Close Redis connection
        if self.redis_client:
            await self.redis_client.close()
        
        logger.info("✅ Research Stream Manager shutdown complete")
    
    def _start_background_tasks(self):
        """Start essential background tasks"""
        tasks = [
            asyncio.create_task(self._heartbeat_monitor()),
            asyncio.create_task(self._cleanup_stale_connections()),
            asyncio.create_task(self._rate_limit_reset()),
            asyncio.create_task(self._metrics_reporter())
        ]
        
        self.background_tasks.update(tasks)
        
        # Clean up completed tasks
        for task in tasks:
            task.add_done_callback(self.background_tasks.discard)
    
    async def connect_user(
        self,
        websocket: WebSocket,
        user_id: str,
        session_id: str,
        auth_token: Optional[str] = None
    ) -> Optional[str]:
        """Connect a user to a research session stream"""
        
        # Validate user connection limits
        if len(self.user_connections[user_id]) >= self.config.max_connections_per_user:
            await websocket.close(code=4001, reason="Too many connections")
            return None
        
        # Create connection
        connection = ResearchConnection(
            websocket=websocket,
            user_id=user_id,
            session_id=session_id,
            connection_id=str(uuid4()),
            state=ConnectionState.CONNECTING,
            subscribed_events=set(),
            last_heartbeat=datetime.now(timezone.utc),
            rate_limit_tokens=self.config.rate_limit_tokens,
            rate_limit_reset=datetime.now(timezone.utc)
        )
        
        try:
            # Accept WebSocket connection
            await websocket.accept()
            connection.state = ConnectionState.CONNECTED
            
            # Authenticate user (placeholder for auth system integration)
            if await self._authenticate_user(user_id, auth_token):
                connection.state = ConnectionState.AUTHENTICATED
            else:
                await websocket.close(code=4003, reason="Authentication failed")
                return None
            
            # Register connection
            self.connections[connection.connection_id] = connection
            self.session_connections[session_id].add(connection.connection_id)
            self.user_connections[user_id].add(connection.connection_id)
            
            # Update metrics
            self.metrics["total_connections"] += 1
            self.metrics["active_connections"] = len(self.connections)
            
            # Send welcome message
            await self._send_event(StreamEvent(
                event_type=StreamEventType.CONNECTION_STATUS,
                session_id=session_id,
                user_id=user_id,
                timestamp=datetime.now(timezone.utc),
                data={
                    "status": "connected",
                    "connection_id": connection.connection_id,
                    "session_id": session_id,
                    "capabilities": {
                        "real_time_updates": True,
                        "user_intervention": True,
                        "collaboration": self.config.enable_collaboration,
                        "persistence": self.config.enable_persistence
                    }
                }
            ), connection.connection_id)
            
            # Broadcast user joined to other session participants
            if self.config.enable_collaboration:
                await self._broadcast_to_session(
                    session_id,
                    StreamEvent(
                        event_type=StreamEventType.USER_JOINED,
                        session_id=session_id,
                        user_id=user_id,
                        timestamp=datetime.now(timezone.utc),
                        data={"user_id": user_id, "connection_id": connection.connection_id}
                    ),
                    exclude_connection=connection.connection_id
                )
            
            logger.info(f"✅ User {user_id} connected to session {session_id}")
            return connection.connection_id
            
        except Exception as e:
            logger.error(f"❌ Connection failed for user {user_id}: {e}")
            if connection.connection_id in self.connections:
                await self._close_connection(connection.connection_id, "connection_error")
            return None
    
    async def disconnect_user(self, connection_id: str, reason: str = "user_disconnect"):
        """Disconnect a user connection"""
        await self._close_connection(connection_id, reason)
    
    async def subscribe_events(
        self,
        connection_id: str,
        event_types: List[StreamEventType]
    ) -> bool:
        """Subscribe connection to specific event types"""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        connection.subscribed_events.update(event_types)
        connection.state = ConnectionState.SUBSCRIBED
        
        # Send subscription confirmation
        await self._send_event(StreamEvent(
            event_type=StreamEventType.SYSTEM_MESSAGE,
            session_id=connection.session_id,
            user_id=connection.user_id,
            timestamp=datetime.now(timezone.utc),
            data={
                "message": f"Subscribed to {len(event_types)} event types",
                "subscribed_events": [et.value for et in event_types]
            }
        ), connection_id)
        
        return True
    
    async def stream_event(self, event: StreamEvent) -> bool:
        """Stream an event to relevant connections"""
        if not self.is_running:
            return False
        
        # Add to event queue for processing
        self.event_queue.append(event)
        
        # Persist to Redis if enabled
        if self.redis_client and self.config.enable_persistence:
            try:
                serialized = json.dumps(event.to_dict())
                await self.redis_client.lpush(
                    f"research:events:{event.session_id}",
                    serialized
                )
                await self.redis_client.expire(
                    f"research:events:{event.session_id}",
                    86400  # 24 hours
                )
                if self.config.enable_pubsub:
                    await self.redis_client.publish(f"research:session:{event.session_id}", serialized)
            except Exception as e:
                logger.warning(f"⚠️ Failed to persist/pubsub event: {e}")
        
        # Broadcast to session connections
        success_count = await self._broadcast_to_session(event.session_id, event)
        
        # Update metrics
        self.metrics["events_sent"] += success_count
        if success_count == 0:
            self.metrics["events_failed"] += 1
        
        # Trigger event handlers
        await self._trigger_event_handlers(event)
        
        return success_count > 0
    
    async def update_research_progress(
        self,
        session_id: str,
        progress_data: Dict[str, Any]
    ):
        """Update research progress for a session"""
        event = StreamEvent(
            event_type=StreamEventType.SEARCH_PROGRESS,
            session_id=session_id,
            user_id="system",
            timestamp=datetime.now(timezone.utc),
            data=progress_data,
            priority=StreamPriority.HIGH
        )
        
        await self.stream_event(event)
    
    async def notify_source_found(
        self,
        session_id: str,
        source_data: Dict[str, Any]
    ):
        """Notify of new source discovery"""
        event = StreamEvent(
            event_type=StreamEventType.SOURCE_FOUND,
            session_id=session_id,
            user_id="system",
            timestamp=datetime.now(timezone.utc),
            data=source_data,
            priority=StreamPriority.NORMAL
        )
        
        await self.stream_event(event)
    
    async def notify_contradiction(
        self,
        session_id: str,
        contradiction_data: Dict[str, Any]
    ):
        """Notify of detected contradiction"""
        event = StreamEvent(
            event_type=StreamEventType.CONTRADICTION_DETECTED,
            session_id=session_id,
            user_id="system",
            timestamp=datetime.now(timezone.utc),
            data=contradiction_data,
            priority=StreamPriority.HIGH
        )
        
        await self.stream_event(event)
    
    async def handle_user_intervention(
        self,
        connection_id: str,
        intervention_data: Dict[str, Any]
    ) -> bool:
        """Handle user intervention in research process"""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # Rate limiting check
        if not await self._check_rate_limit(connection):
            await self._send_event(StreamEvent(
                event_type=StreamEventType.RATE_LIMIT_WARNING,
                session_id=connection.session_id,
                user_id=connection.user_id,
                timestamp=datetime.now(timezone.utc),
                data={"message": "Rate limit exceeded, please slow down"}
            ), connection_id)
            return False
        
        # Process intervention
        event = StreamEvent(
            event_type=StreamEventType.USER_INTERVENTION,
            session_id=connection.session_id,
            user_id=connection.user_id,
            timestamp=datetime.now(timezone.utc),
            data=intervention_data,
            priority=StreamPriority.HIGH
        )
        
        # Broadcast to all session participants
        await self._broadcast_to_session(connection.session_id, event)
        
        return True
    
    def register_event_handler(
        self,
        event_type: StreamEventType,
        handler: Callable[[StreamEvent], None]
    ):
        """Register a handler for specific event types"""
        self.event_handlers[event_type].append(handler)
    
    async def get_session_history(
        self,
        session_id: str,
        limit: int = 100
    ) -> List[Dict[str, Any]]:
        """Get event history for a session"""
        if not self.redis_client:
            return []
        
        try:
            events = await self.redis_client.lrange(
                f"research:events:{session_id}",
                0,
                limit - 1
            )
            return [json.loads(event) for event in events]
        except Exception as e:
            logger.error(f"❌ Failed to get session history: {e}")
            return []
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get current streaming metrics"""
        return {
            **self.metrics,
            "active_sessions": len(self.session_connections),
            "active_users": len(self.user_connections),
            "event_queue_size": len(self.event_queue),
            "uptime": time.time() - getattr(self, '_start_time', time.time())
        }
    
    # Private methods
    
    async def _authenticate_user(self, user_id: str, auth_token: Optional[str]) -> bool:
        """Authenticate user connection using JWT or local token.
        - If STREAM_JWT_SECRET is set, expect a Bearer JWT token and validate subject == user_id.
        - Otherwise, optionally accept a static token from STREAM_LOCAL_TOKEN env var.
        """
        try:
            if not auth_token:
                return False
            token = auth_token.split()[-1]
            jwt_secret = os.getenv(self.config.jwt_secret_env)
            if jwt and jwt_secret:
                # Validate JWT
                decoded = jwt.decode(token, jwt_secret, algorithms=[self.config.jwt_algorithm])
                sub = decoded.get("sub") or decoded.get("user_id")
                if sub and str(sub) == str(user_id):
                    return True
                return False
            # Fallback to static token check
            local_token = os.getenv("STREAM_LOCAL_TOKEN")
            return bool(local_token) and token == local_token
        except Exception:
            return False
    
    async def _check_rate_limit(self, connection: ResearchConnection) -> bool:
        """Check if connection is within rate limits"""
        now = datetime.now(timezone.utc)
        
        # Reset tokens if window expired
        if now >= connection.rate_limit_reset:
            connection.rate_limit_tokens = self.config.rate_limit_tokens
            connection.rate_limit_reset = now.replace(
                second=(now.second + self.config.rate_limit_window) % 60
            )
        
        # Check if tokens available
        if connection.rate_limit_tokens <= 0:
            return False
        
        connection.rate_limit_tokens -= 1
        return True
    
    async def _send_event(self, event: StreamEvent, connection_id: str) -> bool:
        """Send event to specific connection, with optional compression and retries."""
        if connection_id not in self.connections:
            return False
        
        connection = self.connections[connection_id]
        
        # Check if connection is subscribed to this event type
        if (connection.subscribed_events and 
            event.event_type not in connection.subscribed_events):
            return True  # Skip but don't count as failure
        
        # Prepare message
        payload = event.to_dict()
        message = json.dumps(payload)
        if self.config.compression_enabled:
            try:
                compressed = base64.b64encode(zlib.compress(message.encode("utf-8"))).decode("ascii")
                envelope = json.dumps({
                    "compressed": True,
                    "encoding": "zlib+base64",
                    "payload": compressed
                })
                wire_message = envelope
            except Exception:
                # Fallback to uncompressed on error
                wire_message = message
        else:
            wire_message = message
        
        # Retry send with backoff
        for attempt in range(3):
            try:
                await connection.websocket.send_text(wire_message)
                self.metrics["bytes_sent"] += len(wire_message)
                return True
            except WebSocketDisconnect:
                await self._close_connection(connection_id, "connection_lost")
                return False
            except Exception as e:
                if attempt == 2:
                    logger.error(f"❌ Failed to send event to {connection_id}: {e}")
                    return False
                await asyncio.sleep(0.05 * (attempt + 1))
        return False
    
    async def _broadcast_to_session(
        self,
        session_id: str,
        event: StreamEvent,
        exclude_connection: Optional[str] = None
    ) -> int:
        """Broadcast event to all connections in a session"""
        if session_id not in self.session_connections:
            return 0
        
        success_count = 0
        connection_ids = list(self.session_connections[session_id])
        
        for connection_id in connection_ids:
            if connection_id == exclude_connection:
                continue
            
            if await self._send_event(event, connection_id):
                success_count += 1
        
        return success_count
    
    async def _close_connection(self, connection_id: str, reason: str):
        """Close and clean up a connection"""
        if connection_id not in self.connections:
            return
        
        connection = self.connections[connection_id]
        
        try:
            # Send disconnection event to user
            await self._send_event(StreamEvent(
                event_type=StreamEventType.CONNECTION_STATUS,
                session_id=connection.session_id,
                user_id=connection.user_id,
                timestamp=datetime.now(timezone.utc),
                data={"status": "disconnected", "reason": reason}
            ), connection_id)
            
            # Close WebSocket
            if connection.websocket.client_state.name != "CLOSED":
                await connection.websocket.close()
            
        except Exception as e:
            logger.debug(f"Error during connection cleanup: {e}")
        
        # Clean up references
        self.session_connections[connection.session_id].discard(connection_id)
        self.user_connections[connection.user_id].discard(connection_id)
        del self.connections[connection_id]
        
        # Update metrics
        self.metrics["active_connections"] = len(self.connections)
        
        # Broadcast user left to session
        if self.config.enable_collaboration:
            await self._broadcast_to_session(
                connection.session_id,
                StreamEvent(
                    event_type=StreamEventType.USER_LEFT,
                    session_id=connection.session_id,
                    user_id=connection.user_id,
                    timestamp=datetime.now(timezone.utc),
                    data={"user_id": connection.user_id, "reason": reason}
                ),
                exclude_connection=connection_id
            )
        
        logger.info(f"🔌 Connection {connection_id} closed: {reason}")
    
    async def _trigger_event_handlers(self, event: StreamEvent):
        """Trigger registered event handlers"""
        handlers = self.event_handlers.get(event.event_type, [])
        
        for handler in handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception as e:
                logger.error(f"❌ Event handler failed: {e}")
    
    async def _heartbeat_monitor(self):
        """Monitor connection health with heartbeats (no ping; send lightweight status event)."""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    # Send heartbeat as a lightweight event to confirm liveness
                    try:
                        hb_event = StreamEvent(
                            event_type=StreamEventType.CONNECTION_STATUS,
                            session_id=connection.session_id,
                            user_id=connection.user_id,
                            timestamp=now,
                            data={"status": "heartbeat", "ts": now.isoformat()}
                        )
                        await self._send_event(hb_event, connection_id)
                        connection.last_heartbeat = now
                    except Exception:
                        stale_connections.append(connection_id)
                
                # Clean up stale connections
                for connection_id in stale_connections:
                    await self._close_connection(connection_id, "heartbeat_failed")
                
                await asyncio.sleep(self.config.heartbeat_interval)
                
            except Exception as e:
                logger.error(f"❌ Heartbeat monitor error: {e}")
                await asyncio.sleep(5)
    
    async def _cleanup_stale_connections(self):
        """Clean up stale connections periodically"""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                stale_connections = []
                
                for connection_id, connection in self.connections.items():
                    if (now - connection.last_heartbeat).seconds > self.config.connection_timeout:
                        stale_connections.append(connection_id)
                
                for connection_id in stale_connections:
                    await self._close_connection(connection_id, "connection_timeout")
                
                await asyncio.sleep(60)  # Run every minute
                
            except Exception as e:
                logger.error(f"❌ Cleanup task error: {e}")
                await asyncio.sleep(10)
    
    async def _rate_limit_reset(self):
        """Reset rate limit tokens periodically"""
        while self.is_running:
            try:
                now = datetime.now(timezone.utc)
                
                for connection in self.connections.values():
                    if now >= connection.rate_limit_reset:
                        connection.rate_limit_tokens = self.config.rate_limit_tokens
                        connection.rate_limit_reset = now.replace(
                            second=(now.second + self.config.rate_limit_window) % 60
                        )
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"❌ Rate limit reset error: {e}")
                await asyncio.sleep(10)
    
    async def _metrics_reporter(self):
        """Report metrics periodically"""
        while self.is_running:
            try:
                metrics = self.get_metrics()
                logger.info(f"📊 Stream Metrics: {metrics}")
                
                # Persist metrics to Redis if available
                if self.redis_client:
                    await self.redis_client.setex(
                        "research:stream:metrics",
                        300,  # 5 minutes TTL
                        json.dumps(metrics)
                    )
                
                await asyncio.sleep(60)  # Report every minute
                
            except Exception as e:
                logger.error(f"❌ Metrics reporter error: {e}")
                await asyncio.sleep(10)


# Global stream manager instance
_stream_manager: Optional[ResearchStreamManager] = None


async def get_stream_manager() -> ResearchStreamManager:
    """Get or create global stream manager instance"""
    global _stream_manager
    
    if _stream_manager is None:
        _stream_manager = ResearchStreamManager()
        await _stream_manager.initialize()
    
    return _stream_manager


async def cleanup_stream_manager():
    """Clean up global stream manager"""
    global _stream_manager
    
    if _stream_manager:
        await _stream_manager.shutdown()
        _stream_manager = None