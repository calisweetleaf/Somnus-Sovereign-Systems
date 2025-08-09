"""
MORPHEUS CHAT - Development Session Manager
Service for managing the lifecycle of lightweight, conversational dev sessions.

This module provides the core logic for creating, retrieving, updating, and
persisting "Session-as-Repo" objects using the central MemoryManager.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from uuid import UUID

from core.memory_core import MemoryManager, MemoryType, MemoryImportance, MemoryScope
from schemas.dev_session_schemas import DevSession, DevSessionEvent, DevSessionEventType, CodeBlock

# Type aliases for clarity
DevSessionID = UUID
ChatSessionID = UUID
UserID = str

logger = logging.getLogger(__name__)

class DevSessionManager:
    """
    Manages the lifecycle of DevSession objects.

    This class acts as the service layer between the application logic (e.g., WebSocket handlers)
    and the persistence layer (MemoryManager). It handles the creation, retrieval,
    and state transitions of development sessions.
    """

    def __init__(self, memory_manager: MemoryManager):
        """
        Initializes the DevSessionManager.

        Args:
            memory_manager: An instance of the central MemoryManager for persistence.
        """
        if not isinstance(memory_manager, MemoryManager):
            raise TypeError("memory_manager must be an instance of MemoryManager")
            
        self.memory_manager = memory_manager
        # In-memory cache for active development sessions to reduce DB lookups.
        self.active_sessions: Dict[DevSessionID, DevSession] = {}
        self.cache_lock = asyncio.Lock()
        logger.info("Development Session Manager initialized.")

    async def create_session(self, user_id: UserID, chat_session_id: ChatSessionID, title: str = "Untitled Dev Session") -> DevSession:
        """
        Creates a new development session and persists it.

        Args:
            user_id: The ID of the user initiating the session.
            chat_session_id: The ID of the parent chat session.
            title: An optional initial title for the session.

        Returns:
            The newly created DevSession object.
        """
        session = DevSession(
            user_id=user_id,
            chat_session_id=chat_session_id,
            title=title
        )
        
        # Add the creation event to the log
        session.add_event(
            event_type=DevSessionEventType.SESSION_START,
            content=f"Development session '{title}' started by user {user_id}."
        )

        async with self.cache_lock:
            self.active_sessions[session.dev_session_id] = session
        
        await self._save_session(session)
        logger.info(f"Created and persisted new dev session: {session.dev_session_id}")
        return session

    async def get_session(self, dev_session_id: DevSessionID) -> Optional[DevSession]:
        """
        Retrieves a development session, from cache if available, otherwise from memory.

        Args:
            dev_session_id: The ID of the development session to retrieve.

        Returns:
            The DevSession object if found, otherwise None.
        """
        async with self.cache_lock:
            if dev_session_id in self.active_sessions:
                logger.debug(f"Retrieved dev session {dev_session_id} from cache.")
                return self.active_sessions[dev_session_id]

        # If not in cache, load from persistence layer
        session = await self._load_session(dev_session_id)
        if session:
            async with self.cache_lock:
                self.active_sessions[dev_session_id] = session
            logger.info(f"Loaded dev session {dev_session_id} from memory into cache.")
        
        return session

    async def log_event(self, dev_session_id: DevSessionID, event: DevSessionEvent) -> bool:
        """
        Logs a new event to a development session and updates its persistence.

        Args:
            dev_session_id: The ID of the session to update.
            event: The DevSessionEvent to add.

        Returns:
            True if the event was logged successfully, False otherwise.
        """
        session = await self.get_session(dev_session_id)
        if not session:
            logger.warning(f"Attempted to log event for non-existent dev session: {dev_session_id}")
            return False

        session.event_log.append(event)
        session.last_activity = event.timestamp
        
        await self._save_session(session)
        logger.debug(f"Logged event '{event.event_type}' to dev session {dev_session_id}")
        return True

    async def update_code_block(self, dev_session_id: DevSessionID, block_id: UUID, new_content: str) -> bool:
        """
        Updates a code block, creates a diff event, and persists the session.

        Args:
            dev_session_id: The ID of the session containing the block.
            block_id: The ID of the code block to update.
            new_content: The new, full content of the code block.

        Returns:
            True if successful, False otherwise.
        """
        session = await self.get_session(dev_session_id)
        if not session:
            logger.warning(f"Cannot update code block, dev session not found: {dev_session_id}")
            return False

        try:
            diff_patch = session.update_code_block(block_id, new_content)
            
            # Log the modification event with the diff
            await self.log_event(
                dev_session_id,
                DevSessionEvent(
                    event_type=DevSessionEventType.CODE_BLOCK_MODIFIED,
                    content=diff_patch,
                    target_block_id=block_id,
                    metadata={'new_version': session.get_code_block(block_id).version}
                )
            )
            return True
        except KeyError as e:
            logger.error(f"Error updating code block: {e}")
            return False

    async def set_session_status(self, dev_session_id: DevSessionID, status: DevSessionStatus) -> Optional[DevSession]:
        """
        Updates the status of a development session (e.g., to 'paused' or 'completed').

        Args:
            dev_session_id: The ID of the session to update.
            status: The new DevSessionStatus.

        Returns:
            The updated DevSession object, or None if not found.
        """
        session = await self.get_session(dev_session_id)
        if not session:
            return None

        session.status = status
        if status == DevSessionStatus.COMPLETED:
            session.end_time = datetime.now(timezone.utc)
        
        event_type = DevSessionEventType[f"SESSION_{status.name}"]
        session.add_event(
            event_type=event_type,
            content=f"Session status changed to {status.value}"
        )
        await self._save_session(session)
        
        # If session is finalized, remove from active cache to conserve memory
        if status in [DevSessionStatus.COMPLETED, DevSessionStatus.ARCHIVED, DevSessionStatus.PROMOTED]:
            async with self.cache_lock:
                self.active_sessions.pop(dev_session_id, None)

        return session

    async def _save_session(self, session: DevSession):
        """
        Persists the entire DevSession object to the MemoryManager.
        This uses an "upsert" pattern based on the memory_id.

        Args:
            session: The DevSession object to save.
        """
        # We use the dev_session_id as the memory_id for direct lookup.
        # ASSUMPTION: The MemoryManager's store_memory method needs to be adapted
        # to handle an "upsert" operation when a memory_id_override is provided.
        memory_content = session.json()
        
        # A new MemoryType enum member would be needed in memory_core.py
        # For example: DEV_SESSION = "dev_session"
        await self.memory_manager.store_memory(
            user_id=session.user_id,
            content=memory_content,
            memory_type="dev_session", # This would be MemoryType.DEV_SESSION
            importance=MemoryImportance.HIGH, 
            scope=MemoryScope.PRIVATE,
            tags={"dev_session", session.title},
            memory_id_override=session.dev_session_id
        )

    async def _load_session(self, dev_session_id: DevSessionID) -> Optional[DevSession]:
        """
        Loads a DevSession object from the MemoryManager.

        Args:
            dev_session_id: The ID of the session to load.

        Returns:
            A DevSession object if found, otherwise None.
        """
        # ASSUMPTION: The MemoryManager needs a `get_memory_by_id` method that
        # can retrieve a single memory entry by its primary key (the UUID).
        memory_entry = await self.memory_manager.get_memory_by_id(dev_session_id, user_id="*") # Assuming a way to bypass user check for system loads
        
        if memory_entry and memory_entry.get('content'):
            try:
                session_data = json.loads(memory_entry['content'])
                return DevSession(**session_data)
            except (json.JSONDecodeError, TypeError) as e:
                logger.error(f"Failed to parse DevSession from memory for ID {dev_session_id}: {e}")
                return None
        
        logger.warning(f"Could not find or load dev session with ID {dev_session_id} from memory.")
        return None


