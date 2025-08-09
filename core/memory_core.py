"""
Somnus Sovereign Systems- Persistent Memory Management System
Advanced cross-session memory with semantic indexing, encryption, and privacy controls.

Memory Architecture:
- Semantic vector storage with embedding-based retrieval
- Multi-modal memory support (text, code, files, images)
- User-scoped encryption with granular privacy controls
- Importance-based retention and forgetting mechanisms
- Cross-session context reconstruction and continuity
"""

import asyncio
import logging
import pickle
import sqlite3
import hashlib
import json
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set, Union
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from uuid import UUID, uuid4
from contextlib import asynccontextmanager

import numpy as np
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64

from pydantic import BaseModel, Field, validator
from schemas.session import SessionID, UserID

logger = logging.getLogger(__name__)


class MemoryType(str, Enum):
    """Memory classification for retrieval and retention strategies"""
    CORE_FACT = "core_fact"           # Persistent user facts (name, preferences)
    CONVERSATION = "conversation"      # Chat exchanges and context
    DOCUMENT = "document"             # Uploaded files and analysis
    CODE_SNIPPET = "code_snippet"    # Generated/executed code
    TOOL_RESULT = "tool_result"       # Plugin/tool outputs
    CUSTOM_INSTRUCTION = "custom_instruction"  # User-defined behaviors
    SYSTEM_EVENT = "system_event"     # Technical events and errors


class MemoryImportance(str, Enum):
    """Importance levels for retention and retrieval prioritization"""
    CRITICAL = "critical"    # Never forget (user identity, core preferences)
    HIGH = "high"           # Long-term retention (important facts, insights)
    MEDIUM = "medium"       # Medium-term retention (useful context)
    LOW = "low"            # Short-term retention (ephemeral interactions)
    TEMPORARY = "temporary" # Session-only (debugging, system messages)


class MemoryScope(str, Enum):
    """Memory access scope and sharing permissions"""
    PRIVATE = "private"     # User-only access
    SHARED = "shared"       # Shareable with other users (if permitted)
    SYSTEM = "system"       # System-wide (anonymized analytics)
    ARCHIVED = "archived"   # Compressed long-term storage


@dataclass
class MemoryVector:
    """Semantic vector representation with metadata"""
    embedding: np.ndarray
    content: str
    memory_type: MemoryType
    importance: MemoryImportance
    scope: MemoryScope
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    source_session: Optional[SessionID] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class MemoryEntry(BaseModel):
    """Structured memory entry with full metadata"""
    memory_id: UUID = Field(default_factory=uuid4)
    user_id: UserID = Field(description="Owner of this memory")
    content: str = Field(description="Memory content (may be encrypted)")
    content_hash: str = Field(description="SHA-256 hash for deduplication")
    memory_type: MemoryType = Field(description="Memory classification")
    importance: MemoryImportance = Field(description="Retention priority")
    scope: MemoryScope = Field(description="Access permissions")
    
    # Temporal metadata
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = Field(None, description="Optional expiration")
    
    # Usage tracking
    access_count: int = Field(default=0, ge=0)
    relevance_score: float = Field(default=1.0, ge=0, le=1.0)
    
    # Source tracking
    source_session: Optional[SessionID] = Field(None)
    source_type: str = Field(default="user_input")
    
    # Relationships
    related_memories: List[UUID] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)
    
    # Privacy and encryption
    is_encrypted: bool = Field(default=True)
    encryption_key_id: Optional[str] = Field(None)
    
    # Vector storage reference
    vector_id: Optional[str] = Field(None, description="ChromaDB vector ID")
    
    def update_access(self):
        """Update access tracking"""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
    
    def calculate_retention_score(self) -> float:
        """Calculate dynamic retention score based on usage and importance"""
        age_days = (datetime.now(timezone.utc) - self.created_at).days
        recency_factor = max(0.1, 1.0 - (age_days / 365))  # Decay over a year
        
        importance_weights = {
            MemoryImportance.CRITICAL: 1.0,
            MemoryImportance.HIGH: 0.8,
            MemoryImportance.MEDIUM: 0.6,
            MemoryImportance.LOW: 0.4,
            MemoryImportance.TEMPORARY: 0.1
        }
        
        access_factor = min(1.0, self.access_count / 10)  # Normalize access frequency
        importance_factor = importance_weights[self.importance]
        
        return (recency_factor * 0.3 + access_factor * 0.3 + importance_factor * 0.4)
    
    @property
    def should_retain(self) -> bool:
        """Determine if memory should be retained based on score and policies"""
        if self.importance == MemoryImportance.CRITICAL:
            return True
        if self.importance == MemoryImportance.TEMPORARY:
            return False
        
        return self.calculate_retention_score() > 0.3


class MemoryConfiguration(BaseModel):
    """Comprehensive memory system configuration"""
    # Storage configuration
    vector_db_path: str = Field(default="data/memory/vectors")
    metadata_db_path: str = Field(default="data/memory/metadata.db")
    max_memories_per_user: int = Field(default=50000, ge=1000)
    
    # Embedding configuration
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    embedding_dimension: int = Field(default=384, ge=128)
    batch_size: int = Field(default=32, ge=1, le=256)
    
    # Retention policies
    retention_policies: Dict[MemoryImportance, int] = Field(
        default_factory=lambda: {
            MemoryImportance.CRITICAL: -1,      # Never expire
            MemoryImportance.HIGH: 365,         # 1 year
            MemoryImportance.MEDIUM: 90,        # 3 months
            MemoryImportance.LOW: 30,           # 1 month
            MemoryImportance.TEMPORARY: 1       # 1 day
        }
    )
    
    # Privacy and encryption
    encryption_enabled: bool = Field(default=True)
    user_data_isolation: bool = Field(default=True)
    
    # Performance tuning
    similarity_threshold: float = Field(default=0.7, ge=0.1, le=1.0)
    max_retrieval_results: int = Field(default=20, ge=1, le=100)
    background_cleanup_interval: int = Field(default=3600, ge=300)
    
    # Advanced features
    semantic_clustering: bool = Field(default=True)
    cross_modal_indexing: bool = Field(default=True)
    importance_learning: bool = Field(default=True)


class MemoryEncryption:
    """Advanced encryption for user memory with key derivation"""
    
    def __init__(self, master_key: Optional[bytes] = None):
        self.master_key = master_key or Fernet.generate_key()
        self.user_keys: Dict[UserID, Fernet] = {}
    
    def _derive_user_key(self, user_id: UserID, password: Optional[str] = None) -> Fernet:
        """Derive user-specific encryption key"""
        if user_id in self.user_keys:
            return self.user_keys[user_id]
        
        # Use master key + user_id for key derivation
        salt = hashlib.sha256(user_id.encode()).digest()[:16]
        
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        
        # Derive key from master key + user_id
        key_material = self.master_key + user_id.encode()
        key = base64.urlsafe_b64encode(kdf.derive(key_material))
        
        fernet = Fernet(key)
        self.user_keys[user_id] = fernet
        return fernet
    
    def encrypt_memory(self, content: str, user_id: UserID) -> Tuple[bytes, str]:
        """Encrypt memory content for user"""
        fernet = self._derive_user_key(user_id)
        encrypted = fernet.encrypt(content.encode())
        key_id = hashlib.sha256(user_id.encode()).hexdigest()[:16]
        return encrypted, key_id
    
    def decrypt_memory(self, encrypted_content: bytes, user_id: UserID) -> str:
        """Decrypt memory content for user"""
        fernet = self._derive_user_key(user_id)
        decrypted = fernet.decrypt(encrypted_content)
        return decrypted.decode()


class MemoryManager:
    """
    Production-grade persistent memory management system.
    
    Implements advanced semantic indexing, encryption, and retention policies
    for cross-session user memory with privacy controls and performance optimization.
    """
    
    def __init__(self, config: MemoryConfiguration):
        self.config = config
        self.encryption = MemoryEncryption()
        self.embedding_model = None
        
        # Storage components
        self.vector_db = None
        self.metadata_db_path = Path(config.metadata_db_path)
        self.metadata_db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Performance tracking
        self.metrics = {
            'memories_stored': 0,
            'memories_retrieved': 0,
            'cache_hits': 0,
            'cache_misses': 0,
            'retention_cleanups': 0
        }
        
        # Background tasks
        self.cleanup_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        logger.info("Memory manager initialized")
    
    async def initialize(self):
        """Initialize memory system components"""
        try:
            # Initialize embedding model
            logger.info(f"Loading embedding model: {self.config.embedding_model}")
            self.embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Initialize vector database
            await self._initialize_vector_db()
            
            # Initialize metadata database
            await self._initialize_metadata_db()
            
            # Start background tasks
            self.cleanup_task = asyncio.create_task(self._cleanup_loop())
            
            logger.info("Memory system initialized successfully")
            
        except Exception as e:
            logger.error(f"Memory system initialization failed: {e}")
            raise
    
    async def _initialize_vector_db(self):
        """Initialize ChromaDB for vector storage"""
        vector_path = Path(self.config.vector_db_path)
        vector_path.mkdir(parents=True, exist_ok=True)
        
        # Configure ChromaDB with persistence
        settings = Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=str(vector_path),
            anonymized_telemetry=False
        )
        
        self.vector_db = chromadb.Client(settings)
        logger.info(f"Vector database initialized at {vector_path}")
    
    async def _initialize_metadata_db(self):
        """Initialize SQLite database for memory metadata"""
        async with self._get_db_connection() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS memories (
                    memory_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    memory_type TEXT NOT NULL,
                    importance TEXT NOT NULL,
                    scope TEXT NOT NULL,
                    created_at TIMESTAMP NOT NULL,
                    last_accessed TIMESTAMP NOT NULL,
                    expires_at TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    relevance_score REAL DEFAULT 1.0,
                    source_session TEXT,
                    source_type TEXT,
                    related_memories TEXT,  -- JSON array
                    tags TEXT,             -- JSON array
                    is_encrypted BOOLEAN DEFAULT TRUE,
                    encryption_key_id TEXT,
                    vector_id TEXT,
                    metadata TEXT          -- JSON blob for additional data
                )
            """)
            
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS encrypted_content (
                    memory_id TEXT PRIMARY KEY,
                    encrypted_data BLOB NOT NULL,
                    FOREIGN KEY (memory_id) REFERENCES memories (memory_id)
                )
            """)
            
            # Create indexes for performance
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_user_id ON memories (user_id)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_type ON memories (memory_type)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_importance ON memories (importance)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_created_at ON memories (created_at)")
            await conn.execute("CREATE INDEX IF NOT EXISTS idx_content_hash ON memories (content_hash)")
            
            await conn.commit()
        
        logger.info("Metadata database initialized")
    
    @asynccontextmanager
    async def _get_db_connection(self):
        """Get async database connection with proper error handling"""
        conn = sqlite3.connect(self.metadata_db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()
    
    async def store_memory(
        self, 
        user_id: UserID, 
        content: str, 
        memory_type: MemoryType = MemoryType.CONVERSATION,
        importance: MemoryImportance = MemoryImportance.MEDIUM,
        scope: MemoryScope = MemoryScope.PRIVATE,
        source_session: Optional[SessionID] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> UUID:
        """
        Store new memory with semantic indexing and encryption.
        
        Returns the memory_id for future reference.
        """
        try:
            # Create memory entry
            memory_entry = MemoryEntry(
                user_id=user_id,
                content=content,
                content_hash=hashlib.sha256(content.encode()).hexdigest(),
                memory_type=memory_type,
                importance=importance,
                scope=scope,
                source_session=source_session,
                tags=tags or [],
            )
            
            # Check for duplicates
            existing_memory = await self._find_duplicate_memory(user_id, memory_entry.content_hash)
            if existing_memory:
                # Update existing memory instead of creating duplicate
                await self._update_memory_access(existing_memory['memory_id'])
                return UUID(existing_memory['memory_id'])
            
            # Encrypt content if enabled
            if self.config.encryption_enabled:
                encrypted_content, key_id = self.encryption.encrypt_memory(content, user_id)
                memory_entry.is_encrypted = True
                memory_entry.encryption_key_id = key_id
            else:
                encrypted_content = content.encode()
                memory_entry.is_encrypted = False
            
            # Generate embedding
            embedding = await self._generate_embedding(content)
            
            # Store in vector database
            collection_name = f"user_{hashlib.sha256(user_id.encode()).hexdigest()[:16]}"
            collection = await self._get_or_create_collection(collection_name)
            
            vector_id = str(memory_entry.memory_id)
            collection.add(
                embeddings=[embedding.tolist()],
                documents=[content[:1000]],  # Truncate for ChromaDB
                metadatas=[{
                    'memory_type': memory_type.value,
                    'importance': importance.value,
                    'created_at': memory_entry.created_at.isoformat(),
                    'tags': json.dumps(tags or [])
                }],
                ids=[vector_id]
            )
            
            memory_entry.vector_id = vector_id
            
            # Store in metadata database
            async with self._get_db_connection() as conn:
                await conn.execute("""
                    INSERT INTO memories (
                        memory_id, user_id, content_hash, memory_type, importance, scope,
                        created_at, last_accessed, access_count, source_session,
                        source_type, related_memories, tags, is_encrypted,
                        encryption_key_id, vector_id, metadata
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    str(memory_entry.memory_id), user_id, memory_entry.content_hash,
                    memory_type.value, importance.value, scope.value,
                    memory_entry.created_at.isoformat(), memory_entry.last_accessed.isoformat(),
                    memory_entry.access_count, str(source_session) if source_session else None,
                    memory_entry.source_type, json.dumps(memory_entry.related_memories),
                    json.dumps(memory_entry.tags), memory_entry.is_encrypted,
                    memory_entry.encryption_key_id, vector_id,
                    json.dumps(metadata or {})
                ))
                
                # Store encrypted content separately
                await conn.execute("""
                    INSERT INTO encrypted_content (memory_id, encrypted_data)
                    VALUES (?, ?)
                """, (str(memory_entry.memory_id), encrypted_content))
                
                await conn.commit()
            
            self.metrics['memories_stored'] += 1
            logger.info(f"Stored memory {memory_entry.memory_id} for user {user_id}")
            
            return memory_entry.memory_id
            
        except Exception as e:
            logger.error(f"Failed to store memory for user {user_id}: {e}")
            raise
    
    async def retrieve_memories(
        self,
        user_id: UserID,
        query: Optional[str] = None,
        memory_types: Optional[List[MemoryType]] = None,
        importance_threshold: MemoryImportance = MemoryImportance.LOW,
        limit: int = 20,
        include_content: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories using semantic search and filtering.
        
        Returns list of memory dictionaries with content and metadata.
        """
        try:
            # Get user's vector collection
            collection_name = f"user_{hashlib.sha256(user_id.encode()).hexdigest()[:16]}"
            collection = await self._get_or_create_collection(collection_name)
            
            # Build metadata filters
            where_conditions = []
            if memory_types:
                where_conditions.append({
                    'memory_type': {'$in': [mt.value for mt in memory_types]}
                })
            
            importance_order = [
                MemoryImportance.CRITICAL, MemoryImportance.HIGH, 
                MemoryImportance.MEDIUM, MemoryImportance.LOW, MemoryImportance.TEMPORARY
            ]
            valid_importance = importance_order[:importance_order.index(importance_threshold) + 1]
            where_conditions.append({
                'importance': {'$in': [imp.value for imp in valid_importance]}
            })
            
            where_filter = {'$and': where_conditions} if where_conditions else None
            
            # Perform semantic search if query provided
            if query:
                query_embedding = await self._generate_embedding(query)
                results = collection.query(
                    query_embeddings=[query_embedding.tolist()],
                    n_results=min(limit, self.config.max_retrieval_results),
                    where=where_filter,
                    include=['documents', 'metadatas', 'distances']
                )
                
                memory_ids = results['ids'][0] if results['ids'] else []
                similarities = [1 - d for d in results['distances'][0]] if results['distances'] else []
            else:
                # Get all memories for user with filters
                all_results = collection.get(
                    where=where_filter,
                    include=['documents', 'metadatas']
                )
                memory_ids = all_results['ids'][:limit] if all_results['ids'] else []
                similarities = [1.0] * len(memory_ids)  # No semantic ranking
            
            # Retrieve full memory data from metadata database
            memories = []
            for memory_id, similarity in zip(memory_ids, similarities):
                memory_data = await self._get_memory_by_id(memory_id, user_id, include_content)
                if memory_data:
                    memory_data['similarity_score'] = similarity
                    memories.append(memory_data)
                    await self._update_memory_access(memory_id)
            
            # Sort by importance and similarity
            memories.sort(key=lambda m: (
                m.get('importance_rank', 5),  # CRITICAL=0, TEMPORARY=4
                -m.get('similarity_score', 0)
            ))
            
            self.metrics['memories_retrieved'] += len(memories)
            logger.info(f"Retrieved {len(memories)} memories for user {user_id}")
            
            return memories
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories for user {user_id}: {e}")
            return []
    
    async def _generate_embedding(self, text: str) -> np.ndarray:
        """Generate semantic embedding for text"""
        if not self.embedding_model:
            raise RuntimeError("Embedding model not initialized")
        
        # Run embedding generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embedding = await loop.run_in_executor(
            None, self.embedding_model.encode, text
        )
        return embedding
    
    async def _get_or_create_collection(self, collection_name: str):
        """Get or create ChromaDB collection for user"""
        try:
            collection = self.vector_db.get_collection(collection_name)
        except ValueError:
            # Collection doesn't exist, create it
            collection = self.vector_db.create_collection(
                name=collection_name,
                metadata={"description": f"Memory vectors for user collection {collection_name}"}
            )
        return collection
    
    async def _find_duplicate_memory(self, user_id: UserID, content_hash: str) -> Optional[Dict[str, Any]]:
        """Find existing memory with same content hash"""
        async with self._get_db_connection() as conn:
            cursor = await conn.execute("""
                SELECT memory_id, created_at FROM memories
                WHERE user_id = ? AND content_hash = ?
                ORDER BY created_at DESC LIMIT 1
            """, (user_id, content_hash))
            row = cursor.fetchone()
            return dict(row) if row else None
    
    async def _get_memory_by_id(
        self, 
        memory_id: str, 
        user_id: UserID, 
        include_content: bool = True
    ) -> Optional[Dict[str, Any]]:
        """Retrieve memory by ID with optional content decryption"""
        async with self._get_db_connection() as conn:
            cursor = await conn.execute("""
                SELECT m.*, ec.encrypted_data
                FROM memories m
                LEFT JOIN encrypted_content ec ON m.memory_id = ec.memory_id
                WHERE m.memory_id = ? AND m.user_id = ?
            """, (memory_id, user_id))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            memory_data = dict(row)
            
            # Decrypt content if needed and requested
            if include_content and memory_data['encrypted_data']:
                if memory_data['is_encrypted']:
                    try:
                        decrypted_content = self.encryption.decrypt_memory(
                            memory_data['encrypted_data'], user_id
                        )
                        memory_data['content'] = decrypted_content
                    except Exception as e:
                        logger.error(f"Failed to decrypt memory {memory_id}: {e}")
                        memory_data['content'] = "[DECRYPTION_FAILED]"
                else:
                    memory_data['content'] = memory_data['encrypted_data'].decode()
            
            # Parse JSON fields
            memory_data['related_memories'] = json.loads(memory_data.get('related_memories', '[]'))
            memory_data['tags'] = json.loads(memory_data.get('tags', '[]'))
            memory_data['metadata'] = json.loads(memory_data.get('metadata', '{}'))
            
            # Add importance ranking for sorting
            importance_ranks = {
                MemoryImportance.CRITICAL.value: 0,
                MemoryImportance.HIGH.value: 1,
                MemoryImportance.MEDIUM.value: 2,
                MemoryImportance.LOW.value: 3,
                MemoryImportance.TEMPORARY.value: 4
            }
            memory_data['importance_rank'] = importance_ranks.get(memory_data['importance'], 5)
            
            return memory_data
    
    async def _update_memory_access(self, memory_id: str):
        """Update memory access tracking"""
        async with self._get_db_connection() as conn:
            await conn.execute("""
                UPDATE memories
                SET last_accessed = ?, access_count = access_count + 1
                WHERE memory_id = ?
            """, (datetime.now(timezone.utc).isoformat(), memory_id))
            await conn.commit()
    
    async def _cleanup_loop(self):
        """Background cleanup task for expired and low-importance memories"""
        while not self._shutdown_event.is_set():
            try:
                cleanup_interval = self.config.background_cleanup_interval
                
                # Find memories to cleanup based on retention policies
                await self._cleanup_expired_memories()
                await self._cleanup_low_relevance_memories()
                
                self.metrics['retention_cleanups'] += 1
                logger.debug("Memory cleanup completed")
                
                await asyncio.sleep(cleanup_interval)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in memory cleanup loop: {e}")
                await asyncio.sleep(300)  # Wait before retrying
    
    async def _cleanup_expired_memories(self):
        """Remove expired memories based on retention policies"""
        current_time = datetime.now(timezone.utc)
        
        async with self._get_db_connection() as conn:
            # Find expired memories
            cursor = await conn.execute("""
                SELECT memory_id, user_id, importance, created_at, vector_id
                FROM memories
                WHERE expires_at IS NOT NULL AND expires_at < ?
                   OR (importance = ? AND created_at < ?)
            """, (
                current_time.isoformat(),
                MemoryImportance.TEMPORARY.value,
                (current_time - timedelta(days=1)).isoformat()
            ))
            
            expired_memories = cursor.fetchall()
            
            for memory in expired_memories:
                await self._delete_memory(memory['memory_id'], memory['user_id'], memory['vector_id'])
    
    async def _cleanup_low_relevance_memories(self):
        """Remove low-relevance memories when approaching storage limits"""
        # This would implement more sophisticated cleanup based on user limits
        # and relevance scoring - placeholder for now
        pass
    
    async def _delete_memory(self, memory_id: str, user_id: UserID, vector_id: Optional[str]):
        """Delete memory from all storage systems"""
        try:
            # Remove from vector database
            if vector_id:
                collection_name = f"user_{hashlib.sha256(user_id.encode()).hexdigest()[:16]}"
                collection = await self._get_or_create_collection(collection_name)
                collection.delete(ids=[vector_id])
            
            # Remove from metadata database
            async with self._get_db_connection() as conn:
                await conn.execute("DELETE FROM encrypted_content WHERE memory_id = ?", (memory_id,))
                await conn.execute("DELETE FROM memories WHERE memory_id = ?", (memory_id,))
                await conn.commit()
            
            logger.debug(f"Deleted memory {memory_id}")
            
        except Exception as e:
            logger.error(f"Failed to delete memory {memory_id}: {e}")
    
    async def get_user_memory_stats(self, user_id: UserID) -> Dict[str, Any]:
        """Get comprehensive memory statistics for user"""
        async with self._get_db_connection() as conn:
            # Total memories by type
            cursor = await conn.execute("""
                SELECT memory_type, importance, COUNT(*) as count
                FROM memories
                WHERE user_id = ?
                GROUP BY memory_type, importance
            """, (user_id,))
            type_stats = cursor.fetchall()
            
            # Recent activity
            cursor = await conn.execute("""
                SELECT COUNT(*) as total, 
                       MAX(last_accessed) as last_access,
                       SUM(access_count) as total_accesses
                FROM memories
                WHERE user_id = ?
            """, (user_id,))
            activity_stats = cursor.fetchone()
            
            return {
                'total_memories': activity_stats['total'] if activity_stats else 0,
                'last_access': activity_stats['last_access'] if activity_stats else None,
                'total_accesses': activity_stats['total_accesses'] if activity_stats else 0,
                'by_type': [dict(row) for row in type_stats],
                'storage_usage': 'TODO: Calculate storage usage'
            }
    
    async def export_user_memories(self, user_id: UserID) -> Dict[str, Any]:
        """Export all user memories for backup/migration"""
        memories = await self.retrieve_memories(
            user_id=user_id,
            limit=self.config.max_memories_per_user,
            include_content=True
        )
        
        return {
            'user_id': user_id,
            'export_timestamp': datetime.now(timezone.utc).isoformat(),
            'memory_count': len(memories),
            'memories': memories
        }
    
    async def shutdown(self):
        """Graceful shutdown of memory system"""
        logger.info("Shutting down memory system...")
        
        self._shutdown_event.set()
        
        if self.cleanup_task:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
        
        # Persist vector database
        if self.vector_db:
            try:
                self.vector_db.persist()
            except Exception as e:
                logger.error(f"Error persisting vector database: {e}")
        
        logger.info("Memory system shutdown complete")
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get memory system performance metrics"""
        return {
            **self.metrics,
            'vector_db_collections': len(self.vector_db.list_collections()) if self.vector_db else 0,
            'embedding_model': self.config.embedding_model,
            'encryption_enabled': self.config.encryption_enabled
        }