"""
MORPHEUS CHAT - GitHub Repository Integration & Deep Indexing
Production-grade repository cloning with automated ingestion pipeline.

Architecture Features:
- Secure repository cloning with validation
- Recursive file discovery and classification
- Automated ingestion pipeline integration
- Progress tracking and error recovery
- Metadata extraction and semantic indexing
"""

import asyncio
import hashlib
import logging
import os
import shutil
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Tuple, Set
from pathlib import Path
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse
import re
import random

import git
from git import Repo, GitCommandError
import aiofiles
from pydantic import BaseModel, Field, validator

try:
    from vm_memory_system.memory_core import SessionID, UserID
except Exception:
    try:
        from schemas.session import SessionID, UserID
    except Exception:
        from typing import Any as SessionID  # type: ignore
        from typing import Any as UserID  # type: ignore
from core.accelerated_file_processing import IntelligentFileProcessor, ProcessingPriority
from core.file_upload_system import FileUploadManager, FileType, ProcessingStatus
from core.memory_core import MemoryManager, MemoryType, MemoryImportance

logger = logging.getLogger(__name__)


class RepoStatus(str, Enum):
    """Repository processing status"""
    INITIALIZING = "initializing"
    CLONING = "cloning" 
    INDEXING = "indexing"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class RepoFileType(str, Enum):
    """Extended file classification for repositories"""
    SOURCE_CODE = "source_code"
    DOCUMENTATION = "documentation"
    CONFIGURATION = "configuration"
    DATA_FILE = "data_file"
    IMAGE_ASSET = "image_asset"
    BUILD_SCRIPT = "build_script"
    TEST_FILE = "test_file"
    LICENSE = "license"
    UNKNOWN = "unknown"


@dataclass
class RepoFile:
    """Repository file metadata with processing info"""
    relative_path: str
    absolute_path: Path
    file_type: RepoFileType
    size_bytes: int
    last_modified: datetime
    language: Optional[str] = None
    line_count: Optional[int] = None
    processed: bool = False
    processing_error: Optional[str] = None
    memory_id: Optional[UUID] = None


class RepoMetadata(BaseModel):
    """Comprehensive repository metadata"""
    repo_id: UUID = Field(default_factory=uuid4)
    url: str = Field(description="Repository URL")
    name: str = Field(description="Repository name")
    description: Optional[str] = None

    # Processing metadata
    user_id: str = Field(description="User who initiated clone")
    session_id: str = Field(description="Session context")
    status: RepoStatus = Field(default=RepoStatus.INITIALIZING)

    # Repository info
    default_branch: Optional[str] = None
    latest_commit: Optional[str] = None
    commit_count: Optional[int] = None
    contributor_count: Optional[int] = None

    # File statistics
    total_files: int = Field(default=0)
    processed_files: int = Field(default=0)
    failed_files: int = Field(default=0)

    # Processing progress
    progress_percentage: float = Field(default=0.0, ge=0, le=100)
    processing_start: Optional[datetime] = None
    processing_end: Optional[datetime] = None

    # Storage info
    local_path: Optional[str] = None
    total_size_bytes: int = Field(default=0)

    # Error tracking
    errors: List[str] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    last_updated: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def is_complete(self) -> bool:
        """Check if processing is complete"""
        return self.status == RepoStatus.COMPLETED

    @property
    def processing_time_seconds(self) -> Optional[float]:
        """Calculate processing duration"""
        if self.processing_start and self.processing_end:
            return (self.processing_end - self.processing_start).total_seconds()
        return None


class GitHubIntegrationManager:
    """
    Production-grade GitHub repository integration with deep indexing.
    
    Provides secure cloning, metadata extraction, and automated ingestion
    pipeline integration for comprehensive repository analysis.
    """
    
    def __init__(
        self,
        intelligent_processor: IntelligentFileProcessor,
        memory_manager: MemoryManager,
        clone_dir: str = "data/repositories",
        max_repo_size_gb: float = 5.0,
        timeout_minutes: int = 30,
        max_clone_retries: int = 3,
        backoff_base_seconds: float = 1.0
    ):
        self.intelligent_processor = intelligent_processor
        self.memory_manager = memory_manager
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_repo_size_bytes = int(max_repo_size_gb * 1024**3)
        self.timeout_seconds = timeout_minutes * 60
        self.max_clone_retries = max(1, int(max_clone_retries))
        self.backoff_base_seconds = max(0.1, float(backoff_base_seconds))
        
        # File type classification
        self.file_extensions = {
            # Source code
            RepoFileType.SOURCE_CODE: {
                '.py', '.js', '.ts', '.java', '.cpp', '.c', '.h', '.cs', '.go', 
                '.rs', '.php', '.rb', '.swift', '.kt', '.scala', '.r', '.m', '.sh'
            },
            # Documentation
            RepoFileType.DOCUMENTATION: {
                '.md', '.rst', '.txt', '.adoc', '.wiki', '.tex'
            },
            # Configuration
            RepoFileType.CONFIGURATION: {
                '.json', '.yaml', '.yml', '.toml', '.ini', '.cfg', '.conf',
                '.env', '.gitignore', '.dockerignore', 'dockerfile'
            },
            # Data files
            RepoFileType.DATA_FILE: {
                '.csv', '.tsv', '.xlsx', '.xls', '.sql', '.db', '.sqlite'
            },
            # Images
            RepoFileType.IMAGE_ASSET: {
                '.png', '.jpg', '.jpeg', '.gif', '.svg', '.webp', '.ico'
            },
            # Build/CI
            RepoFileType.BUILD_SCRIPT: {
                '.gradle', '.maven', '.make', '.cmake', '.bazel', '.BUILD'
            },
            # Tests
            RepoFileType.TEST_FILE: {
                '_test.py', '_test.js', '.test.js', '.spec.js', 'test_*.py'
            }
        }
        
        # Active repositories
        self.active_repos: Dict[UUID, RepoMetadata] = {}
        self.repo_files: Dict[UUID, List[RepoFile]] = {}
        
        logger.info(f"GitHub integration manager initialized: {clone_dir}")
    
    async def clone_repository(
        self,
        repo_url: str,
        user_id: str,
        session_id: str,
        branch: Optional[str] = None
    ) -> RepoMetadata:
        """
        Clone repository and initiate deep indexing pipeline.
        
        Returns repository metadata with processing status.
        """
        repo_id = uuid4()
        start_time = time.time()
        
        logger.info(f"Initiating repository clone: {repo_url}")
        
        metadata: Optional[RepoMetadata] = None
        try:
            # Validate URL
            if not self._validate_repo_url(repo_url):
                raise ValueError("Invalid or unsupported repository URL format")
            
            # Extract repository name
            repo_name = self._extract_repo_name(repo_url)
            
            # Create metadata
            metadata = RepoMetadata(
                repo_id=repo_id,
                url=repo_url,
                name=repo_name,
                user_id=user_id,
                session_id=session_id,
                processing_start=datetime.now(timezone.utc)
            )
            
            # Store in active repos
            self.active_repos[repo_id] = metadata
            
            # Create local directory
            local_path = self.clone_dir / f"{repo_id}_{repo_name}"
            metadata.local_path = str(local_path)
            
            # Update status
            metadata.status = RepoStatus.CLONING
            await self._update_progress(metadata, 10, "Starting clone operation")
            
            # Clone repository
            repo = await self._clone_repo(repo_url, local_path, branch)
            
            # Extract repository metadata
            await self._extract_repo_metadata(metadata, repo)
            await self._update_progress(metadata, 30, "Repository cloned successfully")
            
            # Index files
            metadata.status = RepoStatus.INDEXING
            await self._index_repository_files(metadata)
            await self._update_progress(metadata, 60, "File indexing completed")
            
            # Process files through upload system
            metadata.status = RepoStatus.PROCESSING
            await self._process_repository_files(metadata)
            await self._update_progress(metadata, 100, "Repository processing completed")
            
            # Mark as completed
            metadata.status = RepoStatus.COMPLETED
            metadata.processing_end = datetime.now(timezone.utc)
            
            logger.info(f"Repository clone completed: {repo_name} ({metadata.processing_time_seconds:.1f}s)")
            return metadata
            
        except Exception as e:
            logger.error(f"Repository clone failed: {e}")
            if metadata is None:
                # Ensure a well-formed metadata object is returned on early failure
                safe_name = self._extract_repo_name(repo_url)
                metadata = RepoMetadata(
                    repo_id=repo_id,
                    url=repo_url,
                    name=safe_name,
                    user_id=user_id,
                    session_id=session_id
                )
            metadata.status = RepoStatus.FAILED
            metadata.errors.append(str(e))
            metadata.processing_end = datetime.now(timezone.utc)
            
            # Cleanup on failure
            try:
                if metadata.local_path and Path(metadata.local_path).exists():
                    shutil.rmtree(metadata.local_path, ignore_errors=True)
            except Exception:
                logger.warning("Cleanup failed after clone failure")
            return metadata

    def _validate_repo_url(self, url: str) -> bool:
        """Validate repository URL format and accessibility"""
        try:
            # Accept SSH scp-like URLs (e.g., git@github.com:org/repo.git)
            scp_like = re.compile(r"^[\w\-\.]+@[\w\-\.]+:[\w\-/\.]+(\.git)?$")
            if scp_like.match(url):
                host = url.split("@", 1)[1].split(":", 1)[0].lower()
                # Allow common hosts and enterprise domains
                return bool(host)

            # Standard URL formats
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False

            allowed_schemes = {"http", "https", "ssh", "git"}
            if parsed.scheme.lower() not in allowed_schemes:
                return False

            # Basic path sanity: expect at least /owner/repo
            path_parts = [p for p in parsed.path.split("/") if p]
            if len(path_parts) < 2:
                return False

            return True
        except Exception:
            return False

    def _extract_repo_name(self, url: str) -> str:
        """Extract repository name from URL"""
        try:
            # Handle different URL formats
            if url.endswith('.git'):
                url = url[:-4]
            
            parts = url.rstrip('/').split('/')
            return parts[-1] if parts else 'unknown-repo'
            
        except Exception:
            return 'unknown-repo'
    
    async def _clone_repo(
        self, 
        url: str, 
        local_path: Path, 
        branch: Optional[str] = None
    ) -> Repo:
        """Clone repository with timeout, retries, and size limits"""
        # Ensure parent directory exists
        local_path.parent.mkdir(parents=True, exist_ok=True)

        # Disallow interactive prompts to avoid hangs
        env = os.environ.copy()
        env["GIT_TERMINAL_PROMPT"] = "0"
        env["GIT_ASKPASS"] = "echo"  # Prevent credential prompts

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_clone_retries + 1):
            try:
                if local_path.exists():
                    shutil.rmtree(local_path, ignore_errors=True)

                cmd = ['git', 'clone', '--no-tags', '--depth', '1']
                if branch:
                    cmd.extend(['-b', branch])
                cmd.extend([url, str(local_path)])

                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )

                try:
                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout_seconds
                    )
                except asyncio.TimeoutError:
                    process.kill()
                    raise TimeoutError(f"Repository clone timed out after {self.timeout_seconds}s")

                if process.returncode != 0:
                    err = stderr.decode(errors="ignore")
                    raise RuntimeError(f"Git clone failed (attempt {attempt}/{self.max_clone_retries}): {err.strip()}")

                # Validate repository structure
                if not (local_path.exists() and (local_path / ".git").exists()):
                    raise RuntimeError("Clone completed but .git directory not found")

                # Check repository size
                repo_size = 0
                for f in local_path.rglob('*'):
                    try:
                        if f.is_file():
                            repo_size += f.stat().st_size
                        # Early abort if exceeding size
                        if repo_size > self.max_repo_size_bytes:
                            raise ValueError("size_exceeded")
                    except FileNotFoundError:
                        # Handle ephemeral files disappearing during traversal
                        continue

                if repo_size > self.max_repo_size_bytes:
                    shutil.rmtree(local_path, ignore_errors=True)
                    raise ValueError(f"Repository too large: {repo_size / (1024**3):.1f}GB")

                # Load with GitPython for metadata
                return Repo(local_path)

            except Exception as e:
                last_error = e
                # Retry only for transient errors
                message = str(e).lower()
                is_transient = any(
                    kw in message
                    for kw in [
                        "timed out", "timeout", "temporarily unavailable",
                        "could not resolve", "connection reset", "connection refused",
                        "remote end hung up", "failed to connect", "http 5", "early eof"
                    ]
                )
                if attempt >= self.max_clone_retries or not is_transient:
                    # Do not retry non-transient failures
                    break

                # Exponential backoff with jitter
                sleep_for = self.backoff_base_seconds * (2 ** (attempt - 1))
                sleep_for += random.uniform(0, 0.25 * sleep_for)
                logger.warning(f"Clone failed (attempt {attempt}); retrying in {sleep_for:.2f}s")
                await asyncio.sleep(sleep_for)

        # Cleanup on failure and re-raise
        try:
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
        finally:
            if last_error:
                raise last_error
            raise RuntimeError("Unknown cloning error")

    async def _extract_repo_metadata(self, metadata: RepoMetadata, repo: Repo):
        """Extract repository metadata from Git repo"""
        try:
            # Determine default branch robustly, even in shallow/detached clones
            default_branch = None
            try:
                default_branch = repo.active_branch.name  # May fail in detached HEAD
            except Exception:
                try:
                    # Derive from origin/HEAD symbolic ref
                    head_ref = repo.git.symbolic_ref("refs/remotes/origin/HEAD")
                    # Format: refs/remotes/origin/main
                    default_branch = head_ref.rsplit("/", 1)[-1]
                except Exception:
                    # Fallback: use HEAD commit short SHA
                    default_branch = None

            metadata.default_branch = default_branch
            # Latest commit short SHA
            try:
                metadata.latest_commit = repo.head.commit.hexsha[:8]
            except Exception:
                metadata.latest_commit = None

            # Count commits (bounded for performance)
            try:
                max_count = 1000
                commit_iter = repo.iter_commits(max_count=max_count)
                cnt = 0
                contributors: Set[str] = set()
                for c in commit_iter:
                    cnt += 1
                    if c.author and c.author.email:
                        contributors.add(c.author.email.lower())
                metadata.commit_count = cnt
                # Estimate unique contributor count from sampled commits
                metadata.contributor_count = len(contributors) if contributors else None
            except Exception:
                metadata.commit_count = None
                metadata.contributor_count = None

            # Pull description from README (bounded read)
            readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
            repo_path = Path(repo.working_dir)
            for readme in readme_files:
                readme_path = repo_path / readme
                if readme_path.exists():
                    try:
                        async with aiofiles.open(readme_path, 'r', encoding='utf-8', errors='ignore') as f:
                            # Only read up to 64KB
                            content = await f.read(64 * 1024)
                            lines = content.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line and not line.startswith('#'):
                                    metadata.description = line[:200]
                                    break
                    except Exception:
                        pass
                    break

        except Exception as e:
            logger.warning(f"Failed to extract repository metadata: {e}")

    async def _index_repository_files(self, metadata: RepoMetadata):
        """Index all files in repository with classification"""
        repo_path = Path(metadata.local_path)
        files = []
        total_size = 0
        
        # Skip common directories
        skip_dirs = {
            '.git', '.svn', '.hg', '__pycache__', '.pytest_cache',
            'node_modules', '.venv', 'venv', '.env', 'dist', 'build',
            '.next', '.nuxt', 'target', 'bin', 'obj', '.cache', 'vendor'
        }
        
        try:
            for file_path in repo_path.rglob('*'):
                if file_path.is_file():
                    # Skip files in excluded directories
                    if any(skip_dir in file_path.parts for skip_dir in skip_dirs):
                        continue
                    
                    try:
                        stat = file_path.stat()
                        relative_path = file_path.relative_to(repo_path)
                        
                        # Classify file type
                        file_type = self._classify_file(file_path)
                        
                        # Extract basic metadata
                        repo_file = RepoFile(
                            relative_path=str(relative_path),
                            absolute_path=file_path,
                            file_type=file_type,
                            size_bytes=stat.st_size,
                            last_modified=datetime.fromtimestamp(stat.st_mtime, timezone.utc)
                        )
                        
                        # Add language detection for source files
                        if file_type == RepoFileType.SOURCE_CODE:
                            repo_file.language = self._detect_language(file_path)
                        
                        # Count lines for text files
                        if file_type in [RepoFileType.SOURCE_CODE, RepoFileType.DOCUMENTATION]:
                            repo_file.line_count = await self._count_lines(file_path)
                        
                        files.append(repo_file)
                        total_size += stat.st_size
                        
                    except Exception as e:
                        logger.warning(f"Failed to index file {file_path}: {e}")
                        continue
            
            # Update metadata
            metadata.total_files = len(files)
            metadata.total_size_bytes = total_size
            
            # Store file list
            self.repo_files[metadata.repo_id] = files
            
            logger.info(f"Indexed {len(files)} files totaling {total_size / (1024**2):.1f}MB")
            
        except Exception as e:
            logger.error(f"File indexing failed: {e}")
            metadata.errors.append(f"Indexing error: {str(e)}")
    
    def _classify_file(self, file_path: Path) -> RepoFileType:
        """Classify file type based on extension and name patterns"""
        file_ext = file_path.suffix.lower()
        file_name = file_path.name.lower()
        
        # Check special files first
        if file_name in ['license', 'licence', 'copying']:
            return RepoFileType.LICENSE
        
        if 'test' in file_name or file_name.startswith('test_'):
            return RepoFileType.TEST_FILE
        
        if file_name.lower() in ['dockerfile', 'makefile', 'cmakelists.txt']:
            return RepoFileType.BUILD_SCRIPT
        
        # Check by extension
        for file_type, extensions in self.file_extensions.items():
            if file_ext in extensions:
                return file_type
        
        return RepoFileType.UNKNOWN
    
    def _detect_language(self, file_path: Path) -> str:
        """Detect programming language from file extension"""
        ext_to_lang = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.java': 'Java',
            '.cpp': 'C++',
            '.c': 'C',
            '.h': 'C/C++',
            '.cs': 'C#',
            '.go': 'Go',
            '.rs': 'Rust',
            '.php': 'PHP',
            '.rb': 'Ruby',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
            '.scala': 'Scala',
            '.r': 'R',
            '.m': 'Objective-C',
            '.sh': 'Shell'
        }
        
        return ext_to_lang.get(file_path.suffix.lower(), 'Unknown')
    
    async def _count_lines(self, file_path: Path) -> Optional[int]:
        """Count lines in text file"""
        try:
            # Limit file size for line counting
            if file_path.stat().st_size > 1024 * 1024:  # 1MB limit
                return None

            line_count = 0
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                async for _ in f:
                    line_count += 1
            return line_count
        except Exception:
            return None
    
    async def _process_repository_files(self, metadata: RepoMetadata):
        """Process repository files through upload system"""
        if metadata.repo_id not in self.repo_files:
            raise ValueError("Repository files not indexed")
        
        files_to_process = []
        for repo_file in self.repo_files[metadata.repo_id]:
            try:
                # Skip large files
                if repo_file.size_bytes > 10 * 1024 * 1024:  # 10MB limit
                    repo_file.processing_error = "File too large for processing"
                    continue

                # Read file content
                async with aiofiles.open(repo_file.absolute_path, 'rb') as f:
                    file_data = await f.read()
                
                files_to_process.append((file_data, repo_file.relative_path))

            except Exception as e:
                logger.warning(f"Failed to read file {repo_file.relative_path} for processing: {e}")
                repo_file.processing_error = str(e)

        # Queue the entire batch for processing
        logger.info(f"Queueing {len(files_to_process)} files for accelerated processing.")
        task_ids = await self.intelligent_processor.queue_batch(
            file_batch=files_to_process,
            user_id=metadata.user_id,
            session_id=metadata.session_id,
            priority=ProcessingPriority.BACKGROUND,
            source_context=f"git_clone:{metadata.name}"
        )

        # Optionally, you can wait for completion and update status,
        # or this can be handled asynchronously by another part of the system.
        # For this integration, we'll assume it's fire-and-forget for now.
        metadata.processed_files = len(task_ids)
        
        logger.info(f"Successfully queued {len(task_ids)} files for processing.")

    async def _store_repository_memory(
        self,
        metadata: RepoMetadata,
        repo_file: RepoFile,
        content: str
    ) -> UUID:
        """Store repository file in memory system with context"""
        
        # Create contextual content
        context_content = f"""Repository: {metadata.name}
File: {repo_file.relative_path}
Type: {repo_file.file_type.value}
Language: {repo_file.language or 'Unknown'}
Size: {repo_file.size_bytes} bytes
Lines: {repo_file.line_count or 'Unknown'}

Content:
{content}"""
        
        # Determine memory importance based on file type
        importance_map = {
            RepoFileType.DOCUMENTATION: MemoryImportance.HIGH,
            RepoFileType.SOURCE_CODE: MemoryImportance.MEDIUM,
            RepoFileType.CONFIGURATION: MemoryImportance.MEDIUM,
            RepoFileType.DATA_FILE: MemoryImportance.LOW,
            RepoFileType.LICENSE: MemoryImportance.LOW
        }
        
        importance = importance_map.get(repo_file.file_type, MemoryImportance.LOW)
        
        # Store in memory
        memory_id = await self.memory_manager.store_memory(
            user_id=metadata.user_id,
            content=context_content,
            memory_type=MemoryType.DOCUMENT,
            importance=importance,
            source_session=metadata.session_id,
            tags=[
                'repository',
                metadata.name,
                repo_file.file_type.value,
                repo_file.language or 'unknown'
            ],
            metadata={
                'repo_id': str(metadata.repo_id),
                'repo_url': metadata.url,
                'file_path': repo_file.relative_path,
                'file_type': repo_file.file_type.value,
                'language': repo_file.language,
                'size_bytes': repo_file.size_bytes,
                'line_count': repo_file.line_count
            }
        )
        
        return memory_id
    
    async def _update_progress(self, metadata: RepoMetadata, progress: float, message: str):
        """Update processing progress with logging and state persistence."""
        if metadata is None:
            logger.warning("Attempted to update progress with None metadata")
            return

        # Clamp progress to the valid range [0, 100]
        clamped_progress = max(0.0, min(100.0, progress))
        metadata.progress_percentage = clamped_progress
        metadata.last_updated = datetime.now(timezone.utc)

        logger.info(f"Repository {metadata.name}: {clamped_progress:.1f}% - {message}")

    async def get_repository_status(self, repo_id: UUID) -> Optional[RepoMetadata]:
        """Retrieve the current processing metadata for a repository."""
        metadata = self.active_repos.get(repo_id)
        if metadata is None:
            logger.debug(f"Requested status for unknown repository ID: {repo_id}")
        return metadata

    async def list_user_repositories(self, user_id: str) -> List[RepoMetadata]:
        """Return all repository metadata objects owned by the specified user."""
        user_repos = [
            repo for repo in self.active_repos.values()
            if repo.user_id == user_id
        ]
        logger.debug(f"Listing {len(user_repos)} repositories for user {user_id}")
        return user_repos
    
    async def search_repository_content(
        self,
        user_id: str,
        query: str,
        repo_id: Optional[UUID] = None
    ) -> List[Dict[str, Any]]:
        """Search repository content through memory system"""
        
        # Build search tags
        search_tags = ['repository']
        if repo_id:
            search_tags.append(str(repo_id))
        
        # Search through memory system
        memories = await self.memory_manager.retrieve_memories(
            user_id=user_id,
            query=query,
            memory_types=[MemoryType.DOCUMENT],
            limit=20
        )
        
        # Filter repository memories
        repo_memories = []
        for memory in memories:
            tags = memory.get('tags', [])
            if 'repository' not in tags:
                continue
            metadata_dict = memory.get('metadata', {})
            if repo_id and metadata_dict.get('repo_id') != str(repo_id):
                continue
            repo_memories.append({
                'memory_id': memory['memory_id'],
                'repo_id': metadata_dict.get('repo_id'),
                'repo_url': metadata_dict.get('repo_url'),
                'file_path': metadata_dict.get('file_path'),
                'file_type': metadata_dict.get('file_type'),
                'language': metadata_dict.get('language'),
                'similarity_score': memory.get('similarity_score', 0),
                'content_preview': memory['content'][:500] + '...' if len(memory['content']) > 500 else memory['content']
            })
        
        return repo_memories
    
    async def cleanup_repository(self, repo_id: UUID, user_id: str) -> bool:
        """Clean up repository files and data"""
        try:
            metadata = self.active_repos.get(repo_id)
            if not metadata or metadata.user_id != user_id:
                return False
            
            # Remove local files
            if metadata.local_path and Path(metadata.local_path).exists():
                shutil.rmtree(metadata.local_path, ignore_errors=True)
            
            # Remove from tracking
            self.active_repos.pop(repo_id, None)
            self.repo_files.pop(repo_id, None)
            
            logger.info(f"Repository {metadata.name} cleaned up successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to cleanup repository {repo_id}: {e}")
            return False
    
    def get_system_stats(self) -> Dict[str, Any]:
        """Get system statistics"""
        total_repos = len(self.active_repos)
        completed_repos = sum(1 for r in self.active_repos.values() if r.is_complete)
        failed_repos = sum(1 for r in self.active_repos.values() if r.status == RepoStatus.FAILED)
        active_processing = sum(
            1 for r in self.active_repos.values()
            if r.status in {RepoStatus.INITIALIZING, RepoStatus.CLONING, RepoStatus.INDEXING, RepoStatus.PROCESSING}
        )
        total_files = sum(r.total_files for r in self.active_repos.values())
        total_size = sum(r.total_size_bytes for r in self.active_repos.values())
        
        return {
            'total_repositories': total_repos,
            'completed_repositories': completed_repos,
            'failed_repositories': failed_repos,
            'total_files_indexed': total_files,
            'total_size_gb': total_size / (1024**3),
            'active_processing': active_processing,
            'clone_directory': str(self.clone_dir)
        }

