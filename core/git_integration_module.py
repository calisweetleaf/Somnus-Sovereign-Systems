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

import git
from git import Repo, GitCommandError
import aiofiles
from pydantic import BaseModel, Field, validator

from schemas.session import SessionID, UserID
from file_upload_system import FileUploadManager, FileType, ProcessingStatus
from memory_core import MemoryManager, MemoryType, MemoryImportance

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
    user_id: UserID = Field(description="User who initiated clone")
    session_id: SessionID = Field(description="Session context")
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
        file_manager: FileUploadManager,
        memory_manager: MemoryManager,
        clone_dir: str = "data/repositories",
        max_repo_size_gb: float = 5.0,
        timeout_minutes: int = 30
    ):
        self.file_manager = file_manager
        self.memory_manager = memory_manager
        self.clone_dir = Path(clone_dir)
        self.clone_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_repo_size_bytes = int(max_repo_size_gb * 1024**3)
        self.timeout_seconds = timeout_minutes * 60
        
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
        user_id: UserID,
        session_id: SessionID,
        branch: Optional[str] = None
    ) -> RepoMetadata:
        """
        Clone repository and initiate deep indexing pipeline.
        
        Returns repository metadata with processing status.
        """
        repo_id = uuid4()
        start_time = time.time()
        
        logger.info(f"Initiating repository clone: {repo_url}")
        
        try:
            # Validate URL
            if not self._validate_repo_url(repo_url):
                raise ValueError("Invalid repository URL")
            
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
            metadata.status = RepoStatus.FAILED
            metadata.errors.append(str(e))
            metadata.processing_end = datetime.now(timezone.utc)
            
            # Cleanup on failure
            if metadata.local_path and Path(metadata.local_path).exists():
                shutil.rmtree(metadata.local_path, ignore_errors=True)
            
            return metadata
    
    def _validate_repo_url(self, url: str) -> bool:
        """Validate repository URL format and accessibility"""
        try:
            parsed = urlparse(url)
            
            # Check basic URL structure
            if not parsed.scheme or not parsed.netloc:
                return False
            
            # Support GitHub, GitLab, Bitbucket
            allowed_hosts = {
                'github.com', 'gitlab.com', 'bitbucket.org',
                'raw.githubusercontent.com'
            }
            
            if parsed.netloc.lower() not in allowed_hosts:
                logger.warning(f"Repository host not in allowed list: {parsed.netloc}")
                # Allow but warn - could be enterprise instance
            
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
        """Clone repository with timeout and size limits"""
        try:
            # Use asyncio subprocess for timeout control
            cmd = ['git', 'clone']
            if branch:
                cmd.extend(['-b', branch])
            cmd.extend(['--depth', '1'])  # Shallow clone for speed
            cmd.extend([url, str(local_path)])
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=self.timeout_seconds
                )
                
                if process.returncode != 0:
                    raise RuntimeError(f"Git clone failed: {stderr.decode()}")
                
            except asyncio.TimeoutError:
                process.kill()
                raise TimeoutError(f"Repository clone timed out after {self.timeout_seconds}s")
            
            # Check repository size
            repo_size = sum(
                f.stat().st_size for f in local_path.rglob('*') 
                if f.is_file()
            )
            
            if repo_size > self.max_repo_size_bytes:
                shutil.rmtree(local_path)
                raise ValueError(f"Repository too large: {repo_size / (1024**3):.1f}GB")
            
            # Load with GitPython for metadata
            return Repo(local_path)
            
        except Exception as e:
            if local_path.exists():
                shutil.rmtree(local_path, ignore_errors=True)
            raise
    
    async def _extract_repo_metadata(self, metadata: RepoMetadata, repo: Repo):
        """Extract repository metadata from Git repo"""
        try:
            # Basic repository info
            metadata.default_branch = repo.active_branch.name
            metadata.latest_commit = repo.head.commit.hexsha[:8]
            
            # Count commits (limited for performance)
            try:
                commit_count = sum(1 for _ in repo.iter_commits(max_count=1000))
                metadata.commit_count = min(commit_count, 1000)
            except Exception:
                metadata.commit_count = None
            
            # Get description from README if available
            readme_files = ['README.md', 'README.rst', 'README.txt', 'README']
            repo_path = Path(repo.working_dir)
            
            for readme in readme_files:
                readme_path = repo_path / readme
                if readme_path.exists():
                    try:
                        async with aiofiles.open(readme_path, 'r', encoding='utf-8') as f:
                            content = await f.read()
                            # Extract first paragraph as description
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
            '.next', '.nuxt', 'target', 'bin', 'obj'
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
            
            async with aiofiles.open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = await f.read()
                return len(content.splitlines())
                
        except Exception:
            return None
    
    async def _process_repository_files(self, metadata: RepoMetadata):
        """Process repository files through upload system"""
        if metadata.repo_id not in self.repo_files:
            raise ValueError("Repository files not indexed")
        
        files = self.repo_files[metadata.repo_id]
        processed_count = 0
        failed_count = 0
        
        # Prioritize file processing
        priority_order = [
            RepoFileType.DOCUMENTATION,
            RepoFileType.SOURCE_CODE,
            RepoFileType.CONFIGURATION,
            RepoFileType.DATA_FILE,
            RepoFileType.IMAGE_ASSET
        ]
        
        # Sort files by priority
        sorted_files = []
        for file_type in priority_order:
            sorted_files.extend([f for f in files if f.file_type == file_type])
        
        # Add remaining files
        processed_types = set(priority_order)
        sorted_files.extend([f for f in files if f.file_type not in processed_types])
        
        # Process files with progress updates
        for i, repo_file in enumerate(sorted_files):
            try:
                # Skip large files
                if repo_file.size_bytes > 10 * 1024 * 1024:  # 10MB limit
                    repo_file.processing_error = "File too large for processing"
                    failed_count += 1
                    continue
                
                # Read file content
                async with aiofiles.open(repo_file.absolute_path, 'rb') as f:
                    file_data = await f.read()
                
                # Process through file upload system
                file_metadata = await self.file_manager.upload_file(
                    file_data=file_data,
                    filename=repo_file.relative_path,
                    user_id=metadata.user_id,
                    session_id=metadata.session_id
                )
                
                if file_metadata.processing_status == ProcessingStatus.COMPLETED:
                    repo_file.processed = True
                    processed_count += 1
                    
                    # Store repository context memory
                    if file_metadata.extracted_text:
                        memory_id = await self._store_repository_memory(
                            metadata, repo_file, file_metadata.extracted_text
                        )
                        repo_file.memory_id = memory_id
                else:
                    repo_file.processing_error = file_metadata.processing_error
                    failed_count += 1
                
            except Exception as e:
                logger.warning(f"Failed to process file {repo_file.relative_path}: {e}")
                repo_file.processing_error = str(e)
                failed_count += 1
            
            # Update progress
            progress = ((i + 1) / len(sorted_files)) * 40 + 60  # 60-100% range
            await self._update_progress(
                metadata, 
                progress, 
                f"Processed {processed_count}/{len(sorted_files)} files"
            )
        
        # Update final counts
        metadata.processed_files = processed_count
        metadata.failed_files = failed_count
        
        logger.info(f"Repository processing completed: {processed_count} processed, {failed_count} failed")
    
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
        """Update processing progress with logging"""
        metadata.progress_percentage = min(100.0, max(0.0, progress))
        metadata.last_updated = datetime.now(timezone.utc)
        
        logger.info(f"Repository {metadata.name}: {progress:.1f}% - {message}")
    
    async def get_repository_status(self, repo_id: UUID) -> Optional[RepoMetadata]:
        """Get repository processing status"""
        return self.active_repos.get(repo_id)
    
    async def list_user_repositories(self, user_id: UserID) -> List[RepoMetadata]:
        """List repositories for user"""
        return [
            repo for repo in self.active_repos.values()
            if repo.user_id == user_id
        ]
    
    async def search_repository_content(
        self,
        user_id: UserID,
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
            if 'repository' in memory.get('tags', []):
                # Add repository context
                metadata_dict = memory.get('metadata', {})
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
    
    async def cleanup_repository(self, repo_id: UUID, user_id: UserID) -> bool:
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
        total_files = sum(r.total_files for r in self.active_repos.values())
        total_size = sum(r.total_size_bytes for r in self.active_repos.values())
        
        return {
            'total_repositories': total_repos,
            'completed_repositories': completed_repos,
            'total_files_indexed': total_files,
            'total_size_gb': total_size / (1024**3),
            'active_processing': total_repos - completed_repos,
            'clone_directory': str(self.clone_dir)
        }
