"""
MORPHEUS CHAT - Project File Management System
Comprehensive file management for project environments

Revolutionary Features:
- Unlimited file storage (no arbitrary limits like SaaS services)
- Intelligent file organization and categorization
- Automatic file processing and analysis
- Version control and file history
- Real-time file monitoring and updates
- Integration with project knowledge and intelligence
- Collaborative file editing and sharing
- Advanced file search and filtering
"""

import asyncio
import hashlib
import json
import logging
import mimetypes
import shutil
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple, Union
from uuid import UUID, uuid4
from dataclasses import dataclass, field
from enum import Enum

import aiofiles
import magic

# Optional dependencies for advanced processing
try:
    import pytesseract
    from PIL import Image
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False

try:
    from pypdf import PdfReader
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

try:
    import chardet
    CHARDET_AVAILABLE = True
except ImportError:
    CHARDET_AVAILABLE = False

logger = logging.getLogger(__name__)


class FileProcessingStatus(str, Enum):
    """File processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class FileCategory(str, Enum):
    """File categories for organization"""
    DOCUMENT = "document"
    CODE = "code"
    DATA = "data"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    ARCHIVE = "archive"
    RESEARCH = "research"
    CONFIGURATION = "configuration"
    EXECUTABLE = "executable"
    UNKNOWN = "unknown"


@dataclass
class ProjectFile:
    """Comprehensive project file metadata"""
    file_id: UUID = field(default_factory=uuid4)
    filename: str = ""
    original_filename: str = ""
    file_path: str = ""  # Path within project VM
    
    # File properties
    size_bytes: int = 0
    mime_type: str = ""
    file_hash: str = ""
    encoding: Optional[str] = None
    
    # Classification
    category: FileCategory = FileCategory.UNKNOWN
    subcategory: Optional[str] = None
    tags: Set[str] = field(default_factory=set)
    
    # Processing
    processing_status: FileProcessingStatus = FileProcessingStatus.PENDING
    processing_error: Optional[str] = None
    analysis_completed: bool = False
    knowledge_extracted: bool = False
    
    # Content analysis
    content_preview: str = ""
    key_concepts: List[str] = field(default_factory=list)
    language_detected: Optional[str] = None
    
    # Relationships
    related_files: Set[UUID] = field(default_factory=set)
    derived_from: Optional[UUID] = None
    versions: List[UUID] = field(default_factory=list)
    
    # Collaboration
    uploaded_by: str = ""
    last_modified_by: str = ""
    collaborators: Set[str] = field(default_factory=set)
    
    # Integration
    knowledge_items: List[str] = field(default_factory=list)
    artifacts_created: List[str] = field(default_factory=list)
    automation_triggered: List[str] = field(default_factory=list)
    
    # Timestamps
    uploaded_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    processed_at: Optional[datetime] = None


@dataclass
class FileSearchResult:
    """File search result with relevance scoring"""
    file: ProjectFile
    relevance_score: float
    match_reasons: List[str]
    content_matches: List[str]


class ProjectFileManager:
    """
    Comprehensive file management system for projects
    
    Features:
    - Unlimited file storage within project VMs
    - Intelligent file organization and categorization
    - Automatic content analysis and knowledge extraction
    - Version control and file history tracking
    - Real-time file monitoring and processing
    - Advanced search and filtering capabilities
    - Integration with project intelligence and knowledge systems
    """
    
    def __init__(self, project_id: str, project_vm, project_intelligence=None, project_knowledge=None):
        self.project_id = project_id
        self.project_vm = project_vm
        self.project_intelligence = project_intelligence
        self.project_knowledge = project_knowledge
        
        # File tracking
        self.project_files: Dict[UUID, ProjectFile] = {}
        self.files_by_path: Dict[str, UUID] = {}
        self.files_by_category: Dict[FileCategory, Set[UUID]] = {
            category: set() for category in FileCategory
        }
        
        # Processing queues
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.processing_active: Dict[UUID, bool] = {}
        
        # File monitoring
        self.file_monitor_running = False
        self.file_monitor_task: Optional[asyncio.Task] = None
        
        # Configuration
        self.max_file_size = 1024 * 1024 * 1024  # 1GB default limit per file
        self.supported_text_extensions = {
            '.txt', '.md', '.rst', '.py', '.js', '.html', '.css', '.json', 
            '.xml', '.yaml', '.yml', '.csv', '.log', '.sql', '.sh', '.bat',
            '.c', '.cpp', '.h', '.hpp', '.java', '.php', '.rb', '.go', '.rs'
        }
        
        # Performance metrics
        self.metrics = {
            'total_files': 0,
            'total_size_bytes': 0,
            'files_processed': 0,
            'processing_failures': 0,
            'average_processing_time': 0.0
        }
        
        logger.info(f"File manager initialized for project {project_id}")
    
    async def initialize(self):
        """Initialize file management system"""
        
        try:
            # Setup file directories in VM
            await self._setup_file_directories()
            
            # Initialize file monitoring
            await self._initialize_file_monitoring()
            
            # Load existing files
            await self._scan_existing_files()
            
            # Start processing system
            await self._start_file_processing()
            
            logger.info(f"File management system ready for project {self.project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize file manager: {e}")
            raise
    
    async def _setup_file_directories(self):
        """Setup file directory structure in project VM"""
        
        directory_commands = [
            # Create comprehensive file directory structure
            "mkdir -p /project/files/{incoming,processed,archives,versions}",
            "mkdir -p /project/files/categories/{document,code,data,image,audio,video,research,configuration}",
            "mkdir -p /project/files/{temporary,trash,exports}",
            
            # Create file processing workspace
            "mkdir -p /project/file_processing/{queue,active,completed,failed,logs}",
            
            # Create file metadata storage
            "mkdir -p /project/file_metadata/{index,relationships,versions,analysis}",
            
            # Set permissions
            "chown -R morpheus:morpheus /project/files /project/file_processing /project/file_metadata",
            "chmod -R 755 /project/files /project/file_processing /project/file_metadata"
        ]
        
        for cmd in directory_commands:
            await self.project_vm.execute_command(cmd)
        
        # Create file management toolkit
        toolkit_script = '''#!/usr/bin/env python3
"""
Project File Management Toolkit
Provides file operations within project VM
"""

import json
import os
import hashlib
import mimetypes
from pathlib import Path
from datetime import datetime

class FileToolkit:
    def __init__(self):
        self.project_root = Path("/project")
        self.files_root = self.project_root / "files"
        self.metadata_root = self.project_root / "file_metadata"
        
    def calculate_file_hash(self, file_path):
        """Calculate SHA256 hash of file"""
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)
        return hasher.hexdigest()
    
    def get_file_info(self, file_path):
        """Get comprehensive file information"""
        path = Path(file_path)
        
        if not path.exists():
            return None
        
        stat = path.stat()
        mime_type, _ = mimetypes.guess_type(str(path))
        
        return {
            "filename": path.name,
            "size_bytes": stat.st_size,
            "mime_type": mime_type or "unknown",
            "modified_time": stat.st_mtime,
            "created_time": stat.st_ctime,
            "file_hash": self.calculate_file_hash(path)
        }
    
    def create_file_index(self, file_info, file_id):
        """Create file index entry"""
        index_file = self.metadata_root / "index" / f"{file_id}.json"
        index_file.parent.mkdir(exist_ok=True)
        
        with open(index_file, 'w') as f:
            json.dump(file_info, f, indent=2)
    
    def scan_directory(self, directory):
        """Scan directory for files"""
        dir_path = Path(directory)
        files = []
        
        if dir_path.exists():
            for file_path in dir_path.rglob("*"):
                if file_path.is_file():
                    file_info = self.get_file_info(file_path)
                    if file_info:
                        files.append({
                            "path": str(file_path),
                            "relative_path": str(file_path.relative_to(dir_path)),
                            **file_info
                        })
        
        return files

# Global toolkit instance
file_toolkit = FileToolkit()

# Command line interface for VM operations
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python toolkit.py <command> [args...]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "scan":
        directory = sys.argv[2] if len(sys.argv) > 2 else "/project/files"
        files = file_toolkit.scan_directory(directory)
        print(json.dumps(files, indent=2))
    
    elif command == "info":
        file_path = sys.argv[2]
        info = file_toolkit.get_file_info(file_path)
        print(json.dumps(info, indent=2))
    
    elif command == "hash":
        file_path = sys.argv[2]
        hash_value = file_toolkit.calculate_file_hash(file_path)
        print(hash_value)
    
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
'''
        
        await self.project_vm.write_file("/project/file_processing/toolkit.py", toolkit_script)
        await self.project_vm.execute_command("chmod +x /project/file_processing/toolkit.py")
    
    async def upload_file(
        self,
        file_data: bytes,
        filename: str,
        uploaded_by: str,
        target_category: Optional[FileCategory] = None,
        tags: Optional[Set[str]] = None
    ) -> ProjectFile:
        """Upload file to project with comprehensive processing"""
        
        # Validate file
        if len(file_data) > self.max_file_size:
            raise ValueError(f"File too large: {len(file_data)} bytes > {self.max_file_size}")
        
        # Calculate file hash
        file_hash = hashlib.sha256(file_data).hexdigest()
        
        # Check for duplicates
        existing_file = await self._find_file_by_hash(file_hash)
        if existing_file:
            logger.info(f"File {filename} already exists (duplicate)")
            return existing_file
        
        # Create file metadata
        file_obj = ProjectFile(
            filename=self._sanitize_filename(filename),
            original_filename=filename,
            size_bytes=len(file_data),
            file_hash=file_hash,
            uploaded_by=uploaded_by,
            tags=tags or set()
        )
        
        try:
            # Detect MIME type
            file_obj.mime_type = magic.from_buffer(file_data, mime=True)
            
            # Categorize file
            file_obj.category = target_category or self._categorize_file(filename, file_obj.mime_type)
            
            # Determine storage path
            category_dir = f"/project/files/categories/{file_obj.category.value}"
            file_obj.file_path = f"{category_dir}/{file_obj.filename}"
            
            # Save file to VM
            await self.project_vm.execute_command(f"mkdir -p {category_dir}")
            
            # Write file data to VM
            await self._write_file_to_vm(file_obj.file_path, file_data)
            
            # Extract content preview
            if self._is_text_file(file_obj):
                file_obj.content_preview = await self._extract_text_preview(file_obj)
                file_obj.encoding = await self._detect_encoding(file_data)
            
            # Register file
            self.project_files[file_obj.file_id] = file_obj
            self.files_by_path[file_obj.file_path] = file_obj.file_id
            self.files_by_category[file_obj.category].add(file_obj.file_id)
            
            # Update metrics
            self.metrics['total_files'] += 1
            self.metrics['total_size_bytes'] += file_obj.size_bytes
            
            # Queue for processing
            await self.processing_queue.put(file_obj.file_id)
            
            # Save file metadata
            await self._save_file_metadata(file_obj)
            
            logger.info(f"File uploaded: {filename} ({file_obj.size_bytes} bytes)")
            return file_obj
            
        except Exception as e:
            logger.error(f"Failed to upload file {filename}: {e}")
            raise
    
    async def _write_file_to_vm(self, vm_path: str, file_data: bytes):
        """Write file data to VM"""
        
        # Write to temporary file first
        temp_file = f"/tmp/upload_{uuid4().hex}"
        
        try:
            # Write to local temp file
            with open(temp_file, 'wb') as f:
                f.write(file_data)
            
            # Copy to VM using the VM's file writing method
            await self.project_vm.write_file(vm_path, file_data)
            
        except Exception as e:
            logger.error(f"Failed to write file to VM: {e}")
            raise
        finally:
            # Cleanup temp file
            if Path(temp_file).exists():
                Path(temp_file).unlink()
    
    async def process_file(self, file_id: UUID) -> bool:
        """Process file with comprehensive analysis"""
        
        if file_id not in self.project_files:
            return False
        
        file_obj = self.project_files[file_id]
        
        if file_obj.processing_status == FileProcessingStatus.PROCESSING:
            return False
        
        try:
            file_obj.processing_status = FileProcessingStatus.PROCESSING
            self.processing_active[file_id] = True
            processing_start = datetime.now(timezone.utc)
            
            # Content analysis
            await self._analyze_file_content(file_obj)
            
            # Extract knowledge
            if self.project_knowledge:
                await self._extract_file_knowledge(file_obj)
            
            # Detect relationships
            await self._detect_file_relationships(file_obj)
            
            # Language detection for text files
            if self._is_text_file(file_obj):
                file_obj.language_detected = await self._detect_language(file_obj)
            
            # Integration with project intelligence
            if self.project_intelligence:
                await self._integrate_with_intelligence(file_obj)
            
            # Mark as completed
            file_obj.processing_status = FileProcessingStatus.COMPLETED
            file_obj.processed_at = datetime.now(timezone.utc)
            file_obj.analysis_completed = True
            
            # Update metrics
            processing_time = (file_obj.processed_at - processing_start).total_seconds()
            self.metrics['files_processed'] += 1
            self.metrics['average_processing_time'] = (
                (self.metrics['average_processing_time'] * (self.metrics['files_processed'] - 1) + processing_time) /
                self.metrics['files_processed']
            )
            
            # Save updated metadata
            await self._save_file_metadata(file_obj)
            
            logger.info(f"File processed successfully: {file_obj.filename}")
            return True
            
        except Exception as e:
            file_obj.processing_status = FileProcessingStatus.FAILED
            file_obj.processing_error = str(e)
            self.metrics['processing_failures'] += 1
            
            logger.error(f"File processing failed for {file_obj.filename}: {e}")
            return False
            
        finally:
            self.processing_active.pop(file_id, None)
    
    async def _analyze_file_content(self, file_obj: ProjectFile):
        """Analyze file content for insights"""
        
        if self._is_text_file(file_obj):
            # Get file content
            content = await self._read_file_content(file_obj)
            
            if content:
                # Extract key concepts using simple analysis
                file_obj.key_concepts = await self._extract_key_concepts(content)
                
                # Update content preview
                file_obj.content_preview = content[:1000]  # First 1000 characters
        
        elif self._is_image_file(file_obj):
            # Image analysis
            await self._analyze_image_content(file_obj)
        
        elif self._is_code_file(file_obj):
            # Code analysis
            content = await self._read_file_content(file_obj)
            if content:
                file_obj.key_concepts = await self._analyze_code_content(content)
        
        elif file_obj.category == FileCategory.DOCUMENT:
            # Document analysis (PDF, DOCX, etc.)
            await self._analyze_document_content(file_obj)
    
    async def _analyze_image_content(self, file_obj: ProjectFile):
        """Analyze image content using OCR and metadata"""
        
        try:
            # Basic image metadata
            file_obj.key_concepts = ["image", "visual_content"]
            
            # OCR text extraction if available
            if OCR_AVAILABLE:
                content = await self._extract_text_from_image(file_obj)
                if content:
                    file_obj.content_preview = content[:500]
                    file_obj.key_concepts.extend(await self._extract_key_concepts(content))
            
        except Exception as e:
            logger.error(f"Failed to analyze image {file_obj.filename}: {e}")
    
    async def _analyze_document_content(self, file_obj: ProjectFile):
        """Analyze document content (PDF, DOCX)"""
        
        try:
            content = ""
            
            # PDF extraction
            if file_obj.filename.lower().endswith('.pdf') and PDF_AVAILABLE:
                content = await self._extract_text_from_pdf(file_obj)
            
            # DOCX extraction
            elif file_obj.filename.lower().endswith('.docx'):
                content = await self._extract_text_from_docx(file_obj)
            
            if content:
                file_obj.content_preview = content[:1000]
                file_obj.key_concepts = await self._extract_key_concepts(content)
            
        except Exception as e:
            logger.error(f"Failed to analyze document {file_obj.filename}: {e}")
    
    async def _extract_text_from_image(self, file_obj: ProjectFile) -> Optional[str]:
        """Extract text from image using OCR"""
        
        if not OCR_AVAILABLE:
            return None
        
        try:
            # Get file data from VM
            file_data = await self._read_file_data(file_obj)
            
            # Convert to PIL Image
            image = Image.open(BytesIO(file_data))
            
            # Extract text using Tesseract
            text = pytesseract.image_to_string(image)
            
            return text.strip() if text else None
            
        except Exception as e:
            logger.error(f"OCR failed for {file_obj.filename}: {e}")
            return None
    
    async def _extract_text_from_pdf(self, file_obj: ProjectFile) -> Optional[str]:
        """Extract text from PDF"""
        
        if not PDF_AVAILABLE:
            return None
        
        try:
            # Get file data from VM
            file_data = await self._read_file_data(file_obj)
            
            # Extract text using pypdf
            reader = PdfReader(BytesIO(file_data))
            text = ""
            
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip() if text else None
            
        except Exception as e:
            logger.error(f"PDF extraction failed for {file_obj.filename}: {e}")
            return None
    
    async def _extract_text_from_docx(self, file_obj: ProjectFile) -> Optional[str]:
        """Extract text from DOCX"""
        
        try:
            # Get file data from VM
            file_data = await self._read_file_data(file_obj)
            
            # Extract text using python-docx
            doc = Document(BytesIO(file_data))
            text = ""
            
            for paragraph in doc.paragraphs:
                text += paragraph.text + "\n"
            
            return text.strip() if text else None
            
        except Exception as e:
            logger.error(f"DOCX extraction failed for {file_obj.filename}: {e}")
            return None
    
    async def _extract_file_knowledge(self, file_obj: ProjectFile):
        """Extract knowledge items from file"""
        
        if not self.project_knowledge:
            return
        
        try:
            # Get file content
            content = None
            
            if self._is_text_file(file_obj):
                content = await self._read_file_content(file_obj)
            elif file_obj.category == FileCategory.DOCUMENT:
                content = file_obj.content_preview
            
            if not content:
                return
            
            # Create file analysis for knowledge extraction
            file_analysis = {
                'file_type': file_obj.category.value,
                'mime_type': file_obj.mime_type,
                'size_bytes': file_obj.size_bytes,
                'content_summary': file_obj.content_preview,
                'key_concepts': file_obj.key_concepts,
                'importance_score': await self._calculate_importance_score(file_obj),
                'relationships': list(file_obj.related_files),
                'category': file_obj.category.value
            }
            
            # Extract knowledge using knowledge base
            knowledge_items = await self.project_knowledge.extract_knowledge_from_content(
                content=content,
                source_file=file_obj.filename,
                file_analysis=file_analysis
            )
            
            # Track knowledge items
            file_obj.knowledge_items = [str(item.item_id) for item in knowledge_items]
            file_obj.knowledge_extracted = True
            
            logger.info(f"Extracted {len(knowledge_items)} knowledge items from {file_obj.filename}")
            
        except Exception as e:
            logger.error(f"Failed to extract knowledge from {file_obj.filename}: {e}")
    
    async def _detect_file_relationships(self, file_obj: ProjectFile):
        """Detect relationships with other files"""
        
        for other_file_id, other_file in self.project_files.items():
            if other_file_id == file_obj.file_id:
                continue
            
            # Calculate similarity
            similarity = await self._calculate_file_similarity(file_obj, other_file)
            
            if similarity > 0.7:  # High similarity threshold
                file_obj.related_files.add(other_file_id)
                other_file.related_files.add(file_obj.file_id)
    
    async def _calculate_file_similarity(self, file1: ProjectFile, file2: ProjectFile) -> float:
        """Calculate similarity between two files"""
        
        similarity = 0.0
        
        # Category similarity
        if file1.category == file2.category:
            similarity += 0.3
        
        # Concept overlap
        concepts1 = set(file1.key_concepts)
        concepts2 = set(file2.key_concepts)
        
        if concepts1 and concepts2:
            overlap = len(concepts1 & concepts2)
            union = len(concepts1 | concepts2)
            similarity += (overlap / union) * 0.5
        
        # Tag similarity
        tags1 = file1.tags
        tags2 = file2.tags
        
        if tags1 and tags2:
            tag_overlap = len(tags1 & tags2)
            tag_union = len(tags1 | tags2)
            similarity += (tag_overlap / tag_union) * 0.2
        
        return similarity
    
    async def search_files(
        self,
        query: str,
        categories: Optional[List[FileCategory]] = None,
        tags: Optional[Set[str]] = None,
        date_range: Optional[Tuple[datetime, datetime]] = None,
        limit: int = 20
    ) -> List[FileSearchResult]:
        """Advanced file search with relevance scoring"""
        
        results = []
        query_terms = set(query.lower().split())
        
        for file_obj in self.project_files.values():
            # Category filter
            if categories and file_obj.category not in categories:
                continue
            
            # Tag filter
            if tags and not (tags & file_obj.tags):
                continue
            
            # Date range filter
            if date_range:
                start_date, end_date = date_range
                if not (start_date <= file_obj.uploaded_at <= end_date):
                    continue
            
            # Calculate relevance
            relevance_score = 0.0
            match_reasons = []
            content_matches = []
            
            # Filename match
            filename_terms = set(file_obj.filename.lower().split('.'))
            filename_overlap = len(query_terms & filename_terms)
            if filename_overlap > 0:
                relevance_score += filename_overlap * 0.3
                match_reasons.append("filename")
            
            # Concept match
            concept_terms = set(concept.lower() for concept in file_obj.key_concepts)
            concept_overlap = len(query_terms & concept_terms)
            if concept_overlap > 0:
                relevance_score += concept_overlap * 0.4
                match_reasons.append("concepts")
            
            # Tag match
            tag_terms = set(tag.lower() for tag in file_obj.tags)
            tag_overlap = len(query_terms & tag_terms)
            if tag_overlap > 0:
                relevance_score += tag_overlap * 0.2
                match_reasons.append("tags")
            
            # Content match (for text files)
            if self._is_text_file(file_obj) and file_obj.content_preview:
                content_terms = set(file_obj.content_preview.lower().split())
                content_overlap = len(query_terms & content_terms)
                if content_overlap > 0:
                    relevance_score += content_overlap * 0.1
                    match_reasons.append("content")
                    content_matches = [term for term in query_terms if term in content_terms]
            
            # Include if relevant
            if relevance_score > 0:
                results.append(FileSearchResult(
                    file=file_obj,
                    relevance_score=relevance_score,
                    match_reasons=match_reasons,
                    content_matches=content_matches
                ))
        
        # Sort by relevance
        results.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return results[:limit]
    
    async def get_file_by_id(self, file_id: UUID) -> Optional[ProjectFile]:
        """Get file by ID"""
        return self.project_files.get(file_id)
    
    async def get_file_by_path(self, file_path: str) -> Optional[ProjectFile]:
        """Get file by path"""
        file_id = self.files_by_path.get(file_path)
        return self.project_files.get(file_id) if file_id else None
    
    async def list_files(
        self,
        category: Optional[FileCategory] = None,
        limit: int = 100,
        sort_by: str = "uploaded_at",
        ascending: bool = False
    ) -> List[ProjectFile]:
        """List files with filtering and sorting"""
        
        files = list(self.project_files.values())
        
        # Filter by category
        if category:
            files = [f for f in files if f.category == category]
        
        # Sort files
        if sort_by in ["uploaded_at", "last_modified", "last_accessed"]:
            files.sort(key=lambda f: getattr(f, sort_by), reverse=not ascending)
        elif sort_by == "size":
            files.sort(key=lambda f: f.size_bytes, reverse=not ascending)
        elif sort_by == "filename":
            files.sort(key=lambda f: f.filename.lower(), reverse=not ascending)
        
        return files[:limit]
    
    async def delete_file(self, file_id: UUID) -> bool:
        """Delete file from project"""
        
        if file_id not in self.project_files:
            return False
        
        file_obj = self.project_files[file_id]
        
        try:
            # Remove file from VM
            await self.project_vm.execute_command(f"rm -f '{file_obj.file_path}'")
            
            # Remove from tracking
            del self.project_files[file_id]
            self.files_by_path.pop(file_obj.file_path, None)
            self.files_by_category[file_obj.category].discard(file_id)
            
            # Update metrics
            self.metrics['total_files'] -= 1
            self.metrics['total_size_bytes'] -= file_obj.size_bytes
            
            # Remove metadata
            await self._remove_file_metadata(file_id)
            
            logger.info(f"File deleted: {file_obj.filename}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete file {file_obj.filename}: {e}")
            return False
    
    async def get_file_statistics(self) -> Dict[str, Any]:
        """Get comprehensive file statistics"""
        
        stats = {
            "total_files": len(self.project_files),
            "total_size_bytes": sum(f.size_bytes for f in self.project_files.values()),
            "files_by_category": {
                category.value: len(file_ids)
                for category, file_ids in self.files_by_category.items()
                if file_ids
            },
            "processing_status": {
                status.value: len([f for f in self.project_files.values() if f.processing_status == status])
                for status in FileProcessingStatus
            },
            "average_file_size": (
                sum(f.size_bytes for f in self.project_files.values()) / max(len(self.project_files), 1)
            ),
            "files_with_knowledge": len([f for f in self.project_files.values() if f.knowledge_extracted]),
            "files_with_relationships": len([f for f in self.project_files.values() if f.related_files]),
            "recent_uploads": [
                {
                    "filename": f.filename,
                    "size_bytes": f.size_bytes,
                    "category": f.category.value,
                    "uploaded_at": f.uploaded_at.isoformat()
                }
                for f in sorted(
                    self.project_files.values(),
                    key=lambda x: x.uploaded_at,
                    reverse=True
                )[:10]
            ],
            "performance_metrics": self.metrics
        }
        
        return stats
    
    # Helper methods
    
    def _sanitize_filename(self, filename: str) -> str:
        """Sanitize filename for safe storage"""
        
        # Remove or replace unsafe characters
        unsafe_chars = '<>:"/\\|?*'
        for char in unsafe_chars:
            filename = filename.replace(char, '_')
        
        # Ensure unique filename if needed
        base_name = filename
        counter = 1
        
        while any(f.filename == filename for f in self.project_files.values()):
            name, ext = base_name.rsplit('.', 1) if '.' in base_name else (base_name, '')
            filename = f"{name}_{counter}.{ext}" if ext else f"{name}_{counter}"
            counter += 1
        
        return filename
    
    def _categorize_file(self, filename: str, mime_type: str) -> FileCategory:
        """Automatically categorize file based on name and MIME type"""
        
        ext = Path(filename).suffix.lower()
        
        # Document files
        if ext in ['.pdf', '.doc', '.docx', '.txt', '.md', '.rtf'] or mime_type.startswith('text/'):
            return FileCategory.DOCUMENT
        
        # Code files
        elif ext in ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c', '.h', '.php', '.rb', '.go', '.rs']:
            return FileCategory.CODE
        
        # Data files
        elif ext in ['.csv', '.json', '.xml', '.xlsx', '.sql'] or 'application/json' in mime_type:
            return FileCategory.DATA
        
        # Image files
        elif mime_type.startswith('image/'):
            return FileCategory.IMAGE
        
        # Audio files
        elif mime_type.startswith('audio/'):
            return FileCategory.AUDIO
        
        # Video files
        elif mime_type.startswith('video/'):
            return FileCategory.VIDEO
        
        # Archive files
        elif ext in ['.zip', '.tar', '.gz', '.rar', '.7z']:
            return FileCategory.ARCHIVE
        
        # Configuration files
        elif ext in ['.conf', '.config', '.ini', '.yaml', '.yml', '.toml']:
            return FileCategory.CONFIGURATION
        
        # Executable files
        elif ext in ['.exe', '.dll', '.so', '.bin'] or 'application/octet-stream' in mime_type:
            return FileCategory.EXECUTABLE
        
        else:
            return FileCategory.UNKNOWN
    
    def _is_text_file(self, file_obj: ProjectFile) -> bool:
        """Check if file is text-based"""
        
        ext = Path(file_obj.filename).suffix.lower()
        return (
            ext in self.supported_text_extensions or
            file_obj.mime_type.startswith('text/') or
            file_obj.category in [FileCategory.DOCUMENT, FileCategory.CODE]
        )
    
    def _is_image_file(self, file_obj: ProjectFile) -> bool:
        """Check if file is an image"""
        return file_obj.category == FileCategory.IMAGE or file_obj.mime_type.startswith('image/')
    
    def _is_code_file(self, file_obj: ProjectFile) -> bool:
        """Check if file is code"""
        return file_obj.category == FileCategory.CODE
    
    async def _read_file_content(self, file_obj: ProjectFile) -> Optional[str]:
        """Read file content from VM"""
        
        try:
            # Use VM to read file content
            result = await self.project_vm.execute_command(
                f"python3 /project/file_processing/toolkit.py info '{file_obj.file_path}'"
            )
            
            if result and "Error" not in result:
                # Read actual file content
                content_result = await self.project_vm.execute_command(
                    f"head -c 10000 '{file_obj.file_path}'"  # Read first 10KB
                )
                return content_result.strip() if content_result else None
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to read file content for {file_obj.filename}: {e}")
            return None
    
    async def _read_file_data(self, file_obj: ProjectFile) -> Optional[bytes]:
        """Read file data from VM"""
        
        try:
            # This would need to be implemented based on VM's file reading capabilities
            # For now, return None as placeholder
            return None
            
        except Exception as e:
            logger.error(f"Failed to read file data for {file_obj.filename}: {e}")
            return None
    
    async def _extract_text_preview(self, file_obj: ProjectFile) -> str:
        """Extract text preview from file"""
        
        content = await self._read_file_content(file_obj)
        return content[:500] if content else ""
    
    async def _detect_encoding(self, file_data: bytes) -> Optional[str]:
        """Detect file encoding"""
        
        if not CHARDET_AVAILABLE:
            return None
        
        try:
            result = chardet.detect(file_data[:1024])  # Sample first 1KB
            return result.get('encoding') if result.get('confidence', 0) > 0.7 else None
        except Exception:
            return None
    
    async def _extract_key_concepts(self, content: str) -> List[str]:
        """Extract key concepts from text content"""
        
        # Simple keyword extraction (could be enhanced with NLP)
        words = content.lower().split()
        
        # Filter common words
        stop_words = {
            'the', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 
            'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 
            'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 
            'can', 'this', 'that', 'these', 'those'
        }
        
        # Count word frequency
        word_freq = {}
        for word in words:
            if len(word) > 3 and word not in stop_words and word.isalpha():
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Return top concepts
        top_concepts = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in top_concepts[:10]]
    
    async def _analyze_code_content(self, content: str) -> List[str]:
        """Analyze code content for concepts"""
        
        concepts = []
        
        # Programming language detection
        if 'def ' in content or 'import ' in content:
            concepts.append('python')
        if 'function' in content or 'var ' in content or 'const ' in content:
            concepts.append('javascript')
        if 'class ' in content:
            concepts.append('class_definition')
        if 'import ' in content or 'from ' in content:
            concepts.append('imports')
        if '#include' in content:
            concepts.append('c_cpp')
        if 'public class' in content:
            concepts.append('java')
        
        return concepts
    
    async def _detect_language(self, file_obj: ProjectFile) -> Optional[str]:
        """Detect programming/natural language"""
        
        ext = Path(file_obj.filename).suffix.lower()
        
        # Programming language mapping
        lang_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.html': 'html',
            '.css': 'css',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.php': 'php',
            '.rb': 'ruby',
            '.go': 'go',
            '.rs': 'rust'
        }
        
        return lang_map.get(ext)
    
    async def _calculate_importance_score(self, file_obj: ProjectFile) -> float:
        """Calculate file importance score"""
        
        score = 0.0
        
        # Size factor (larger files might be more important)
        if file_obj.size_bytes > 1024 * 1024:  # > 1MB
            score += 0.3
        elif file_obj.size_bytes > 10 * 1024:  # > 10KB
            score += 0.1
        
        # Concept richness
        score += min(len(file_obj.key_concepts) / 10.0, 0.4)
        
        # Category importance
        category_weights = {
            FileCategory.DOCUMENT: 0.8,
            FileCategory.CODE: 0.9,
            FileCategory.DATA: 0.7,
            FileCategory.RESEARCH: 0.8,
            FileCategory.CONFIGURATION: 0.6
        }
        score += category_weights.get(file_obj.category, 0.3)
        
        return min(score, 1.0)
    
    async def _integrate_with_intelligence(self, file_obj: ProjectFile):
        """Integrate file with project intelligence system"""
        
        if hasattr(self.project_intelligence, 'register_file'):
            await self.project_intelligence.register_file(file_obj)
    
    async def _find_file_by_hash(self, file_hash: str) -> Optional[ProjectFile]:
        """Find existing file by hash"""
        
        for file_obj in self.project_files.values():
            if file_obj.file_hash == file_hash:
                return file_obj
        
        return None
    
    async def _save_file_metadata(self, file_obj: ProjectFile):
        """Save file metadata to VM storage"""
        
        metadata = {
            'file_id': str(file_obj.file_id),
            'filename': file_obj.filename,
            'original_filename': file_obj.original_filename,
            'file_path': file_obj.file_path,
            'size_bytes': file_obj.size_bytes,
            'mime_type': file_obj.mime_type,
            'file_hash': file_obj.file_hash,
            'category': file_obj.category.value,
            'tags': list(file_obj.tags),
            'processing_status': file_obj.processing_status.value,
            'key_concepts': file_obj.key_concepts,
            'uploaded_by': file_obj.uploaded_by,
            'uploaded_at': file_obj.uploaded_at.isoformat(),
            'last_modified': file_obj.last_modified.isoformat(),
            'knowledge_items': file_obj.knowledge_items,
            'related_files': [str(fid) for fid in file_obj.related_files]
        }
        
        metadata_path = f"/project/file_metadata/index/{file_obj.file_id}.json"
        await self.project_vm.write_file(metadata_path, json.dumps(metadata, indent=2))
    
    async def _remove_file_metadata(self, file_id: UUID):
        """Remove file metadata"""
        
        metadata_path = f"/project/file_metadata/index/{file_id}.json"
        await self.project_vm.execute_command(f"rm -f {metadata_path}")
    
    async def _scan_existing_files(self):
        """Scan for existing files in project"""
        
        try:
            # Use VM toolkit to scan files
            result = await self.project_vm.execute_command(
                "python3 /project/file_processing/toolkit.py scan /project/files"
            )
            
            if result.strip():
                try:
                    existing_files = json.loads(result)
                    logger.info(f"Found {len(existing_files)} existing files")
                    
                    # TODO: Recreate file objects from existing files
                    # This would involve loading metadata and reconstructing ProjectFile objects
                except json.JSONDecodeError:
                    logger.warning("Failed to parse existing files JSON")
        
        except Exception as e:
            logger.info(f"No existing files found: {e}")
    
    async def _initialize_file_monitoring(self):
        """Initialize file system monitoring"""
        
        # Start file monitoring task
        if not self.file_monitor_running:
            self.file_monitor_task = asyncio.create_task(self._run_file_monitor())
            self.file_monitor_running = True
    
    async def _start_file_processing(self):
        """Start file processing system"""
        
        # Start processing worker
        asyncio.create_task(self._process_file_queue())
    
    async def _process_file_queue(self):
        """Process file queue worker"""
        
        while True:
            try:
                # Get file from queue
                file_id = await self.processing_queue.get()
                
                # Process file
                await self.process_file(file_id)
                
                # Mark task done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"File processing worker error: {e}")
                await asyncio.sleep(5)
    
    async def _run_file_monitor(self):
        """Run file system monitoring"""
        
        while self.file_monitor_running:
            try:
                # Check for new files (simplified monitoring)
                await asyncio.sleep(30)  # Check every 30 seconds
                
                # TODO: Implement proper file system monitoring
                # This could use inotify or similar for real-time detection
                
            except Exception as e:
                logger.error(f"File monitor error: {e}")
                await asyncio.sleep(60)
    
    async def cleanup(self):
        """Cleanup file management resources"""
        
        # Stop file monitoring
        self.file_monitor_running = False
        if self.file_monitor_task:
            self.file_monitor_task.cancel()
        
        # Clear data structures
        self.project_files.clear()
        self.files_by_path.clear()
        for category_set in self.files_by_category.values():
            category_set.clear()
        
        logger.info(f"File manager cleanup completed for project {self.project_id}")


# Additional utility classes for integration

class FileUploadResponse(BaseModel):
    """Response model for file upload operations"""
    file_id: UUID
    filename: str
    size_bytes: int
    category: FileCategory
    processing_status: FileProcessingStatus
    uploaded_at: datetime


class FileSearchRequest(BaseModel):
    """Request model for file search operations"""
    query: str
    categories: Optional[List[FileCategory]] = None
    tags: Optional[Set[str]] = None
    date_from: Optional[datetime] = None
    date_to: Optional[datetime] = None
    limit: int = 20


class FileListRequest(BaseModel):
    """Request model for file listing operations"""
    category: Optional[FileCategory] = None
    sort_by: str = "uploaded_at"
    ascending: bool = False
    limit: int = 100
    offset: int = 0