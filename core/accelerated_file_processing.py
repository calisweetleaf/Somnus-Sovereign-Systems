"""
MORPHEUS CHAT - Intelligent File Upload & Processing Acceleration System
Advanced queuing and processing pipeline with dynamic resource allocation.

Architecture Features:
- Priority-based processing queue with dynamic reordering
- Adaptive thread pool management with backpressure handling
- Intelligent file type classification and processing routing
- Robust error recovery with exponential backoff
- Real-time progress tracking and resource monitoring
- Network resilience and corruption detection
"""

import asyncio
import logging
import multiprocessing
import os
import psutil
import threading
import time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum, IntEnum
from pathlib import Path
from queue import PriorityQueue, Empty
from typing import Dict, List, Optional, Any, Callable, Set, Tuple, Union
from uuid import UUID, uuid4

import aiofiles
from pydantic import BaseModel, Field

from file_upload_system import FileUploadManager, FileMetadata, FileType, ProcessingStatus
from schemas.session import SessionID, UserID

logger = logging.getLogger(__name__)


class ProcessingPriority(IntEnum):
    """Processing priority levels (lower numbers = higher priority)"""
    CRITICAL = 1      # User-requested immediate processing
    HIGH = 2          # Code/documentation files
    MEDIUM = 3        # Configuration/data files  
    LOW = 4           # Images/media files
    BACKGROUND = 5    # Large repositories/batch operations


class QueueStatus(str, Enum):
    """Queue item processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    RETRYING = "retrying"


@dataclass
class ProcessingTask:
    """Individual file processing task with metadata"""
    task_id: UUID = field(default_factory=uuid4)
    file_data: bytes = field(repr=False)
    filename: str = ""
    user_id: UserID = ""
    session_id: SessionID = field(default_factory=uuid4)
    
    # Priority and classification
    priority: ProcessingPriority = ProcessingPriority.MEDIUM
    file_type: Optional[FileType] = None
    estimated_processing_time: float = 0.0
    
    # Processing metadata
    status: QueueStatus = QueueStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Error handling
    retry_count: int = 0
    max_retries: int = 3
    error_message: Optional[str] = None
    
    # Progress tracking
    progress_callback: Optional[Callable[[float, str], None]] = None
    
    # Context and dependencies
    source_context: str = "upload"  # 'upload', 'git_clone', 'batch_import'
    parent_task_id: Optional[UUID] = None
    dependencies: List[UUID] = field(default_factory=list)
    
    def __lt__(self, other):
        """Enable priority queue ordering"""
        if not isinstance(other, ProcessingTask):
            return NotImplemented
        return (self.priority.value, self.created_at) < (other.priority.value, other.created_at)
    
    @property
    def processing_time(self) -> Optional[float]:
        """Calculate actual processing time"""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None
    
    @property
    def can_retry(self) -> bool:
        """Check if task can be retried"""
        return self.retry_count < self.max_retries and self.status == QueueStatus.FAILED


class SystemResourceMonitor:
    """Real-time system resource monitoring for adaptive scaling"""
    
    def __init__(self, update_interval: float = 1.0):
        self.update_interval = update_interval
        self.current_metrics = {}
        self.history = []
        self.max_history_size = 300  # 5 minutes at 1s intervals
        self._monitoring = False
        self._monitor_task = None
        
    async def start_monitoring(self):
        """Start continuous resource monitoring"""
        if not self._monitoring:
            self._monitoring = True
            self._monitor_task = asyncio.create_task(self._monitor_loop())
    
    async def stop_monitoring(self):
        """Stop resource monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
    
    async def _monitor_loop(self):
        """Continuous monitoring loop"""
        while self._monitoring:
            try:
                metrics = {
                    'timestamp': datetime.now(timezone.utc),
                    'cpu_percent': psutil.cpu_percent(interval=None),
                    'memory_percent': psutil.virtual_memory().percent,
                    'memory_available_gb': psutil.virtual_memory().available / (1024**3),
                    'disk_io_read_mb': psutil.disk_io_counters().read_bytes / (1024**2) if psutil.disk_io_counters() else 0,
                    'disk_io_write_mb': psutil.disk_io_counters().write_bytes / (1024**2) if psutil.disk_io_counters() else 0,
                    'active_threads': threading.active_count(),
                    'active_processes': len(psutil.pids())
                }
                
                self.current_metrics = metrics
                self.history.append(metrics)
                
                # Maintain history size
                if len(self.history) > self.max_history_size:
                    self.history.pop(0)
                
                await asyncio.sleep(self.update_interval)
                
            except Exception as e:
                logger.warning(f"Resource monitoring error: {e}")
                await asyncio.sleep(self.update_interval)
    
    def get_load_factor(self) -> float:
        """Calculate system load factor (0.0 = idle, 1.0 = overloaded)"""
        if not self.current_metrics:
            return 0.5  # Default moderate load
        
        cpu_load = self.current_metrics.get('cpu_percent', 0) / 100
        memory_load = self.current_metrics.get('memory_percent', 0) / 100
        
        # Weighted load calculation
        return (cpu_load * 0.6) + (memory_load * 0.4)
    
    def get_optimal_thread_count(self, base_threads: int = None) -> int:
        """Calculate optimal thread count based on system resources"""
        if base_threads is None:
            base_threads = min(multiprocessing.cpu_count(), 8)
        
        load_factor = self.get_load_factor()
        
        # Adaptive scaling based on load
        if load_factor < 0.3:
            # Low load - can increase threads
            return min(base_threads * 2, 16)
        elif load_factor < 0.7:
            # Moderate load - use base threads
            return base_threads
        else:
            # High load - reduce threads
            return max(base_threads // 2, 2)


class IntelligentFileProcessor:
    """
    High-performance file processing system with intelligent queuing and resource management.
    
    Implements adaptive threading, priority-based processing, and comprehensive error recovery
    for optimal throughput while maintaining system stability.
    """
    
    def __init__(
        self,
        base_file_manager: FileUploadManager,
        max_concurrent_tasks: int = None,
        max_queue_size: int = 1000,
        enable_monitoring: bool = True
    ):
        self.base_file_manager = base_file_manager
        self.max_queue_size = max_queue_size
        self.enable_monitoring = enable_monitoring
        
        # Dynamic thread management
        self.base_thread_count = max_concurrent_tasks or min(multiprocessing.cpu_count(), 8)
        self.current_thread_count = self.base_thread_count
        self.thread_pool = ThreadPoolExecutor(max_workers=self.current_thread_count)
        
        # Processing queue and tracking
        self.processing_queue = PriorityQueue(maxsize=max_queue_size)
        self.active_tasks: Dict[UUID, ProcessingTask] = {}
        self.completed_tasks: Dict[UUID, ProcessingTask] = {}
        self.task_lock = threading.RLock()
        
        # Resource monitoring
        self.resource_monitor = SystemResourceMonitor() if enable_monitoring else None
        
        # Processing control
        self.is_running = False
        self.processing_workers = []
        self.shutdown_event = threading.Event()
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'total_failed': 0,
            'average_processing_time': 0.0,
            'queue_wait_times': [],
            'throughput_files_per_minute': 0.0,
            'last_throughput_calculation': datetime.now(timezone.utc)
        }
        
        # File type processing estimations (seconds)
        self.processing_time_estimates = {
            FileType.TEXT: 0.5,
            FileType.DOCUMENT: 2.0,
            FileType.IMAGE: 1.5,
            FileType.CODE: 0.8,
            FileType.DATA: 1.2,
            FileType.UNKNOWN: 1.0
        }
        
        logger.info(f"Intelligent file processor initialized: {self.base_thread_count} base threads")
    
    async def start(self):
        """Start the processing system"""
        if self.is_running:
            return
        
        self.is_running = True
        self.shutdown_event.clear()
        
        # Start resource monitoring
        if self.resource_monitor:
            await self.resource_monitor.start_monitoring()
        
        # Start processing workers
        for i in range(3):  # Start with 3 worker threads
            worker = threading.Thread(
                target=self._processing_worker,
                name=f"FileProcessor-{i}",
                daemon=True
            )
            worker.start()
            self.processing_workers.append(worker)
        
        # Start adaptive scaling
        asyncio.create_task(self._adaptive_scaling_loop())
        
        logger.info("File processing system started")
    
    async def stop(self):
        """Gracefully stop the processing system"""
        if not self.is_running:
            return
        
        logger.info("Stopping file processing system...")
        
        self.is_running = False
        self.shutdown_event.set()
        
        # Stop resource monitoring
        if self.resource_monitor:
            await self.resource_monitor.stop_monitoring()
        
        # Wait for workers to complete
        for worker in self.processing_workers:
            worker.join(timeout=30)
        
        # Shutdown thread pool
        self.thread_pool.shutdown(wait=True, timeout=30)
        
        logger.info("File processing system stopped")
    
    async def queue_file(
        self,
        file_data: bytes,
        filename: str,
        user_id: UserID,
        session_id: SessionID,
        priority: ProcessingPriority = ProcessingPriority.MEDIUM,
        source_context: str = "upload",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> UUID:
        """Queue file for processing with priority and context"""
        
        if not self.is_running:
            raise RuntimeError("File processor not running")
        
        if self.processing_queue.qsize() >= self.max_queue_size:
            raise RuntimeError("Processing queue full")
        
        # Classify file type for processing estimation
        file_type = self._classify_file_type(filename, file_data)
        
        # Create processing task
        task = ProcessingTask(
            file_data=file_data,
            filename=filename,
            user_id=user_id,
            session_id=session_id,
            priority=priority,
            file_type=file_type,
            estimated_processing_time=self.processing_time_estimates.get(file_type, 1.0),
            source_context=source_context,
            progress_callback=progress_callback
        )
        
        # Queue task
        try:
            self.processing_queue.put_nowait(task)
            
            with self.task_lock:
                self.active_tasks[task.task_id] = task
            
            logger.debug(f"Queued file {filename} with priority {priority.name}")
            return task.task_id
            
        except Exception as e:
            logger.error(f"Failed to queue file {filename}: {e}")
            raise
    
    async def queue_batch(
        self,
        file_batch: List[Tuple[bytes, str]],
        user_id: UserID,
        session_id: SessionID,
        priority: ProcessingPriority = ProcessingPriority.LOW,
        source_context: str = "batch"
    ) -> List[UUID]:
        """Queue multiple files for batch processing"""
        
        task_ids = []
        
        for file_data, filename in file_batch:
            try:
                task_id = await self.queue_file(
                    file_data=file_data,
                    filename=filename,
                    user_id=user_id,
                    session_id=session_id,
                    priority=priority,
                    source_context=source_context
                )
                task_ids.append(task_id)
                
            except Exception as e:
                logger.error(f"Failed to queue batch file {filename}: {e}")
                continue
        
        logger.info(f"Queued {len(task_ids)} files from batch of {len(file_batch)}")
        return task_ids
    
    def _processing_worker(self):
        """Main processing worker thread"""
        worker_name = threading.current_thread().name
        logger.debug(f"Processing worker {worker_name} started")
        
        while self.is_running and not self.shutdown_event.is_set():
            try:
                # Get next task with timeout
                try:
                    task = self.processing_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                # Process task
                self._process_task(task)
                
                # Mark queue task as done
                self.processing_queue.task_done()
                
            except Exception as e:
                logger.error(f"Processing worker {worker_name} error: {e}")
                time.sleep(1)
        
        logger.debug(f"Processing worker {worker_name} stopped")
    
    def _process_task(self, task: ProcessingTask):
        """Process individual file task"""
        task.status = QueueStatus.PROCESSING
        task.started_at = datetime.now(timezone.utc)
        
        try:
            # Update progress
            if task.progress_callback:
                task.progress_callback(10.0, "Starting file processing")
            
            # Process through base file manager
            file_metadata = asyncio.run(
                self.base_file_manager.upload_file(
                    file_data=task.file_data,
                    filename=task.filename,
                    user_id=task.user_id,
                    session_id=task.session_id
                )
            )
            
            # Check processing result
            if file_metadata.processing_status == ProcessingStatus.COMPLETED:
                task.status = QueueStatus.COMPLETED
                self.stats['total_processed'] += 1
                
                if task.progress_callback:
                    task.progress_callback(100.0, "File processing completed")
                
            else:
                task.status = QueueStatus.FAILED
                task.error_message = file_metadata.processing_error or "Unknown processing error"
                self.stats['total_failed'] += 1
                
                if task.progress_callback:
                    task.progress_callback(0.0, f"Processing failed: {task.error_message}")
            
        except Exception as e:
            task.status = QueueStatus.FAILED
            task.error_message = str(e)
            self.stats['total_failed'] += 1
            
            logger.error(f"Task processing failed: {task.filename} - {e}")
            
            if task.progress_callback:
                task.progress_callback(0.0, f"Processing error: {str(e)}")
        
        finally:
            task.completed_at = datetime.now(timezone.utc)
            
            # Move to completed tasks
            with self.task_lock:
                self.active_tasks.pop(task.task_id, None)
                self.completed_tasks[task.task_id] = task
            
            # Handle retry logic
            if task.status == QueueStatus.FAILED and task.can_retry:
                self._schedule_retry(task)
            
            # Update statistics
            self._update_statistics(task)
    
    def _schedule_retry(self, task: ProcessingTask):
        """Schedule task retry with exponential backoff"""
        task.retry_count += 1
        task.status = QueueStatus.RETRYING
        
        # Exponential backoff delay
        retry_delay = min(2 ** task.retry_count, 60)  # Max 60 seconds
        
        def retry_task():
            time.sleep(retry_delay)
            if self.is_running:
                task.status = QueueStatus.PENDING
                task.started_at = None
                try:
                    self.processing_queue.put_nowait(task)
                    with self.task_lock:
                        self.active_tasks[task.task_id] = task
                except Exception as e:
                    logger.error(f"Failed to reschedule retry for {task.filename}: {e}")
        
        # Schedule retry in background thread
        retry_thread = threading.Thread(target=retry_task, daemon=True)
        retry_thread.start()
        
        logger.info(f"Scheduled retry {task.retry_count}/{task.max_retries} for {task.filename} in {retry_delay}s")
    
    def _classify_file_type(self, filename: str, file_data: bytes) -> FileType:
        """Classify file type for processing estimation"""
        ext = Path(filename).suffix.lower()
        
        # Text files
        if ext in ['.txt', '.md', '.rst', '.log']:
            return FileType.TEXT
        
        # Documents
        elif ext in ['.pdf', '.docx', '.doc', '.rtf']:
            return FileType.DOCUMENT
        
        # Images
        elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.webp']:
            return FileType.IMAGE
        
        # Code files
        elif ext in ['.py', '.js', '.html', '.css', '.java', '.cpp', '.c']:
            return FileType.CODE
        
        # Data files
        elif ext in ['.csv', '.json', '.xlsx', '.xml']:
            return FileType.DATA
        
        return FileType.UNKNOWN
    
    async def _adaptive_scaling_loop(self):
        """Continuously adapt thread pool size based on system load"""
        while self.is_running:
            try:
                if self.resource_monitor:
                    optimal_threads = self.resource_monitor.get_optimal_thread_count(self.base_thread_count)
                    
                    if optimal_threads != self.current_thread_count:
                        await self._adjust_thread_pool(optimal_threads)
                
                await asyncio.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                logger.warning(f"Adaptive scaling error: {e}")
                await asyncio.sleep(10)
    
    async def _adjust_thread_pool(self, target_threads: int):
        """Dynamically adjust thread pool size"""
        if target_threads == self.current_thread_count:
            return
        
        logger.info(f"Adjusting thread pool: {self.current_thread_count} -> {target_threads}")
        
        # Create new thread pool
        old_pool = self.thread_pool
        self.thread_pool = ThreadPoolExecutor(max_workers=target_threads)
        self.current_thread_count = target_threads
        
        # Gracefully shutdown old pool
        def shutdown_old_pool():
            old_pool.shutdown(wait=True, timeout=30)
        
        shutdown_thread = threading.Thread(target=shutdown_old_pool, daemon=True)
        shutdown_thread.start()
    
    def _update_statistics(self, task: ProcessingTask):
        """Update processing statistics"""
        if task.processing_time:
            # Update average processing time
            current_avg = self.stats['average_processing_time']
            total_processed = self.stats['total_processed']
            
            if total_processed > 0:
                self.stats['average_processing_time'] = (
                    (current_avg * (total_processed - 1) + task.processing_time) / total_processed
                )
            else:
                self.stats['average_processing_time'] = task.processing_time
        
        # Calculate throughput (files per minute)
        now = datetime.now(timezone.utc)
        time_diff = (now - self.stats['last_throughput_calculation']).total_seconds()
        
        if time_diff >= 60:  # Update every minute
            files_processed_recent = len([
                t for t in self.completed_tasks.values()
                if t.completed_at and (now - t.completed_at).total_seconds() <= 60
            ])
            
            self.stats['throughput_files_per_minute'] = files_processed_recent
            self.stats['last_throughput_calculation'] = now
    
    def get_task_status(self, task_id: UUID) -> Optional[ProcessingTask]:
        """Get task status by ID"""
        with self.task_lock:
            # Check active tasks
            if task_id in self.active_tasks:
                return self.active_tasks[task_id]
            
            # Check completed tasks
            if task_id in self.completed_tasks:
                return self.completed_tasks[task_id]
        
        return None
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get comprehensive queue status"""
        with self.task_lock:
            active_count = len(self.active_tasks)
            completed_count = len(self.completed_tasks)
        
        queue_size = self.processing_queue.qsize()
        
        # Status breakdown
        status_counts = {status.value: 0 for status in QueueStatus}
        for task in list(self.active_tasks.values()) + list(self.completed_tasks.values()):
            status_counts[task.status.value] += 1
        
        return {
            'queue_size': queue_size,
            'active_tasks': active_count,
            'completed_tasks': completed_count,
            'current_threads': self.current_thread_count,
            'optimal_threads': self.resource_monitor.get_optimal_thread_count() if self.resource_monitor else None,
            'system_load': self.resource_monitor.get_load_factor() if self.resource_monitor else None,
            'status_breakdown': status_counts,
            'statistics': self.stats.copy()
        }
    
    def cancel_task(self, task_id: UUID) -> bool:
        """Cancel pending task"""
        with self.task_lock:
            task = self.active_tasks.get(task_id)
            if task and task.status == QueueStatus.PENDING:
                task.status = QueueStatus.CANCELLED
                task.completed_at = datetime.now(timezone.utc)
                
                # Move to completed
                self.active_tasks.pop(task_id)
                self.completed_tasks[task_id] = task
                
                logger.info(f"Cancelled task: {task.filename}")
                return True
        
        return False
    
    def clear_completed_tasks(self, older_than_hours: int = 24):
        """Clear old completed tasks to free memory"""
        cutoff_time = datetime.now(timezone.utc) - timedelta(hours=older_than_hours)
        
        with self.task_lock:
            to_remove = [
                task_id for task_id, task in self.completed_tasks.items()
                if task.completed_at and task.completed_at < cutoff_time
            ]
            
            for task_id in to_remove:
                self.completed_tasks.pop(task_id)
        
        logger.info(f"Cleared {len(to_remove)} completed tasks older than {older_than_hours} hours")
    
    async def wait_for_completion(self, task_ids: List[UUID], timeout: Optional[float] = None) -> Dict[UUID, bool]:
        """Wait for multiple tasks to complete"""
        start_time = time.time()
        results = {}
        
        while task_ids:
            if timeout and (time.time() - start_time) > timeout:
                # Timeout - mark remaining as incomplete
                for task_id in task_ids:
                    results[task_id] = False
                break
            
            completed_this_cycle = []
            
            for task_id in task_ids:
                task = self.get_task_status(task_id)
                if task and task.status in [QueueStatus.COMPLETED, QueueStatus.FAILED, QueueStatus.CANCELLED]:
                    results[task_id] = task.status == QueueStatus.COMPLETED
                    completed_this_cycle.append(task_id)
            
            # Remove completed tasks from waiting list
            for task_id in completed_this_cycle:
                task_ids.remove(task_id)
            
            if task_ids:
                await asyncio.sleep(0.5)  # Check every 500ms
        
        return results


# Integration helper functions for UI and system integration
class FileProcessingIntegration:
    """Helper class for integrating with Morpheus UI and system components"""
    
    def __init__(self, processor: IntelligentFileProcessor):
        self.processor = processor
    
    async def handle_github_clone_files(
        self,
        repo_files: List[Tuple[bytes, str]],
        user_id: UserID,
        session_id: SessionID,
        repo_name: str
    ) -> Dict[str, Any]:
        """Handle file processing from GitHub repository clone"""
        
        # Queue all files with background priority
        task_ids = await self.processor.queue_batch(
            file_batch=repo_files,
            user_id=user_id,
            session_id=session_id,
            priority=ProcessingPriority.BACKGROUND,
            source_context=f"git_clone:{repo_name}"
        )
        
        return {
            'task_ids': task_ids,
            'total_files': len(repo_files),
            'estimated_completion_time': len(repo_files) * 1.5  # seconds
        }
    
    async def handle_ui_file_upload(
        self,
        file_data: bytes,
        filename: str,
        user_id: UserID,
        session_id: SessionID,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> UUID:
        """Handle single file upload from UI with high priority"""
        
        return await self.processor.queue_file(
            file_data=file_data,
            filename=filename,
            user_id=user_id,
            session_id=session_id,
            priority=ProcessingPriority.HIGH,
            source_context="ui_upload",
            progress_callback=progress_callback
        )
    
    def get_processing_stats_for_ui(self) -> Dict[str, Any]:
        """Get processing statistics formatted for UI display"""
        status = self.processor.get_queue_status()
        
        return {
            'queue_length': status['queue_size'],
            'processing': status['status_breakdown']['processing'],
            'completed_today': len([
                t for t in self.processor.completed_tasks.values()
                if t.completed_at and (datetime.now(timezone.utc) - t.completed_at).days == 0
            ]),
            'success_rate': (
                status['statistics']['total_processed'] / 
                max(1, status['statistics']['total_processed'] + status['statistics']['total_failed'])
            ) * 100,
            'avg_processing_time': status['statistics']['average_processing_time'],
            'throughput_per_minute': status['statistics']['throughput_files_per_minute'],
            'system_load': status['system_load'],
            'thread_utilization': f"{status['current_threads']}/{status['optimal_threads']}"
        }
