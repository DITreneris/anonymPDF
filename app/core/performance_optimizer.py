"""
Performance Optimizer - Priority 3 Session 3
Implements parallel processing, document streaming, and batch processing
for 3x speed improvement in PDF processing operations.
"""

import time
import threading
import queue
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, List, Any, Optional, Callable, Union, Iterator, Tuple
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
import psutil
import gc
from functools import lru_cache
from contextlib import contextmanager

from app.core.logging import StructuredLogger
from app.core.config_manager import get_config
from app.core.performance import PerformanceMonitor
from app.core.memory_optimizer import get_memory_optimizer
from app.core.intelligent_cache import get_intelligent_cache

optimizer_logger = StructuredLogger("anonympdf.performance_optimizer")


class ProcessingMode(Enum):
    """Processing execution modes."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREADS = "parallel_threads"
    PARALLEL_PROCESSES = "parallel_processes"
    ADAPTIVE = "adaptive"


@dataclass
class ProcessingTask:
    """Represents a processing task with metadata."""
    task_id: str
    data: Any
    priority: int = 1
    estimated_size: int = 0
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class ProcessingResult:
    """Represents processing result with performance metrics."""
    task_id: str
    result: Any
    success: bool
    duration: float
    memory_used: float
    error: Optional[str] = None
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class DocumentStreamer:
    """Handles memory-efficient streaming of large documents."""

    def __init__(self, chunk_size: int = 1024 * 1024):  # 1MB chunks
        self.chunk_size = chunk_size
        self.monitor = PerformanceMonitor()
        
    def stream_document(self, file_path: Path) -> Iterator[bytes]:
        """Stream document in chunks for memory-efficient processing."""
        try:
            file_size = file_path.stat().st_size
            optimizer_logger.info(
                "Starting document streaming",
                file_path=str(file_path),
                file_size_mb=file_size / 1024 / 1024,
                chunk_size_kb=self.chunk_size / 1024
            )
            
            with open(file_path, 'rb') as file:
                chunks_processed = 0
                while True:
                    chunk = file.read(self.chunk_size)
                    if not chunk:
                        break
                    
                    chunks_processed += 1
                    yield chunk
                    
                    # Periodic memory cleanup
                    if chunks_processed % 100 == 0:
                        gc.collect()
                        
        except Exception as e:
            optimizer_logger.error(
                "Document streaming failed",
                file_path=str(file_path),
                error=str(e)
            )
            raise

    def estimate_processing_time(self, file_path: Path) -> float:
        """Estimate processing time based on file size and historical data."""
        try:
            file_size_mb = file_path.stat().st_size / 1024 / 1024
            
            # Base estimation: 0.1 seconds per MB (can be calibrated)
            base_time = file_size_mb * 0.1
            
            # Adjust based on file type
            if file_path.suffix.lower() == '.pdf':
                base_time *= 1.2  # PDFs are more complex
            
            return max(base_time, 0.1)  # Minimum 100ms
            
        except Exception:
            return 1.0  # Default 1 second


class LoadBalancer:
    """Intelligent load balancing for optimal resource utilization."""

    def __init__(self):
        self.cpu_count = psutil.cpu_count()
        self.memory_gb = psutil.virtual_memory().total / (1024**3)
        self.current_load = 0
        self._lock = threading.Lock()
        
    def get_optimal_workers(self, task_count: int, avg_task_size_mb: float = 1.0) -> int:
        """Calculate optimal number of workers based on system resources."""
        with self._lock:
            # Base workers on CPU count
            cpu_workers = min(self.cpu_count, task_count)
            
            # Adjust for memory constraints
            # Safety check for zero division
            if avg_task_size_mb <= 0:
                avg_task_size_mb = 1.0  # Default to 1MB if not specified
            
            memory_workers = max(1, int(self.memory_gb / (avg_task_size_mb * 0.1)))
            
            # Take minimum to avoid resource exhaustion
            optimal_workers = min(cpu_workers, memory_workers, 8)  # Max 8 workers
            
            optimizer_logger.debug(
                "Load balancer calculation",
                cpu_count=self.cpu_count,
                memory_gb=self.memory_gb,
                task_count=task_count,
                avg_task_size_mb=avg_task_size_mb,
                optimal_workers=optimal_workers
            )
            
            return optimal_workers

    def adjust_for_current_load(self, base_workers: int) -> int:
        """Adjust worker count based on current system load."""
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            memory_percent = psutil.virtual_memory().percent
            
            # Reduce workers if system is under heavy load
            if cpu_percent > 80 or memory_percent > 85:
                return max(1, base_workers // 2)
            elif cpu_percent > 60 or memory_percent > 70:
                return max(1, int(base_workers * 0.75))
            
            return base_workers
            
        except Exception:
            return base_workers


class ParallelProcessor:
    """Main parallel processing engine with intelligent task distribution."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()['performance']['parallel_processing']
        self.max_workers = self.config.get('max_workers', 4)
        self.chunk_size = self.config.get('chunk_size', 100)
        
        self.load_balancer = LoadBalancer()
        self.document_streamer = DocumentStreamer()
        self.monitor = PerformanceMonitor()
        
        self._task_queue = queue.PriorityQueue()
        self._results = {}
        self._lock = threading.Lock()
        
        optimizer_logger.info(
            "ParallelProcessor initialized",
            max_workers=self.max_workers,
            chunk_size=self.chunk_size
        )

    def _determine_processing_mode(self, tasks: List[ProcessingTask]) -> ProcessingMode:
        """Intelligently determine the best processing mode."""
        if len(tasks) == 1:
            # Single task - check size
            task = tasks[0]
            if task.estimated_size > 10 * 1024 * 1024:  # >10MB
                return ProcessingMode.PARALLEL_THREADS
            return ProcessingMode.SEQUENTIAL
            
        # Multiple tasks
        total_size = sum(task.estimated_size for task in tasks)
        avg_size = total_size / len(tasks) if tasks else 0
        
        if len(tasks) <= 2:
            return ProcessingMode.PARALLEL_THREADS
        elif avg_size > 5 * 1024 * 1024:  # >5MB average
            return ProcessingMode.PARALLEL_PROCESSES
        else:
            return ProcessingMode.PARALLEL_THREADS

    def process_single_document(
        self, 
        document_path: Path, 
        processor_func: Callable,
        **kwargs
    ) -> ProcessingResult:
        """Process a single document with optimal parallelization."""
        start_time = time.time()
        task_id = f"single_doc_{int(time.time())}"
        
        try:
            # Estimate processing requirements
            file_size = document_path.stat().st_size
            estimated_time = self.document_streamer.estimate_processing_time(document_path)
            
            optimizer_logger.info(
                "Starting single document processing",
                document_path=str(document_path),
                file_size_mb=file_size / 1024 / 1024,
                estimated_time=estimated_time
            )
            
            # Track memory usage
            process = psutil.Process()
            start_memory = process.memory_info().rss
            
            # Process the document
            result = processor_func(document_path, **kwargs)
            
            # Calculate metrics
            end_time = time.time()
            end_memory = process.memory_info().rss
            duration = end_time - start_time
            memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
            
            processing_result = ProcessingResult(
                task_id=task_id,
                result=result,
                success=True,
                duration=duration,
                memory_used=memory_used,
                metadata={
                    "file_size_mb": file_size / 1024 / 1024,
                    "processing_mode": "single_document",
                    "estimated_time": estimated_time,
                    "actual_vs_estimated": duration / estimated_time if estimated_time > 0 else 1.0
                }
            )
            
            optimizer_logger.info(
                "Single document processing completed",
                task_id=task_id,
                duration=duration,
                memory_used_mb=memory_used,
                success=True
            )
            
            return processing_result
            
        except Exception as e:
            duration = time.time() - start_time
            error_msg = str(e)
            
            optimizer_logger.error(
                "Single document processing failed",
                task_id=task_id,
                error=error_msg,
                duration=duration
            )
            
            return ProcessingResult(
                task_id=task_id,
                result=None,
                success=False,
                duration=duration,
                memory_used=0,
                error=error_msg
            )

    def process_batch(
        self,
        tasks: List[ProcessingTask],
        processor_func: Callable,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process multiple tasks in parallel with intelligent load balancing."""
        if not tasks:
            return []
        
        start_time = time.time()
        batch_id = f"batch_{int(start_time)}"
        
        optimizer_logger.info(
            "Starting batch processing",
            batch_id=batch_id,
            task_count=len(tasks),
            total_estimated_size=sum(task.estimated_size for task in tasks)
        )
        
        # Determine processing mode
        processing_mode = self._determine_processing_mode(tasks)
        
        # Calculate optimal workers
        avg_size_mb = sum(task.estimated_size for task in tasks) / len(tasks) / 1024 / 1024
        optimal_workers = self.load_balancer.get_optimal_workers(len(tasks), avg_size_mb)
        optimal_workers = self.load_balancer.adjust_for_current_load(optimal_workers)
        
        optimizer_logger.info(
            "Batch processing configuration",
            batch_id=batch_id,
            processing_mode=processing_mode.value,
            workers=optimal_workers,
            avg_task_size_mb=avg_size_mb
        )
        
        results = []
        
        try:
            if processing_mode == ProcessingMode.SEQUENTIAL:
                results = self._process_sequential(tasks, processor_func, **kwargs)
            elif processing_mode == ProcessingMode.PARALLEL_THREADS:
                results = self._process_parallel_threads(tasks, processor_func, optimal_workers, **kwargs)
            elif processing_mode == ProcessingMode.PARALLEL_PROCESSES:
                results = self._process_parallel_processes(tasks, processor_func, optimal_workers, **kwargs)
            
            duration = time.time() - start_time
            success_count = sum(1 for r in results if r.success)
            
            optimizer_logger.info(
                "Batch processing completed",
                batch_id=batch_id,
                total_duration=duration,
                tasks_processed=len(results),
                success_count=success_count,
                failure_count=len(results) - success_count,
                avg_task_duration=duration / len(results) if results else 0
            )
            
            return results
            
        except Exception as e:
            optimizer_logger.error(
                "Batch processing failed",
                batch_id=batch_id,
                error=str(e)
            )
            raise

    def _process_sequential(
        self,
        tasks: List[ProcessingTask],
        processor_func: Callable,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process tasks sequentially."""
        results = []
        for task in tasks:
            start_time = time.time()
            try:
                result = processor_func(task.data, **kwargs)
                results.append(ProcessingResult(
                    task_id=task.task_id,
                    result=result,
                    success=True,
                    duration=time.time() - start_time,
                    memory_used=0  # Sequential doesn't track individual memory
                ))
            except Exception as e:
                results.append(ProcessingResult(
                    task_id=task.task_id,
                    result=None,
                    success=False,
                    duration=time.time() - start_time,
                    memory_used=0,
                    error=str(e)
                ))
        return results

    def _process_parallel_threads(
        self,
        tasks: List[ProcessingTask],
        processor_func: Callable,
        workers: int,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process tasks using thread pool."""
        results = []
        
        with ThreadPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_task, task, processor_func, **kwargs): task
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(ProcessingResult(
                        task_id=task.task_id,
                        result=None,
                        success=False,
                        duration=0,
                        memory_used=0,
                        error=str(e)
                    ))
        
        return results

    def _process_parallel_processes(
        self,
        tasks: List[ProcessingTask],
        processor_func: Callable,
        workers: int,
        **kwargs
    ) -> List[ProcessingResult]:
        """Process tasks using process pool."""
        results = []
        
        with ProcessPoolExecutor(max_workers=workers) as executor:
            # Submit all tasks
            future_to_task = {
                executor.submit(self._execute_task, task, processor_func, **kwargs): task
                for task in tasks
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_task):
                task = future_to_task[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    results.append(ProcessingResult(
                        task_id=task.task_id,
                        result=None,
                        success=False,
                        duration=0,
                        memory_used=0,
                        error=str(e)
                    ))
        
        return results

    def _execute_task(
        self,
        task: ProcessingTask,
        processor_func: Callable,
        **kwargs
    ) -> ProcessingResult:
        """Execute a single task with performance tracking."""
        start_time = time.time()
        
        try:
            # Track memory if possible
            try:
                process = psutil.Process()
                start_memory = process.memory_info().rss
            except:
                start_memory = 0
            
            # Execute the task
            result = processor_func(task.data, **kwargs)
            
            # Calculate metrics
            duration = time.time() - start_time
            try:
                end_memory = psutil.Process().memory_info().rss
                memory_used = (end_memory - start_memory) / 1024 / 1024  # MB
            except:
                memory_used = 0
            
            return ProcessingResult(
                task_id=task.task_id,
                result=result,
                success=True,
                duration=duration,
                memory_used=memory_used,
                metadata=task.metadata
            )
            
        except Exception as e:
            return ProcessingResult(
                task_id=task.task_id,
                result=None,
                success=False,
                duration=time.time() - start_time,
                memory_used=0,
                error=str(e),
                metadata=task.metadata
            )


class BatchEngine:
    """High-level batch processing engine with queue management."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or get_config()['performance']
        self.parallel_processor = ParallelProcessor(self.config.get('parallel_processing'))
        
        self._processing_queue = queue.Queue()
        self._results_storage = {}
        self._active_batches = {}
        self._lock = threading.Lock()
        
        optimizer_logger.info("BatchEngine initialized")

    def submit_batch(
        self,
        batch_id: str,
        tasks: List[ProcessingTask],
        processor_func: Callable,
        priority: int = 1,
        **kwargs
    ) -> str:
        """Submit a batch for processing."""
        with self._lock:
            batch_info = {
                'batch_id': batch_id,
                'tasks': tasks,
                'processor_func': processor_func,
                'priority': priority,
                'kwargs': kwargs,
                'submitted_at': time.time(),
                'status': 'queued'
            }
            
            self._processing_queue.put((priority, batch_info))
            self._active_batches[batch_id] = batch_info
            
            optimizer_logger.info(
                "Batch submitted",
                batch_id=batch_id,
                task_count=len(tasks),
                priority=priority
            )
            
            return batch_id

    def process_next_batch(self) -> Optional[str]:
        """Process the next batch in the queue."""
        try:
            priority, batch_info = self._processing_queue.get_nowait()
            batch_id = batch_info['batch_id']
            
            with self._lock:
                self._active_batches[batch_id]['status'] = 'processing'
                self._active_batches[batch_id]['started_at'] = time.time()
            
            optimizer_logger.info(
                "Starting batch processing",
                batch_id=batch_id,
                priority=priority
            )
            
            # Process the batch
            results = self.parallel_processor.process_batch(
                batch_info['tasks'],
                batch_info['processor_func'],
                **batch_info['kwargs']
            )
            
            # Store results
            with self._lock:
                self._results_storage[batch_id] = results
                self._active_batches[batch_id]['status'] = 'completed'
                self._active_batches[batch_id]['completed_at'] = time.time()
            
            optimizer_logger.info(
                "Batch processing completed",
                batch_id=batch_id,
                results_count=len(results)
            )
            
            return batch_id
            
        except queue.Empty:
            return None
        except Exception as e:
            optimizer_logger.error(
                "Batch processing error",
                error=str(e)
            )
            raise

    def get_batch_results(self, batch_id: str) -> Optional[List[ProcessingResult]]:
        """Get results for a completed batch."""
        with self._lock:
            return self._results_storage.get(batch_id)

    def get_batch_status(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get status information for a batch."""
        with self._lock:
            return self._active_batches.get(batch_id)

    def get_queue_stats(self) -> Dict[str, Any]:
        """Get statistics about the processing queue."""
        with self._lock:
            return {
                'queued_batches': self._processing_queue.qsize(),
                'active_batches': len([b for b in self._active_batches.values() if b['status'] == 'processing']),
                'completed_batches': len([b for b in self._active_batches.values() if b['status'] == 'completed']),
                'total_batches': len(self._active_batches)
            }


# Global instances for easy access
_parallel_processor = None
_batch_engine = None

def get_parallel_processor() -> ParallelProcessor:
    """Get the global parallel processor instance."""
    global _parallel_processor
    if _parallel_processor is None:
        _parallel_processor = ParallelProcessor()
    return _parallel_processor

def get_batch_engine() -> BatchEngine:
    """Get the global batch engine instance."""
    global _batch_engine
    if _batch_engine is None:
        _batch_engine = BatchEngine()
    return _batch_engine

# Global performance optimizer instance
_performance_optimizer = None

class PerformanceOptimizer:
    """Main performance optimization coordinator that manages all optimization components."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        app_config = get_config()
        if config is None:
            self.config = app_config.get('performance_optimizer', app_config.get('performance', {}))
        else:
            self.config = config
        
        # Initialize all optimization components
        self.parallel_processor = ParallelProcessor(self.config.get('parallel_processing'))
        self.batch_engine = BatchEngine(self.config)
        
        self.memory_optimizer = get_memory_optimizer()
        self.intelligent_cache = get_intelligent_cache()
        
        # Performance monitoring
        self.monitor = PerformanceMonitor()
        
        # Optimization statistics
        self._optimization_stats = {
            'total_optimizations': 0,
            'speed_improvements': [],
            'memory_reductions': [],
            'cache_hit_rates': []
        }
        self._lock = threading.Lock()
        
        optimizer_logger.info(
            "PerformanceOptimizer initialized",
            has_cache=self.intelligent_cache is not None,
            config_keys=list(self.config.keys())
        )
    
    @contextmanager
    def parallel_processing_context(self, max_workers: Optional[int] = None, use_processes: bool = False, processing_mode: str = "normal"):
        """
        Context manager to set up and tear down optimizations around a parallel processing block.
        Currently, it primarily leverages the memory_optimizer's context.
        Args:
            max_workers: Hint for max workers, can be used for tuning.
            use_processes: Hint if process-based parallelism is used.
            processing_mode: The mode for memory optimization (e.g., "normal", "heavy").
        """
        optimizer_logger.info(
            "Entering PerformanceOptimizer parallel_processing_context",
            max_workers=max_workers,
            use_processes=use_processes,
            memory_optimization_mode=processing_mode
        )
        # Use its own memory_optimizer's context
        with self.memory_optimizer.optimized_processing(processing_mode):
            try:
                yield self # The context yields the PerformanceOptimizer instance
            finally:
                optimizer_logger.info("Exiting PerformanceOptimizer parallel_processing_context")

    def optimize_document_processing(
        self, 
        document_path: Path, 
        processor_func: Callable,
        **kwargs
    ) -> ProcessingResult:
        """Optimize single document processing with all available optimizations."""
        start_time = time.time()
        
        # Check cache first if available
        if self.intelligent_cache:
            cache_key = f"doc_processing_{document_path.name}_{hash(str(kwargs))}"
            cached_result = self.intelligent_cache.get(cache_key)
            if cached_result is not None:
                optimizer_logger.debug(
                    "Document processing cache hit",
                    document_path=str(document_path)
                )
                return cached_result
        
        # Process with parallel processor
        result = self.parallel_processor.process_single_document(
            document_path, 
            processor_func,
            **kwargs
        )
        
        # Cache result if available
        if self.intelligent_cache and result.success:
            self.intelligent_cache.put(cache_key, result, ttl_seconds=3600)
        
        # Update optimization stats
        with self._lock:
            self._optimization_stats['total_optimizations'] += 1
            if result.success:
                processing_speed = (document_path.stat().st_size / 1024 / 1024) / result.duration
                self._optimization_stats['speed_improvements'].append(processing_speed)
        
        optimizer_logger.info(
            "Document processing optimized",
            duration=time.time() - start_time,
            success=result.success,
            used_cache=self.intelligent_cache is not None
        )
        
        return result
    
    def optimize_batch_processing(
        self,
        tasks: List[ProcessingTask],
        processor_func: Callable,
        **kwargs
    ) -> List[ProcessingResult]:
        """Optimize batch processing with parallel processing and caching."""
        start_time = time.time()
        
        # Check for cached results
        cached_results = []
        uncached_tasks = []
        
        if self.intelligent_cache:
            for task in tasks:
                cache_key = f"batch_task_{task.task_id}_{hash(str(kwargs))}"
                cached_result = self.intelligent_cache.get(cache_key)
                if cached_result:
                    cached_results.append(cached_result)
                else:
                    uncached_tasks.append(task)
        else:
            uncached_tasks = tasks
        
        # Process uncached tasks
        if uncached_tasks:
            batch_results = self.parallel_processor.process_batch(
                uncached_tasks,
                processor_func,
                **kwargs
            )
            
            # Cache successful results
            if self.intelligent_cache:
                for result in batch_results:
                    if result.success:
                        cache_key = f"batch_task_{result.task_id}_{hash(str(kwargs))}"
                        self.intelligent_cache.put(cache_key, result, ttl_seconds=1800)
        else:
            batch_results = []
        
        # Combine cached and processed results
        all_results = cached_results + batch_results
        
        # Update optimization stats
        with self._lock:
            self._optimization_stats['total_optimizations'] += 1
            cache_hits = len(cached_results)
            total_tasks = len(tasks)
            if total_tasks > 0:
                cache_hit_rate = cache_hits / total_tasks
                self._optimization_stats['cache_hit_rates'].append(cache_hit_rate)
        
        optimizer_logger.info(
            "Batch processing optimized",
            total_tasks=len(tasks),
            cached_results=len(cached_results),
            processed_tasks=len(batch_results),
            duration=time.time() - start_time
        )
        
        return all_results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get comprehensive optimization statistics."""
        with self._lock:
            stats = {
                'total_optimizations': self._optimization_stats['total_optimizations'],
                'performance': {
                    'avg_speed_mb_per_sec': (
                        sum(self._optimization_stats['speed_improvements']) / 
                        len(self._optimization_stats['speed_improvements'])
                        if self._optimization_stats['speed_improvements'] else 0
                    ),
                    'max_speed_mb_per_sec': (
                        max(self._optimization_stats['speed_improvements'])
                        if self._optimization_stats['speed_improvements'] else 0
                    )
                },
                'caching': {
                    'avg_cache_hit_rate': (
                        sum(self._optimization_stats['cache_hit_rates']) /
                        len(self._optimization_stats['cache_hit_rates'])
                        if self._optimization_stats['cache_hit_rates'] else 0
                    ),
                    'cache_available': self.intelligent_cache is not None
                }
            }
            
            # Add component-specific stats
            if self.intelligent_cache:
                stats['cache_stats'] = self.intelligent_cache.get_comprehensive_stats()
            
            stats['parallel_processor'] = self.batch_engine.get_queue_stats()
            
            return stats
    
    def clear_caches(self):
        """Clear all caches."""
        if self.intelligent_cache:
            self.intelligent_cache.clear_all()
            optimizer_logger.info("All caches cleared")
    
    def get_memory_usage(self) -> Dict[str, Any]:
        """Get memory usage statistics."""
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            return {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'memory_percent': process.memory_percent(),
                'cache_memory_estimate': (
                    self.intelligent_cache.get_comprehensive_stats()['memory_estimate']
                    if self.intelligent_cache else None
                )
            }
        except Exception as e:
            optimizer_logger.error("Failed to get memory usage", error=str(e))
            return {}

def get_performance_optimizer() -> PerformanceOptimizer:
    """Get the global performance optimizer instance."""
    global _performance_optimizer
    if _performance_optimizer is None:
        _performance_optimizer = PerformanceOptimizer()
    return _performance_optimizer


def get_performance_stats() -> Dict[str, Any]:
    """Get performance optimizer statistics."""
    return get_performance_optimizer().get_optimization_stats()


# Convenience functions 