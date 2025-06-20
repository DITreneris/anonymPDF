import time
import psutil
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
from contextlib import contextmanager
from enum import Enum
from collections import defaultdict

from app.core.logging import StructuredLogger

performance_logger = StructuredLogger("anonympdf.performance")


class ProcessingMode(Enum):
    """Processing modes for the performance optimized processor."""
    SEQUENTIAL = "sequential"
    PARALLEL_THREADS = "parallel_threads"
    PARALLEL_PROCESSES = "parallel_processes"


class PerformanceMonitor:
    """Monitor and track performance metrics for PDF processing operations."""

    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process()
        self._lock = threading.Lock()

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,
            "vms_mb": memory_info.vms / 1024 / 1024,
        }

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        memory = self.get_memory_usage()
        cpu = self.get_cpu_usage()
        system_memory = psutil.virtual_memory()
        
        return {
            "process_memory_mb": memory["rss_mb"],
            "process_virtual_memory_mb": memory["vms_mb"],
            "process_cpu_percent": cpu,
            "system_memory_percent": system_memory.percent,
            "system_memory_available_mb": system_memory.available / 1024 / 1024,
            "timestamp": time.time(),
        }

    def track_operation(self, operation_name: str, **kwargs) -> Dict[str, Any]:
        """Track performance metrics for a specific operation."""
        start_time = time.time()
        start_memory = self.get_memory_usage()
        
        def end_tracking():
            end_time = time.time()
            end_memory = self.get_memory_usage()
            
            metrics = {
                "operation": operation_name,
                "duration_seconds": end_time - start_time,
                "memory_delta_mb": end_memory["rss_mb"] - start_memory["rss_mb"],
                "timestamp": start_time,
                **kwargs,
            }

            with self._lock:
                if operation_name not in self.metrics:
                    self.metrics[operation_name] = []
                self.metrics[operation_name].append(metrics)

            performance_logger.info(
                f"Performance tracking: {operation_name}",
                duration=metrics["duration_seconds"],
                memory_delta=metrics["memory_delta_mb"],
            )
            return metrics

        return {"end_tracking": end_tracking}

    def _get_operation_stats_nolock(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get statistical summary for an operation without acquiring a lock."""
        if operation_name not in self.metrics or not self.metrics[operation_name]:
            return None
        operations = self.metrics[operation_name].copy()
        
        durations = [op["duration_seconds"] for op in operations]
        return {
            "count": len(operations),
            "avg_duration_seconds": sum(durations) / len(durations),
        }

    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get statistical summary for an operation."""
        with self._lock:
            return self._get_operation_stats_nolock(operation_name)

    def get_all_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all tracked operations."""
        with self._lock:
            return {name: self._get_operation_stats_nolock(name) for name in self.metrics.keys()}

    def clear_metrics(self):
        """Clear all stored metrics."""
        with self._lock:
            self.metrics.clear()


class ParallelPerformanceTracker:
    """Enhanced performance tracking for parallel operations."""
    
    def __init__(self):
        self._parallel_sessions = {}
        self._session_lock = threading.Lock()
    
    def start_parallel_session(self, session_id: str, worker_count: int) -> Dict[str, Any]:
        """Start tracking a parallel processing session."""
        session_data = {
            'session_id': session_id,
            'start_time': time.time(),
            'worker_count': worker_count,
            'tasks_completed': 0,
            'tasks_failed': 0,
            'lock': threading.Lock()
        }
        with self._session_lock:
            self._parallel_sessions[session_id] = session_data
        return session_data
    
    def track_task_completion(self, session_id: str, success: bool = True):
        """Track completion of a parallel task."""
        with self._session_lock:
            if session_id in self._parallel_sessions:
                with self._parallel_sessions[session_id]['lock']:
                    if success:
                        self._parallel_sessions[session_id]['tasks_completed'] += 1
                    else:
                        self._parallel_sessions[session_id]['tasks_failed'] += 1
    
    def end_parallel_session(self, session_id: str) -> Dict[str, Any]:
        """End parallel session and calculate performance metrics."""
        with self._session_lock:
            if session_id not in self._parallel_sessions:
                return {}
            session = self._parallel_sessions.pop(session_id)
        
        duration = time.time() - session['start_time']
        total_tasks = session['tasks_completed'] + session['tasks_failed']
        
        return {
            'session_duration_seconds': duration,
            'tasks_completed': session['tasks_completed'],
            'tasks_failed': session['tasks_failed'],
            'throughput_tasks_per_second': total_tasks / duration if duration > 0 else 0,
        }


class FileProcessingMetrics:
    """Track file processing metrics and throughput."""
    
    def __init__(self):
        self._file_metrics = defaultdict(list)
        self._active_tracking = {}
        self._lock = threading.Lock()
    
    def track_file_processing(self, file_path: Path, file_size_bytes: int) -> Dict[str, Callable]:
        """
        Start tracking file processing and return tracking context.
        
        Args:
            file_path: Path to the file being processed
            file_size_bytes: Size of the file in bytes
            
        Returns:
            Dictionary with 'end_tracking' callable
        """
        # Convert to Path if string
        if isinstance(file_path, str):
            file_path = Path(file_path)
            
        session_id = str(file_path)
        start_time = time.time()
        start_memory = psutil.virtual_memory().used
        start_cpu = psutil.cpu_percent()
        
        # Store session data
        session_data = {
            'file_path': file_path,
            'file_size_bytes': file_size_bytes,
            'file_size_mb': file_size_bytes / (1024 * 1024),
            'start_time': start_time,
            'start_memory': start_memory,
            'start_cpu': start_cpu,
            'file_extension': file_path.suffix.lower()
        }
        
        with self._lock:
            self._active_tracking[session_id] = session_data
        
        def end_tracking() -> Dict[str, Any]:
            """End the tracking session and return metrics."""
            end_time = time.time()
            end_memory = psutil.virtual_memory().used
            end_cpu = psutil.cpu_percent()
            
            with self._lock:
                if session_id in self._active_tracking:
                    session = self._active_tracking.pop(session_id)
                else:
                    # Session not found, use current data
                    session = session_data
            
            duration = end_time - session['start_time']
            memory_delta = (end_memory - session['start_memory']) / (1024 * 1024)  # MB
            throughput = session['file_size_mb'] / duration if duration > 0 else 0
            
            metrics = {
                'file_size_mb': session['file_size_mb'],
                'file_size_bytes': session['file_size_bytes'],
                'duration_seconds': duration,
                'file_extension': session['file_extension'],
                'cpu_start_percent': session['start_cpu'],
                'cpu_end_percent': end_cpu,
                'memory_delta': memory_delta,
                'throughput': throughput
            }
            
            # Store in metrics history
            operation = f"pdf_processing"
            with self._lock:
                self._file_metrics[operation].append(metrics)
            
            # Log the completion
            performance_logger.info(
                f"File processing completed: {operation}",
                file_size_mb=metrics['file_size_mb'],
                duration=metrics['duration_seconds'],
                throughput=metrics['throughput'],
                memory_delta=metrics['memory_delta']
            )
            
            return metrics
        
        return {'end_tracking': end_tracking}
    
    def track_file_processing_legacy(self, operation: str, file_size_bytes: int, duration: float):
        """Backwards compatibility for tracking."""
        throughput = (file_size_bytes / (1024 * 1024)) / duration if duration > 0 else 0
        with self._lock:
            self._file_metrics[operation].append({'throughput': throughput, 'duration': duration})
    
    def get_throughput_stats(self, operation: str) -> Dict[str, Any]:
        """Calculate throughput statistics."""
        with self._lock:
            if not self._file_metrics.get(operation):
                return {"avg_throughput_mbps": 0, "count": 0}
            throughputs = [m['throughput'] for m in self._file_metrics[operation]]
            return {"avg_throughput_mbps": sum(throughputs) / len(throughputs), "count": len(throughputs)}
    
    def process_batch(self, file_list) -> Dict[str, Any]:
        """Process a batch of files and get performance stats."""
        start_time = time.time()
        # This is a placeholder for batch processing logic
        time.sleep(0.1 * len(file_list))  # Simulate work
        end_time = time.time()
        
        total_size_mb = sum(f.stat().st_size for f in file_list) / (1024 * 1024)
        duration = end_time - start_time
        
        return {
            'batch_size': len(file_list),
            'total_size_mb': total_size_mb,
            'duration_seconds': duration,
            'throughput_mbps': total_size_mb / duration if duration > 0 else 0
        }

# Global instance for file processing metrics
file_processing_metrics = FileProcessingMetrics()


class PerformanceOptimizedProcessor:
    """Orchestrates high-performance processing using parallelization."""

    def __init__(self, max_workers=4, chunk_size=1000, timeout=30, enable_monitoring=True):
        self.max_workers = max_workers
        self.chunk_size = chunk_size
        self.timeout = timeout
        self.enable_monitoring = enable_monitoring
        self._load_optimizers()

    def _load_optimizers(self):
        """Load any optimizers or monitors."""
        if self.enable_monitoring:
            self.monitor = PerformanceMonitor()
            self.parallel_tracker = ParallelPerformanceTracker()

    @contextmanager
    def optimized_processing_session(self, **kwargs):
        """Context manager for an optimized processing session."""
        if self.enable_monitoring:
            tracking_context = self.monitor.track_operation(**kwargs)
            yield
            tracking_context['end_tracking']()
        else:
            yield

    def process_files_parallel(
        self, files: List[Path], process_function: Callable, max_workers: int = None, use_processes: bool = False
    ) -> List[Any]:
        """Process a list of files in parallel using threads or processes."""
        workers = max_workers or self.max_workers
        Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        
        results = []
        with Executor(max_workers=workers) as executor:
            future_to_file = {executor.submit(process_function, f): f for f in files}
            for future in as_completed(future_to_file):
                try:
                    results.append(future.result())
                except Exception as e:
                    performance_logger.error(f"Parallel processing failed for {future_to_file[future]}", exc_info=e)
        return results
    
    def process_files(self, file_list):
        """Placeholder for a generic file processing method."""
        # Simple sequential processing
        results = []
        for file_path in file_list:
            # Simulate processing
            results.append({"file": file_path, "status": "processed"})
        return results


def performance_monitor(operation_name: str = None):
    """Decorator to monitor the performance of a function."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            with get_optimized_processor().optimized_processing_session(operation_name=op_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator

@functools.lru_cache(maxsize=1)
def get_optimized_processor() -> PerformanceOptimizedProcessor:
    """Get a singleton instance of the performance processor."""
    return PerformanceOptimizedProcessor()


def get_performance_report() -> Dict[str, Any]:
    """Get a comprehensive performance report."""
    processor = get_optimized_processor()
    if processor.enable_monitoring:
        return processor.monitor.get_all_stats()
    return {"status": "monitoring_disabled"}

@functools.lru_cache(maxsize=1)
def get_parallel_processor() -> PerformanceOptimizedProcessor:
    return PerformanceOptimizedProcessor()

# Export all the classes and instances
__all__ = [
    'FileProcessingMetrics',
    'PerformanceMonitor', 
    'PerformanceOptimizedProcessor',
    'ParallelPerformanceTracker',
    'file_processing_metrics',  # This was missing
    'performance_monitor',
    'get_optimized_processor',
    'get_parallel_processor', 
    'get_performance_report',
    'performance_logger'
]