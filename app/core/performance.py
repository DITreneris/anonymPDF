import time
import psutil
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from typing import Dict, Any, Optional, Callable, List, Union
from pathlib import Path
from contextlib import contextmanager

from app.core.logging import StructuredLogger

performance_logger = StructuredLogger("anonympdf.performance")


class PerformanceMonitor:
    """Monitor and track performance metrics for PDF processing operations."""

    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process()
        self._lock = threading.Lock()  # Thread safety for parallel operations

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in MB."""
        memory_info = self.process.memory_info()
        return {
            "rss_mb": memory_info.rss / 1024 / 1024,  # Resident Set Size
            "vms_mb": memory_info.vms / 1024 / 1024,  # Virtual Memory Size
        }

    def get_cpu_usage(self) -> float:
        """Get current CPU usage percentage."""
        return self.process.cpu_percent()

    def get_system_metrics(self) -> Dict[str, Any]:
        """Get comprehensive system metrics."""
        memory = self.get_memory_usage()
        cpu = self.get_cpu_usage()
        
        # Get system-wide memory info
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
        start_cpu = self.get_cpu_usage()

        def end_tracking():
            end_time = time.time()
            end_memory = self.get_memory_usage()
            end_cpu = self.get_cpu_usage()

            metrics = {
                "operation": operation_name,
                "duration_seconds": end_time - start_time,
                "memory_start_mb": start_memory["rss_mb"],
                "memory_end_mb": end_memory["rss_mb"],
                "memory_delta_mb": end_memory["rss_mb"] - start_memory["rss_mb"],
                "cpu_start_percent": start_cpu,
                "cpu_end_percent": end_cpu,
                "timestamp": start_time,
                **kwargs,
            }

            # Store metrics with thread safety
            with self._lock:
                if operation_name not in self.metrics:
                    self.metrics[operation_name] = []
                self.metrics[operation_name].append(metrics)

            # Log performance metrics
            performance_logger.info(
                f"Performance tracking: {operation_name}",
                duration=metrics["duration_seconds"],
                memory_delta=metrics["memory_delta_mb"],
                **kwargs,
            )

            return metrics

        return {"end_tracking": end_tracking, "start_time": start_time}

    def get_operation_stats(self, operation_name: str) -> Optional[Dict[str, Any]]:
        """Get statistical summary for an operation."""
        with self._lock:
            if operation_name not in self.metrics or not self.metrics[operation_name]:
                return None

            operations = self.metrics[operation_name].copy()
        
        durations = [op["duration_seconds"] for op in operations]
        memory_deltas = [op["memory_delta_mb"] for op in operations]

        return {
            "operation": operation_name,
            "count": len(operations),
            "avg_duration_seconds": sum(durations) / len(durations),
            "min_duration_seconds": min(durations),
            "max_duration_seconds": max(durations),
            "avg_memory_delta_mb": sum(memory_deltas) / len(memory_deltas),
            "min_memory_delta_mb": min(memory_deltas),
            "max_memory_delta_mb": max(memory_deltas),
            "total_duration_seconds": sum(durations),
        }

    def get_all_stats(self) -> Dict[str, Any]:
        """Get performance statistics for all tracked operations."""
        with self._lock:
            operation_names = list(self.metrics.keys())
        
        stats = {}
        for operation_name in operation_names:
            stats[operation_name] = self.get_operation_stats(operation_name)
        return stats

    def clear_metrics(self):
        """Clear all stored metrics."""
        with self._lock:
            self.metrics.clear()
        performance_logger.info("Performance metrics cleared")


class ParallelPerformanceTracker:
    """Enhanced performance tracking for parallel operations."""
    
    def __init__(self):
        self.monitor = PerformanceMonitor()
        self._parallel_sessions = {}
        self._session_lock = threading.Lock()
    
    def start_parallel_session(self, session_id: str, worker_count: int = None) -> Dict[str, Any]:
        """Start tracking a parallel processing session."""
        session_data = {
            'session_id': session_id,
            'start_time': time.time(),
            'worker_count': worker_count or psutil.cpu_count(),
            'tasks_completed': 0,
            'tasks_failed': 0,
            'total_processing_time': 0.0,
            'start_memory': self.monitor.get_memory_usage(),
            'start_cpu': self.monitor.get_cpu_usage(),
            'lock': threading.Lock()
        }
        
        with self._session_lock:
            self._parallel_sessions[session_id] = session_data
        
        performance_logger.info(
            "Parallel session started",
            session_id=session_id,
            worker_count=session_data['worker_count']
        )
        
        return session_data
    
    def track_task_completion(self, session_id: str, task_duration: float, success: bool = True):
        """Track completion of a parallel task."""
        with self._session_lock:
            if session_id not in self._parallel_sessions:
                return
            
            session = self._parallel_sessions[session_id]
            
        with session['lock']:
            if success:
                session['tasks_completed'] += 1
            else:
                session['tasks_failed'] += 1
            session['total_processing_time'] += task_duration
    
    def end_parallel_session(self, session_id: str) -> Dict[str, Any]:
        """End parallel session and calculate performance metrics."""
        with self._session_lock:
            if session_id not in self._parallel_sessions:
                return {}
            
            session = self._parallel_sessions.pop(session_id)
        
        end_time = time.time()
        session_duration = end_time - session['start_time']
        end_memory = self.monitor.get_memory_usage()
        end_cpu = self.monitor.get_cpu_usage()
        
        total_tasks = session['tasks_completed'] + session['tasks_failed']
        
        metrics = {
            'session_id': session_id,
            'session_duration_seconds': session_duration,
            'worker_count': session['worker_count'],
            'tasks_completed': session['tasks_completed'],
            'tasks_failed': session['tasks_failed'],
            'total_tasks': total_tasks,
            'success_rate': session['tasks_completed'] / max(1, total_tasks),
            'total_processing_time': session['total_processing_time'],
            'parallel_efficiency': session['total_processing_time'] / (session_duration * session['worker_count']) if session_duration > 0 else 0,
            'throughput_tasks_per_second': total_tasks / session_duration if session_duration > 0 else 0,
            'memory_delta_mb': end_memory['rss_mb'] - session['start_memory']['rss_mb'],
            'avg_task_duration': session['total_processing_time'] / max(1, session['tasks_completed'])
        }
        
        # Track in main monitor
        self.monitor.track_operation(f"parallel_session_{session_id}", **metrics)
        
        performance_logger.info(
            "Parallel session completed",
            **metrics
        )
        
        return metrics


class FileProcessingMetrics:
    """Specialized metrics for file processing operations with parallel support."""

    def __init__(self):
        self.monitor = PerformanceMonitor()
        self.parallel_tracker = ParallelPerformanceTracker()

    def track_file_processing(
        self, file_path: Path, file_size_bytes: int, operation: str = "pdf_processing"
    ):
        """Track file processing with file-specific metrics."""
        file_size_mb = file_size_bytes / 1024 / 1024

        tracking_kwargs = {
            "file_size_mb": file_size_mb,
            "file_extension": file_path.suffix.lower(),
            "operation_type": operation,
        }

        tracking = self.monitor.track_operation(operation, **tracking_kwargs)

        def end_with_throughput():
            metrics = tracking["end_tracking"]()
            
            # Calculate throughput
            if metrics["duration_seconds"] > 0:
                throughput_mb_per_sec = file_size_mb / metrics["duration_seconds"]
                metrics["throughput_mb_per_sec"] = throughput_mb_per_sec
                
                performance_logger.info(
                    f"File processing completed: {operation}",
                    file_size_mb=file_size_mb,
                    duration=metrics["duration_seconds"],
                    throughput=throughput_mb_per_sec,
                    memory_delta=metrics["memory_delta_mb"],
                )

            return metrics

        return {"end_tracking": end_with_throughput, "start_time": tracking["start_time"]}
    
    def track_batch_processing(
        self, 
        files: List[Path], 
        operation: str = "batch_pdf_processing",
        parallel: bool = True,
        max_workers: int = None
    ) -> Dict[str, Any]:
        """Track batch file processing with optional parallelization."""
        if not files:
            return {}
        
        total_size_bytes = sum(f.stat().st_size for f in files if f.exists())
        total_size_mb = total_size_bytes / 1024 / 1024
        
        session_id = f"{operation}_{int(time.time())}"
        
        if parallel:
            # Use parallel tracking
            session = self.parallel_tracker.start_parallel_session(
                session_id, 
                max_workers or min(len(files), psutil.cpu_count())
            )
            
            tracking_kwargs = {
                "file_count": len(files),
                "total_size_mb": total_size_mb,
                "operation_type": operation,
                "parallel": True,
                "max_workers": session['worker_count']
            }
        else:
            # Use regular tracking
            tracking_kwargs = {
                "file_count": len(files),
                "total_size_mb": total_size_mb,
                "operation_type": operation,
                "parallel": False
            }
        
        tracking = self.monitor.track_operation(operation, **tracking_kwargs)
        
        def end_batch_tracking():
            metrics = tracking["end_tracking"]()
            
            if parallel:
                # Get parallel session metrics
                parallel_metrics = self.parallel_tracker.end_parallel_session(session_id)
                metrics.update(parallel_metrics)
            
            # Calculate batch throughput
            if metrics["duration_seconds"] > 0:
                metrics["batch_throughput_mb_per_sec"] = total_size_mb / metrics["duration_seconds"]
                metrics["batch_throughput_files_per_sec"] = len(files) / metrics["duration_seconds"]
            
            performance_logger.info(
                f"Batch processing completed: {operation}",
                file_count=len(files),
                total_size_mb=total_size_mb,
                duration=metrics["duration_seconds"],
                parallel=parallel,
                **({k: v for k, v in metrics.items() if k.startswith(('throughput', 'efficiency'))}),
            )
            
            return metrics
        
        return {
            "end_tracking": end_batch_tracking, 
            "start_time": tracking["start_time"],
            "session_id": session_id if parallel else None,
            "parallel_tracker": self.parallel_tracker if parallel else None
        }

    def get_throughput_stats(self) -> Dict[str, Any]:
        """Get throughput statistics for file processing."""
        stats = self.monitor.get_all_stats()
        
        # Add throughput analysis
        for operation_name, operation_stats in stats.items():
            if operation_stats and operation_name in self.monitor.metrics:
                with self.monitor._lock:
                    operations = self.monitor.metrics[operation_name].copy()
                
                throughputs = [
                    op.get("throughput_mb_per_sec", 0) 
                    for op in operations 
                    if "throughput_mb_per_sec" in op
                ]
                
                batch_throughputs = [
                    op.get("batch_throughput_mb_per_sec", 0)
                    for op in operations 
                    if "batch_throughput_mb_per_sec" in op
                ]
                
                if throughputs:
                    operation_stats["avg_throughput_mb_per_sec"] = sum(throughputs) / len(throughputs)
                    operation_stats["min_throughput_mb_per_sec"] = min(throughputs)
                    operation_stats["max_throughput_mb_per_sec"] = max(throughputs)
                
                if batch_throughputs:
                    operation_stats["avg_batch_throughput_mb_per_sec"] = sum(batch_throughputs) / len(batch_throughputs)
                    operation_stats["min_batch_throughput_mb_per_sec"] = min(batch_throughputs)
                    operation_stats["max_batch_throughput_mb_per_sec"] = max(batch_throughputs)

        return stats


class PerformanceOptimizedProcessor:
    """Processor that integrates with Performance Optimizer, Cache, and Memory Optimizer."""
    
    def __init__(self):
        self.file_metrics = FileProcessingMetrics()
        self._performance_optimizer = None
        self._intelligent_cache = None
        self._memory_optimizer = None
        
        # Lazy loading to avoid circular imports
        self._load_optimizers()
    
    def _load_optimizers(self):
        """Lazy load optimizers to avoid circular imports."""
        try:
            from app.core.performance_optimizer import get_performance_optimizer
            from app.core.intelligent_cache import get_intelligent_cache
            from app.core.memory_optimizer import get_memory_optimizer
            
            self._performance_optimizer = get_performance_optimizer()
            self._intelligent_cache = get_intelligent_cache()
            self._memory_optimizer = get_memory_optimizer()
            
            performance_logger.info("Performance optimizers loaded successfully")
        except ImportError as e:
            performance_logger.warning(
                "Some performance optimizers not available",
                error=str(e)
            )
    
    @contextmanager
    def optimized_processing_session(self, processing_mode: str = "normal"):
        """Context manager for fully optimized processing session."""
        session_start = time.time()
        
        # Start memory optimization if available
        if self._memory_optimizer:
            self._memory_optimizer.start_optimization()
        
        try:
            if self._memory_optimizer:
                with self._memory_optimizer.optimized_processing(processing_mode):
                    yield self
            else:
                yield self
        finally:
            # Stop memory optimization
            if self._memory_optimizer:
                self._memory_optimizer.stop_optimization()
            
            session_duration = time.time() - session_start
            performance_logger.info(
                "Optimized processing session completed",
                duration=session_duration,
                mode=processing_mode
            )
    
    def process_files_parallel(
        self,
        files: List[Path],
        process_function: Callable,
        max_workers: int = None,
        use_processes: bool = False,
        chunk_size: int = None
    ) -> List[Any]:
        """Process files in parallel with full optimization."""
        if not files:
            return []
        
        max_workers = max_workers or min(len(files), psutil.cpu_count())
        
        # Start batch tracking
        batch_tracking = self.file_metrics.track_batch_processing(
            files, 
            "parallel_file_processing",
            parallel=True,
            max_workers=max_workers
        )
        
        results = []
        
        try:
            # Use performance optimizer if available
            if self._performance_optimizer:
                with self._performance_optimizer.parallel_processing_context(max_workers, use_processes):
                    results = self._execute_parallel_processing(
                        files, process_function, max_workers, use_processes, chunk_size, batch_tracking
                    )
            else:
                results = self._execute_parallel_processing(
                    files, process_function, max_workers, use_processes, chunk_size, batch_tracking
                )
        
        finally:
            # End batch tracking
            batch_metrics = batch_tracking["end_tracking"]()
            
            performance_logger.info(
                "Parallel file processing completed",
                files_processed=len(files),
                results_count=len(results),
                **{k: v for k, v in batch_metrics.items() if k.startswith(('duration', 'throughput', 'efficiency'))}
            )
        
        return results
    
    def _execute_parallel_processing(
        self,
        files: List[Path],
        process_function: Callable,
        max_workers: int,
        use_processes: bool,
        chunk_size: int,
        batch_tracking: Dict[str, Any]
    ) -> List[Any]:
        """Execute the actual parallel processing."""
        ExecutorClass = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
        results = []
        
        with ExecutorClass(max_workers=max_workers) as executor:
            # Submit tasks
            future_to_file = {
                executor.submit(self._process_single_file, file_path, process_function): file_path
                for file_path in files
            }
            
            # Collect results
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                task_start = time.time()
                
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Track successful task
                    if batch_tracking.get("parallel_tracker"):
                        batch_tracking["parallel_tracker"].track_task_completion(
                            batch_tracking["session_id"],
                            time.time() - task_start,
                            success=True
                        )
                
                except Exception as e:
                    performance_logger.error(
                        "Error processing file in parallel",
                        file=str(file_path),
                        error=str(e)
                    )
                    
                    # Track failed task
                    if batch_tracking.get("parallel_tracker"):
                        batch_tracking["parallel_tracker"].track_task_completion(
                            batch_tracking["session_id"],
                            time.time() - task_start,
                            success=False
                        )
        
        return results
    
    def _process_single_file(self, file_path: Path, process_function: Callable) -> Any:
        """Process a single file with caching and error handling."""
        # Check cache if available
        if self._intelligent_cache:
            cache_key = f"file_processing_{file_path.stat().st_mtime}_{file_path.stat().st_size}"
            cached_result = self._intelligent_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Process file
        file_size = file_path.stat().st_size
        tracking = self.file_metrics.track_file_processing(file_path, file_size)
        
        try:
            result = process_function(file_path)
            
            # Cache result if available
            if self._intelligent_cache:
                self._intelligent_cache.put(cache_key, result, ttl=3600)  # 1 hour TTL
            
            return result
            
        finally:
            tracking["end_tracking"]()


def performance_monitor(operation_name: str = None, log_args: bool = False):
    """Decorator to monitor function performance."""

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Use function name if operation_name not provided
            op_name = operation_name or f"{func.__module__}.{func.__name__}"

            # Prepare tracking kwargs
            tracking_kwargs = {"function": func.__name__}
            if log_args:
                tracking_kwargs["args_count"] = len(args)
                tracking_kwargs["kwargs_count"] = len(kwargs)

            # Start tracking using global monitor
            tracking = global_performance_monitor.track_operation(op_name, **tracking_kwargs)

            try:
                # Execute function
                result = func(*args, **kwargs)
                tracking_kwargs["success"] = True
                return result
            except Exception as e:
                tracking_kwargs["success"] = False
                tracking_kwargs["error"] = str(e)
                raise
            finally:
                # End tracking
                tracking["end_tracking"]()

        return wrapper

    return decorator


# Global instances
global_performance_monitor = PerformanceMonitor()
file_processing_metrics = FileProcessingMetrics()

_optimized_processor: Optional[PerformanceOptimizedProcessor] = None
_optimized_processor_lock = threading.Lock()

def get_optimized_processor() -> PerformanceOptimizedProcessor:
    """Get the global PerformanceOptimizedProcessor instance, lazy loading it."""
    global _optimized_processor
    if _optimized_processor is None:
        with _optimized_processor_lock:
            if _optimized_processor is None:
                _optimized_processor = PerformanceOptimizedProcessor()
    return _optimized_processor


def get_performance_report() -> Dict[str, Any]:
    """Get a comprehensive performance report."""
    report = {
        "system_metrics": global_performance_monitor.get_system_metrics(),
        "operation_stats": global_performance_monitor.get_all_stats(),
        "file_processing_stats": file_processing_metrics.get_throughput_stats(),
        "report_timestamp": time.time(),
    }
    
    try:
        from app.core.performance_optimizer import get_performance_stats
        from app.core.intelligent_cache import get_cache_stats  
        from app.core.memory_utils import get_memory_stats # Corrected import source
        
        report["performance_optimizer_stats"] = get_performance_stats()
        report["cache_stats"] = get_cache_stats()
        report["memory_optimizer_stats"] = get_memory_stats()
        
    except ImportError:
        performance_logger.debug("Some optimizer stats not available for report")
    
    return report


def get_parallel_processor() -> PerformanceOptimizedProcessor:
    """Get the global optimized processor instance."""
    return get_optimized_processor() 