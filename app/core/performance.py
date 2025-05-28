import time
import psutil
import functools
from typing import Dict, Any, Optional, Callable
from pathlib import Path
from app.core.logging import StructuredLogger

performance_logger = StructuredLogger("anonympdf.performance")


class PerformanceMonitor:
    """Monitor and track performance metrics for PDF processing operations."""

    def __init__(self):
        self.metrics = {}
        self.process = psutil.Process()

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

            # Store metrics
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
        if operation_name not in self.metrics or not self.metrics[operation_name]:
            return None

        operations = self.metrics[operation_name]
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
        stats = {}
        for operation_name in self.metrics:
            stats[operation_name] = self.get_operation_stats(operation_name)
        return stats

    def clear_metrics(self):
        """Clear all stored metrics."""
        self.metrics.clear()
        performance_logger.info("Performance metrics cleared")


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


class FileProcessingMetrics:
    """Specialized metrics for file processing operations."""

    def __init__(self):
        self.monitor = PerformanceMonitor()

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

    def get_throughput_stats(self) -> Dict[str, Any]:
        """Get throughput statistics for file processing."""
        stats = self.monitor.get_all_stats()
        
        # Add throughput analysis
        for operation_name, operation_stats in stats.items():
            if operation_stats and operation_name in self.monitor.metrics:
                operations = self.monitor.metrics[operation_name]
                throughputs = [
                    op.get("throughput_mb_per_sec", 0) 
                    for op in operations 
                    if "throughput_mb_per_sec" in op
                ]
                
                if throughputs:
                    operation_stats["avg_throughput_mb_per_sec"] = sum(throughputs) / len(throughputs)
                    operation_stats["min_throughput_mb_per_sec"] = min(throughputs)
                    operation_stats["max_throughput_mb_per_sec"] = max(throughputs)

        return stats


# Global performance monitor instance
global_performance_monitor = PerformanceMonitor()
file_processing_metrics = FileProcessingMetrics()


def get_performance_report() -> Dict[str, Any]:
    """Get a comprehensive performance report."""
    return {
        "system_metrics": global_performance_monitor.get_system_metrics(),
        "operation_stats": global_performance_monitor.get_all_stats(),
        "file_processing_stats": file_processing_metrics.get_throughput_stats(),
        "report_timestamp": time.time(),
    } 