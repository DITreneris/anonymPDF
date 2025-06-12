"""
Memory Utilities - Priority 3 Session 3
Provides global convenience functions and decorators for memory optimization.
This module helps break circular dependencies.
"""
from typing import Dict, Any, Callable

# To break circular dependency, MemoryOptimizer itself is not imported at module level here.
# Instead, get_memory_optimizer will be imported from its defining module.
from app.core.memory_optimizer import get_memory_optimizer

# Convenience functions
def optimize_memory() -> Dict[str, Any]:
    """Perform memory optimization."""
    return get_memory_optimizer().optimize_memory()


def get_memory_stats() -> Dict[str, Any]:
    """Get memory optimization statistics."""
    return get_memory_optimizer().get_optimization_stats()


def start_memory_monitoring():
    """Start memory monitoring."""
    get_memory_optimizer().start_optimization()


def stop_memory_monitoring():
    """Stop memory monitoring."""
    get_memory_optimizer().stop_optimization()


# Memory optimization decorator
def memory_optimized(processing_mode: str = "normal"):
    """Decorator for memory-optimized function execution."""
    def decorator(func: Callable) -> Callable:
        # Ensure the wrapper has a unique name to avoid clashes if the decorator is used multiple times
        # or if the decorated function is introspected.
        # functools.wraps can also be used if preserving original function metadata is critical.
        # For now, a simple wrapper is sufficient.
        def wrapper(*args, **kwargs):
            optimizer = get_memory_optimizer()
            with optimizer.optimized_processing(processing_mode):
                return func(*args, **kwargs)
        # Try to preserve the original function's name and docstring for easier debugging.
        try:
            wrapper.__name__ = func.__name__
            wrapper.__doc__ = func.__doc__
        except AttributeError:
            pass # Some callables might not have these attributes
        return wrapper
    return decorator 