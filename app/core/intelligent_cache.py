"""
Intelligent Cache - Priority 3 Session 3
Implements smart caching with LRU, TTL, and pattern-based caching
for 60% cache hit rate and 30% memory reduction.
"""

import time
import hashlib
import threading
import pickle
from typing import Dict, Any, Optional, Tuple, List, Union, Callable
from collections import OrderedDict
from dataclasses import dataclass, field
from enum import Enum
import weakref
import gc
from pathlib import Path

from app.core.logging import StructuredLogger
from app.core.config_manager import get_config
from app.core.performance import PerformanceMonitor

cache_logger = StructuredLogger("anonympdf.intelligent_cache")


class CachePolicy(Enum):
    """Cache eviction policies."""
    LRU = "lru"  # Least Recently Used
    LFU = "lfu"  # Least Frequently Used
    TTL = "ttl"  # Time To Live
    ADAPTIVE = "adaptive"  # Adaptive based on usage patterns


@dataclass
class CacheEntry:
    """Represents a cache entry with metadata."""
    value: Any
    created_at: float
    last_accessed: float
    access_count: int = 0
    ttl_seconds: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def is_expired(self) -> bool:
        """Check if the cache entry has expired."""
        if self.ttl_seconds is None:
            return False
        return time.time() - self.created_at > self.ttl_seconds
    
    def touch(self):
        """Update access statistics."""
        self.last_accessed = time.time()
        self.access_count += 1


class LRUCache:
    """Least Recently Used cache implementation."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = threading.RLock()
        self._stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0,
            'size': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache."""
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                
                # Check if expired
                if entry.is_expired:
                    del self._cache[key]
                    self._stats['misses'] += 1
                    return None
                
                # Move to end (most recently used)
                self._cache.move_to_end(key)
                entry.touch()
                self._stats['hits'] += 1
                
                cache_logger.debug(
                    "Cache hit",
                    key=key,
                    access_count=entry.access_count
                )
                
                return entry.value
            
            self._stats['misses'] += 1
            return None
    
    def put(self, key: str, value: Any, ttl_seconds: Optional[float] = None, **metadata) -> bool:
        """Put value in cache."""
        with self._lock:
            try:
                # Create cache entry
                entry = CacheEntry(
                    value=value,
                    created_at=time.time(),
                    last_accessed=time.time(),
                    ttl_seconds=ttl_seconds,
                    metadata=metadata
                )
                
                # If key exists, update it
                if key in self._cache:
                    self._cache[key] = entry
                    self._cache.move_to_end(key)
                else:
                    # Add new entry
                    self._cache[key] = entry
                    
                    # Check if we need to evict
                    if len(self._cache) > self.max_size:
                        self._evict_oldest()
                
                self._stats['size'] = len(self._cache)
                
                cache_logger.debug(
                    "Cache put",
                    key=key,
                    ttl_seconds=ttl_seconds,
                    cache_size=len(self._cache)
                )
                
                return True
                
            except Exception as e:
                cache_logger.error(
                    "Cache put failed",
                    key=key,
                    error=str(e)
                )
                return False
    
    def _evict_oldest(self):
        """Evict the oldest (least recently used) entry."""
        if self._cache:
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
            self._stats['evictions'] += 1
            
            cache_logger.debug(
                "Cache eviction",
                evicted_key=oldest_key,
                cache_size=len(self._cache)
            )
    
    def invalidate(self, key: str) -> bool:
        """Remove specific key from cache."""
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                self._stats['size'] = len(self._cache)
                return True
            return False
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()
            self._stats['size'] = 0
            cache_logger.info("Cache cleared")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self._stats['hits'] + self._stats['misses']
            hit_rate = self._stats['hits'] / total_requests if total_requests > 0 else 0
            
            return {
                **self._stats,
                'hit_rate': hit_rate,
                'total_requests': total_requests
            }
    
    def cleanup_expired(self) -> int:
        """Remove expired entries and return count of removed items."""
        with self._lock:
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.is_expired
            ]
            
            for key in expired_keys:
                del self._cache[key]
            
            self._stats['size'] = len(self._cache)
            
            if expired_keys:
                cache_logger.debug(
                    "Expired entries cleaned up",
                    expired_count=len(expired_keys)
                )
            
            return len(expired_keys)


class PatternCache:
    """Specialized cache for pattern-based caching (document types, features, etc.)."""
    
    def __init__(self, max_size: int = 500):
        self.lru_cache = LRUCache(max_size)
        self._pattern_stats = {}
        self._lock = threading.Lock()
    
    def _generate_pattern_key(self, content_type: str, features: Dict[str, Any]) -> str:
        """Generate a cache key based on content pattern."""
        # Create a hash of the features for consistent caching
        feature_str = str(sorted(features.items()))
        pattern_hash = hashlib.md5(feature_str.encode()).hexdigest()
        return f"pattern_{content_type}_{pattern_hash}"
    
    def get_pattern_result(self, content_type: str, features: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for a content pattern."""
        key = self._generate_pattern_key(content_type, features)
        result = self.lru_cache.get(key)
        
        if result is not None:
            with self._lock:
                if content_type not in self._pattern_stats:
                    self._pattern_stats[content_type] = {'hits': 0, 'requests': 0}
                self._pattern_stats[content_type]['hits'] += 1
                self._pattern_stats[content_type]['requests'] += 1
        else:
            with self._lock:
                if content_type not in self._pattern_stats:
                    self._pattern_stats[content_type] = {'hits': 0, 'requests': 0}
                self._pattern_stats[content_type]['requests'] += 1
        
        return result
    
    def cache_pattern_result(self, content_type: str, features: Dict[str, Any], result: Any, ttl_seconds: Optional[float] = 3600):
        """Cache a result for a content pattern."""
        key = self._generate_pattern_key(content_type, features)
        return self.lru_cache.put(
            key, result, ttl_seconds,
            content_type=content_type,
            feature_count=len(features)
        )
    
    def get_pattern_stats(self) -> Dict[str, Dict[str, float]]:
        """Get statistics per pattern type."""
        with self._lock:
            stats = {}
            for pattern_type, data in self._pattern_stats.items():
                hit_rate = data['hits'] / data['requests'] if data['requests'] > 0 else 0
                stats[pattern_type] = {
                    'hit_rate': hit_rate,
                    'requests': data['requests'],
                    'hits': data['hits']
                }
            return stats


class ModelCache:
    """Specialized cache for ML model predictions and features."""
    
    def __init__(self, max_size: int = 200):
        self.lru_cache = LRUCache(max_size)
        self._model_versions = {}
        self._lock = threading.Lock()
    
    def _generate_prediction_key(self, model_name: str, input_hash: str, model_version: str = "default") -> str:
        """Generate cache key for model predictions."""
        return f"model_{model_name}_{model_version}_{input_hash}"
    
    def get_prediction(self, model_name: str, input_data: Any, model_version: str = "default") -> Optional[Any]:
        """Get cached model prediction."""
        input_hash = self._hash_input(input_data)
        key = self._generate_prediction_key(model_name, input_hash, model_version)
        
        result = self.lru_cache.get(key)
        if result is not None:
            cache_logger.debug(
                "Model prediction cache hit",
                model_name=model_name,
                model_version=model_version
            )
        
        return result
    
    def cache_prediction(self, model_name: str, input_data: Any, prediction: Any, 
                        model_version: str = "default", ttl_seconds: Optional[float] = 1800):
        """Cache model prediction."""
        input_hash = self._hash_input(input_data)
        key = self._generate_prediction_key(model_name, input_hash, model_version)
        
        return self.lru_cache.put(
            key, prediction, ttl_seconds,
            model_name=model_name,
            model_version=model_version,
            input_size=len(str(input_data))
        )
    
    def invalidate_model(self, model_name: str, model_version: str = "default"):
        """Invalidate all cached predictions for a specific model version."""
        with self._lock:
            prefix = f"model_{model_name}_{model_version}_"
            keys_to_remove = [
                key for key in self.lru_cache._cache.keys()
                if key.startswith(prefix)
            ]
            
            for key in keys_to_remove:
                self.lru_cache.invalidate(key)
            
            cache_logger.info(
                "Model cache invalidated",
                model_name=model_name,
                model_version=model_version,
                invalidated_count=len(keys_to_remove)
            )
    
    def _hash_input(self, input_data: Any) -> str:
        """Create hash of input data for caching."""
        try:
            # Handle different input types
            if isinstance(input_data, (str, int, float)):
                data_str = str(input_data)
            elif isinstance(input_data, dict):
                data_str = str(sorted(input_data.items()))
            elif isinstance(input_data, (list, tuple)):
                data_str = str(input_data)
            else:
                # For complex objects, use pickle for consistent hashing
                data_str = str(pickle.dumps(input_data, protocol=pickle.HIGHEST_PROTOCOL))
            
            return hashlib.sha256(data_str.encode()).hexdigest()[:16]  # Short hash
            
        except Exception:
            # Fallback to string representation
            return hashlib.sha256(str(input_data).encode()).hexdigest()[:16]


class IntelligentCache:
    """Main intelligent caching system that combines different cache types."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initializes the IntelligentCache system.
        
        Args:
            config (Optional[Dict[str, Any]]): A configuration dictionary. 
                                              If not provided, falls back to global config.
        """
        if config is None:
            # Fallback to global config gracefully
            full_config = get_config()
            performance_config = full_config.get('performance', {})
            self.config = performance_config.get('caching', {})
        else:
            self.config = config
            
        self.policy = CachePolicy(self.config.get("policy", "lru"))
        self.last_cleanup = time.time()

        # Initialize different cache types
        max_size = self.config.get('max_size', 1000)
        self.general_cache = LRUCache(max_size)
        self.pattern_cache = PatternCache(max_size // 2)
        self.model_cache = ModelCache(max_size // 5)
        
        # Cache management
        self._cleanup_interval = 300  # 5 minutes
        self._monitor = PerformanceMonitor()
        
        # Global statistics
        self._global_stats = {
            'total_requests': 0,
            'total_hits': 0,
            'memory_saved_mb': 0.0
        }
        self._lock = threading.Lock()
        
        cache_logger.info(
            "IntelligentCache initialized",
            max_size=max_size,
            cleanup_interval=self._cleanup_interval
        )
    
    def get(self, key: str, cache_type: str = "general") -> Optional[Any]:
        """Get value from appropriate cache."""
        with self._lock:
            self._global_stats['total_requests'] += 1
        
        # Route to appropriate cache
        if cache_type == "general":
            result = self.general_cache.get(key)
        elif cache_type == "pattern":
            # This should use pattern-specific method
            return None
        elif cache_type == "model":
            # This should use model-specific method
            return None
        else:
            result = self.general_cache.get(key)
        
        if result is not None:
            with self._lock:
                self._global_stats['total_hits'] += 1
        
        # Periodic cleanup
        self._maybe_cleanup()
        
        return result
    
    def put(self, key: str, value: Any, cache_type: str = "general", **kwargs) -> bool:
        """Put value in appropriate cache."""
        success = False
        
        if cache_type == "general":
            success = self.general_cache.put(key, value, **kwargs)
        elif cache_type == "pattern":
            # Pattern cache requires special handling
            pass
        elif cache_type == "model":
            # Model cache requires special handling
            pass
        else:
            success = self.general_cache.put(key, value, **kwargs)
        
        # Estimate memory saved
        if success:
            try:
                size_estimate = len(str(value)) / 1024 / 1024  # Rough MB estimate
                with self._lock:
                    self._global_stats['memory_saved_mb'] += size_estimate
            except:
                pass
        
        return success
    
    def get_pattern_result(self, content_type: str, features: Dict[str, Any]) -> Optional[Any]:
        """Get cached result for content pattern."""
        with self._lock:
            self._global_stats['total_requests'] += 1
        
        result = self.pattern_cache.get_pattern_result(content_type, features)
        
        if result is not None:
            with self._lock:
                self._global_stats['total_hits'] += 1
        
        return result
    
    def cache_pattern_result(self, content_type: str, features: Dict[str, Any], result: Any, **kwargs):
        """Cache result for content pattern."""
        # Track the caching operation in global stats
        with self._lock:
            self._global_stats['total_requests'] += 1
        
        success = self.pattern_cache.cache_pattern_result(content_type, features, result, **kwargs)
        
        if success:
            # Estimate memory saved
            try:
                size_estimate = len(str(result)) / 1024 / 1024  # Rough MB estimate
                with self._lock:
                    self._global_stats['memory_saved_mb'] += size_estimate
            except:
                pass
        
        return success
    
    def get_model_prediction(self, model_name: str, input_data: Any, model_version: str = "default") -> Optional[Any]:
        """Get cached model prediction."""
        with self._lock:
            self._global_stats['total_requests'] += 1
        
        result = self.model_cache.get_prediction(model_name, input_data, model_version)
        
        if result is not None:
            with self._lock:
                self._global_stats['total_hits'] += 1
        
        return result
    
    def cache_model_prediction(self, model_name: str, input_data: Any, prediction: Any, model_version: str = "default", **kwargs):
        """Cache model prediction."""
        # Track the caching operation in global stats
        with self._lock:
            self._global_stats['total_requests'] += 1
        
        success = self.model_cache.cache_prediction(model_name, input_data, prediction, model_version, **kwargs)
        
        if success:
            # Estimate memory saved
            try:
                size_estimate = len(str(prediction)) / 1024 / 1024  # Rough MB estimate
                with self._lock:
                    self._global_stats['memory_saved_mb'] += size_estimate
            except:
                pass
        
        return success
    
    def invalidate_model(self, model_name: str, model_version: str = "default"):
        """Invalidate cached model predictions."""
        self.model_cache.invalidate_model(model_name, model_version)
    
    def clear_all(self):
        """Clear all caches."""
        self.general_cache.clear()
        self.pattern_cache.lru_cache.clear()
        self.model_cache.lru_cache.clear()
        
        with self._lock:
            self._global_stats = {
                'total_requests': 0,
                'total_hits': 0,
                'memory_saved_mb': 0.0
            }
        
        cache_logger.info("All caches cleared")
    
    def _maybe_cleanup(self):
        """Perform cleanup if needed."""
        current_time = time.time()
        if current_time - self.last_cleanup > self._cleanup_interval:
            self._cleanup_expired()
            self.last_cleanup = current_time
    
    def _cleanup_expired(self):
        """Clean up expired entries from all caches."""
        total_cleaned = 0
        
        total_cleaned += self.general_cache.cleanup_expired()
        total_cleaned += self.pattern_cache.lru_cache.cleanup_expired()
        total_cleaned += self.model_cache.lru_cache.cleanup_expired()
        
        if total_cleaned > 0:
            cache_logger.info(
                "Cache cleanup completed",
                expired_entries_removed=total_cleaned
            )
        
        # Force garbage collection to free memory
        gc.collect()
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive statistics from all caches."""
        with self._lock:
            global_hit_rate = (
                self._global_stats['total_hits'] / self._global_stats['total_requests']
                if self._global_stats['total_requests'] > 0 else 0
            )
        
        return {
            'global': {
                **self._global_stats,
                'hit_rate': global_hit_rate
            },
            'general_cache': self.general_cache.get_stats(),
            'pattern_cache': {
                'cache_stats': self.pattern_cache.lru_cache.get_stats(),
                'pattern_stats': self.pattern_cache.get_pattern_stats()
            },
            'model_cache': self.model_cache.lru_cache.get_stats(),
            'memory_estimate': {
                'estimated_memory_saved_mb': self._global_stats['memory_saved_mb'],
                'total_entries': (
                    len(self.general_cache._cache) +
                    len(self.pattern_cache.lru_cache._cache) +
                    len(self.model_cache.lru_cache._cache)
                )
            }
        }


_global_cache_instance: Optional[weakref.ref[IntelligentCache]] = None
_global_cache_lock = threading.Lock()


def get_intelligent_cache() -> IntelligentCache:
    """
    Factory function to get a singleton instance of the IntelligentCache.
    This ensures that the same cache instance is used throughout the application.
    """
    global _global_cache_instance
    with _global_cache_lock:
        instance = _global_cache_instance() if _global_cache_instance else None
        if instance is None:
            config = get_config().get('intelligent_cache', {})
            instance = IntelligentCache(config=config)
            _global_cache_instance = weakref.ref(instance)
            cache_logger.info("IntelligentCache singleton instance created.")
        return instance


def get_cache_stats() -> Dict[str, Any]:
    """Convenience function to get stats from the global cache."""
    return get_intelligent_cache().get_comprehensive_stats()


# Convenience functions for easy integration
def cache_get(key: str, cache_type: str = "general") -> Optional[Any]:
    """Get value from cache."""
    return get_intelligent_cache().get(key, cache_type)


def cache_put(key: str, value: Any, cache_type: str = "general", **kwargs) -> bool:
    """Put value in cache."""
    return get_intelligent_cache().put(key, value, cache_type, **kwargs)


def cache_pattern(content_type: str, features: Dict[str, Any]) -> Optional[Any]:
    """Get cached pattern result."""
    return get_intelligent_cache().get_pattern_result(content_type, features)


def cache_model_prediction(model_name: str, input_data: Any, model_version: str = "default") -> Optional[Any]:
    """Get cached model prediction."""
    return get_intelligent_cache().get_model_prediction(model_name, input_data, model_version)


# Cache decorator for easy function caching
def cache_result(cache_type: str = "general", ttl_seconds: Optional[float] = None, key_func: Optional[Callable] = None):
    """Decorator to cache function results."""
    def decorator(func: Callable) -> Callable:
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = f"{func.__module__}.{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try to get from cache
            cached_result = get_intelligent_cache().get(cache_key, cache_type)
            if cached_result is not None:
                return cached_result
            
            # Execute function and cache result
            result = func(*args, **kwargs)
            get_intelligent_cache().put(cache_key, result, cache_type, ttl_seconds=ttl_seconds)
            
            return result
        
        return wrapper
    return decorator 