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
        """Hash complex input data to a stable string."""
        try:
            # For dictionaries, sort by key to ensure hash is consistent
            if isinstance(input_data, dict):
                input_data = sorted(input_data.items())

            # Use pickle to handle complex Python objects
            # Using protocol 4 for better compatibility and efficiency
            input_bytes = pickle.dumps(input_data, protocol=4)
            return hashlib.sha256(input_bytes).hexdigest()
        except (pickle.PicklingError, TypeError) as e:
            cache_logger.warning(
                "Could not pickle input data for caching",
                error=str(e),
                input_type=type(input_data).__name__
            )
            # Fallback to a less stable but still useful representation
            return repr(input_data)


class IntelligentCache:
    """
    A thread-safe, multi-strategy cache manager.
    It orchestrates different cache types (general, patterns, models)
    and provides centralized configuration, statistics, and cleanup.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the IntelligentCache.

        Args:
            config (Optional[Dict[str, Any]]): A configuration dictionary.
                Overrides settings from the default config manager.
        """
        if config is None:
            app_config = get_config()
            self._config = app_config.get('intelligent_cache', {})
        else:
            self._config = config

        self._lock = threading.RLock()
        self._cache_sizes = self._config.get('cache_sizes', {
            'general': 1000,
            'patterns': 500,
            'models': 200
        })

        self._caches: Dict[str, Union[LRUCache, PatternCache, ModelCache]] = {}
        self._stats = {
            "total_requests": 0,
            "total_hits": 0,
            "total_puts": 0,
            "memory_saved_mb": 0.0,
            "size": 0,  # Adding overall size for thread safety test
        }
        self._last_cleanup_time = time.time()
        self.cleanup_interval = self._config.get('cleanup_interval', 300)

        # Initialize caches lazily or eagerly based on config
        if self._config.get('eager_init', False):
            with self._lock:
                self._caches['general'] = LRUCache(max_size=self._cache_sizes.get('general', 1000))
                self._caches['patterns'] = PatternCache(max_size=self._cache_sizes.get('patterns', 500))
                self._caches['models'] = ModelCache(max_size=self._cache_sizes.get('models', 200))

        cache_logger.info("IntelligentCache initialized",
                        max_size=self._cache_sizes,
                        cleanup_interval=self.cleanup_interval)

    @property
    def config(self):
        """Get the current cache configuration."""
        return self._config

    def get(self, key: str, cache_type: str = "general") -> Optional[Any]:
        """
        Get an item from a specified cache.

        Args:
            key (str): The key of the item.
            cache_type (str): The type of cache ('general', 'patterns', 'models').

        Returns:
            Optional[Any]: The cached value, or None if not found or expired.
        """
        with self._lock:
            self._stats['total_requests'] += 1
            cache = self._caches.get(cache_type)
            if not cache:
                return None  # No such cache exists

        # Lock on the specific cache instance for the actual 'get'
        # Note: PatternCache and ModelCache use LRUCache internally, which is locked.
        value = cache.get(key)

        with self._lock:
            if value is not None:
                self._stats['total_hits'] += 1
            return value

    def put(self, key: str, value: Any, cache_type: str = "general", **kwargs) -> bool:
        """
        Put an item into a specified cache.

        Args:
            key (str): The key of the item.
            value (Any): The value to cache.
            cache_type (str): The type of cache ('general', 'patterns', 'models').
            **kwargs: Additional arguments for the specific cache (e.g., ttl_seconds).

        Returns:
            bool: True if the item was successfully cached, False otherwise.
        """
        with self._lock:
            self._maybe_cleanup()

            if cache_type not in self._caches:
                # Use a more specific factory pattern if more types are added
                if cache_type == 'patterns':
                    self._caches[cache_type] = PatternCache(max_size=self._cache_sizes.get(cache_type, 500))
                elif cache_type == 'models':
                    self._caches[cache_type] = ModelCache(max_size=self._cache_sizes.get(cache_type, 200))
                else: # Default to general
                    self._caches[cache_type] = LRUCache(max_size=self._cache_sizes.get(cache_type, 1000))

            cache = self._caches[cache_type]
            # The actual 'put' operation is locked within the specific cache instance
            success = cache.put(key, value, **kwargs)

            if success:
                self._stats['total_puts'] += 1
                # Estimate memory saved
                try:
                    # Use pickle to estimate size. Not perfect, but a decent heuristic.
                    size_bytes = len(pickle.dumps(value, protocol=4))
                    self._stats['memory_saved_mb'] += size_bytes / (1024 * 1024)
                except (pickle.PicklingError, TypeError):
                    pass # Ignore objects that can't be pickled for stats

            # Update total size
            self._update_total_size()
            
            return success

    def _update_total_size(self):
        """Recalculates the total size across all caches. Must be called within a lock."""
        total_size = 0
        for cache in self._caches.values():
            if hasattr(cache, 'lru_cache'): # For PatternCache, ModelCache
                 total_size += cache.lru_cache.get_stats().get('size', 0)
            elif isinstance(cache, LRUCache): # For general LRUCache
                 total_size += cache.get_stats().get('size', 0)
        self._stats['size'] = total_size


    def get_pattern_result(self, content_type: str, features: Dict[str, Any]) -> Optional[Any]:
        """
        Get a cached result for a content pattern.

        Args:
            content_type (str): The type of content (e.g., 'invoice_v1').
            features (Dict[str, Any]): A dictionary of features describing the content.
        """
        with self._lock:
            self._stats['total_requests'] += 1
            if 'patterns' not in self._caches:
                self._caches['patterns'] = PatternCache(max_size=self._cache_sizes.get('patterns', 500))
            
        # The method on PatternCache is thread-safe
        result = self._caches['patterns'].get_pattern_result(content_type, features)
        
        with self._lock:
            if result is not None:
                self._stats['total_hits'] += 1
            return result

    def cache_pattern_result(self, content_type: str, features: Dict[str, Any], result: Any, **kwargs):
        """
        Cache a result for a content pattern.

        Args:
            content_type (str): The type of content.
            features (Dict[str, Any]): The features dictionary.
            result (Any): The result to cache.
            **kwargs: Additional arguments (e.g., ttl_seconds).
        """
        with self._lock:
            self._maybe_cleanup()

            if 'patterns' not in self._caches:
                self._caches['patterns'] = PatternCache(max_size=self._cache_sizes.get('patterns', 500))
            
            cache = self._caches['patterns']
            # The method on PatternCache is thread-safe
            success = cache.cache_pattern_result(content_type, features, result, **kwargs)

            if success:
                self._stats['total_puts'] += 1
                try:
                    size_bytes = len(pickle.dumps(result, protocol=4))
                    self._stats['memory_saved_mb'] += size_bytes / (1024 * 1024)
                except (pickle.PicklingError, TypeError):
                    pass
            
            self._update_total_size()
            return success

    def get_model_prediction(self, model_name: str, input_data: Any, model_version: str = "default") -> Optional[Any]:
        """
        Get a cached model prediction.

        Args:
            model_name (str): The name of the model.
            input_data (Any): The input data for the prediction.
            model_version (str): The version of the model.

        Returns:
            Optional[Any]: The cached prediction, or None if not found.
        """
        with self._lock:
            self._stats['total_requests'] += 1
            if 'models' not in self._caches:
                self._caches['models'] = ModelCache(max_size=self._cache_sizes.get('models', 200))

        # The method on ModelCache is thread-safe
        prediction = self._caches['models'].get_prediction(model_name, input_data, model_version)
        
        with self._lock:
            if prediction is not None:
                self._stats['total_hits'] += 1
            return prediction

    def cache_model_prediction(self, model_name: str, input_data: Any, prediction: Any, model_version: str = "default", **kwargs):
        """
        Cache a model prediction.

        Args:
            model_name (str): The name of the model.
            input_data (Any): The input data for the prediction.
            prediction (Any): The prediction to cache.
            model_version (str): The version of the model.
            **kwargs: Additional arguments for the cache (e.g., ttl_seconds).
        """
        with self._lock:
            self._maybe_cleanup()

            if 'models' not in self._caches:
                self._caches['models'] = ModelCache(max_size=self._cache_sizes.get('models', 200))
            
            cache = self._caches['models']
            # The method on ModelCache is thread-safe
            success = cache.cache_prediction(model_name, input_data, prediction, model_version, **kwargs)

            if success:
                self._stats['total_puts'] += 1
                try:
                    size_bytes = len(pickle.dumps(prediction, protocol=4))
                    self._stats['memory_saved_mb'] += size_bytes / (1024 * 1024)
                except (pickle.PicklingError, TypeError):
                    pass
            
            self._update_total_size()
            return success

    def invalidate_model(self, model_name: str, model_version: str = "default"):
        """
        Invalidate cached model predictions.

        Args:
            model_name (str): The name of the model to invalidate.
            model_version (str): The version of the model to invalidate.
        """
        with self._lock:
            if 'models' in self._caches:
                self._caches['models'].invalidate_model(model_name, model_version)
                self._update_total_size()
                cache_logger.info("Invalidated model cache", model_name=model_name, model_version=model_version)

    def clear_all(self):
        """Clears all entries from all managed caches."""
        with self._lock:
            for cache in self._caches.values():
                if hasattr(cache, 'clear'): # LRUCache
                    cache.clear()
                elif hasattr(cache, 'lru_cache'): # PatternCache, ModelCache
                    cache.lru_cache.clear()

            # Reset statistics
            self._stats = {
                "total_requests": 0,
                "total_hits": 0,
                "total_puts": 0,
                "memory_saved_mb": 0.0,
                "size": 0
            }
            cache_logger.info("All caches cleared.")

    def _maybe_cleanup(self):
        """Perform cleanup if needed."""
        with self._lock:
            if time.time() - self._last_cleanup_time > self.cleanup_interval:
                self._cleanup_expired()
                self._last_cleanup_time = time.time()

    def _cleanup_expired(self) -> int:
        """Clean up expired items from all caches."""
        with self._lock:
            total_expired = 0
            for cache_name, cache in self._caches.items():
                # Handle caches that have the method directly (e.g., LRUCache)
                if hasattr(cache, 'cleanup_expired'):
                    expired_count = cache.cleanup_expired()
                    total_expired += expired_count
                # Handle composite caches (e.g., PatternCache, ModelCache)
                elif hasattr(cache, 'lru_cache') and hasattr(cache.lru_cache, 'cleanup_expired'):
                    expired_count = cache.lru_cache.cleanup_expired()
                    total_expired += expired_count
            
            if total_expired > 0:
                cache_logger.info(f"Cleaned up {total_expired} expired items.")
            
            self._update_total_size()
            return total_expired

    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """
        Get a comprehensive report of statistics from all managed caches.

        Returns:
            Dict[str, Any]: A dictionary containing global stats and
                            per-cache stats.
        """
        with self._lock:
            self._update_total_size() # Ensure size is up-to-date
            
            # Create a deep copy to prevent modification of internal stats
            comprehensive_stats = {
                "global": self._stats.copy()
            }
            
            total_requests = self._stats.get("total_requests", 0)
            total_hits = self._stats.get("total_hits", 0)
            
            if total_requests > 0:
                comprehensive_stats["global"]["hit_rate"] = total_hits / total_requests
            else:
                comprehensive_stats["global"]["hit_rate"] = 0
            
            per_cache_stats = {}
            for name, cache in self._caches.items():
                if hasattr(cache, 'get_stats'):
                    per_cache_stats[name] = cache.get_stats()
                elif hasattr(cache, 'lru_cache'): # For PatternCache and ModelCache
                    per_cache_stats[name] = cache.lru_cache.get_stats()
                    if hasattr(cache, 'get_pattern_stats'): # Specific to PatternCache
                         per_cache_stats[name]['pattern_specific'] = cache.get_pattern_stats()

            comprehensive_stats["per_cache"] = per_cache_stats
            
            return comprehensive_stats


_global_cache_instance: Optional[weakref.ref["IntelligentCache"]] = None
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
    """Decorator for caching function results."""
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

# Globally accessible singleton instance of the cache manager
cache_manager = get_intelligent_cache()