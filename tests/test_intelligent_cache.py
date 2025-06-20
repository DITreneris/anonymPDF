"""
Unit tests for Intelligent Cache - Session 3
Tests LRU caching, pattern caching, model caching, and TTL functionality.
"""

import pytest
import time
import threading
import weakref
from unittest.mock import patch, Mock

from app.core import intelligent_cache as cache_module
from app.core.intelligent_cache import (
    LRUCache,
    PatternCache,
    ModelCache,
    IntelligentCache,
    CacheEntry,
    CachePolicy,
    get_intelligent_cache,
    cache_get,
    cache_put,
    cache_pattern,
    cache_model_prediction,
    cache_result,
    get_cache_stats
)
from app.core.config_manager import get_config


class TestCacheEntry:
    """Test CacheEntry data class."""

    def test_cache_entry_creation(self):
        """Test creating cache entry."""
        entry = CacheEntry(
            value="test_value",
            created_at=time.time(),
            last_accessed=time.time(),
            ttl_seconds=300
        )
        
        assert entry.value == "test_value"
        assert entry.ttl_seconds == 300
        assert entry.access_count == 0

    def test_cache_entry_expiration(self):
        """Test cache entry expiration."""
        # Non-expiring entry
        entry = CacheEntry(
            value="test",
            created_at=time.time(),
            last_accessed=time.time()
        )
        assert not entry.is_expired
        
        # Expired entry
        old_time = time.time() - 100
        expired_entry = CacheEntry(
            value="test",
            created_at=old_time,
            last_accessed=old_time,
            ttl_seconds=50
        )
        assert expired_entry.is_expired

    def test_cache_entry_touch(self):
        """Test updating access statistics."""
        entry = CacheEntry(
            value="test",
            created_at=time.time(),
            last_accessed=time.time()
        )
        
        original_access_count = entry.access_count
        original_last_accessed = entry.last_accessed
        
        time.sleep(0.01)  # Small delay
        entry.touch()
        
        assert entry.access_count == original_access_count + 1
        assert entry.last_accessed > original_last_accessed


class TestLRUCache:
    """Test LRU Cache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = LRUCache(max_size=3)

    def test_basic_get_put(self):
        """Test basic cache operations."""
        # Put and get
        assert self.cache.put("key1", "value1")
        assert self.cache.get("key1") == "value1"
        
        # Non-existent key
        assert self.cache.get("nonexistent") is None

    def test_lru_eviction(self):
        """Test LRU eviction policy."""
        # Fill cache to capacity
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        
        # Add one more - should evict key1
        self.cache.put("key4", "value4")
        
        assert self.cache.get("key1") is None  # Evicted
        assert self.cache.get("key2") == "value2"  # Still there
        assert self.cache.get("key3") == "value3"  # Still there
        assert self.cache.get("key4") == "value4"  # New entry

    def test_access_updates_order(self):
        """Test that accessing updates LRU order."""
        # Fill cache
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        
        # Access key1 to make it most recent
        self.cache.get("key1")
        
        # Add new key - should evict key2 (oldest)
        self.cache.put("key4", "value4")
        
        assert self.cache.get("key1") == "value1"  # Should still be there
        assert self.cache.get("key2") is None      # Should be evicted
        assert self.cache.get("key3") == "value3"  # Should still be there
        assert self.cache.get("key4") == "value4"  # New entry

    def test_ttl_expiration(self):
        """Test TTL-based expiration."""
        # Put with short TTL
        self.cache.put("key1", "value1", ttl_seconds=0.1)
        
        # Should be accessible immediately
        assert self.cache.get("key1") == "value1"
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Should be expired
        assert self.cache.get("key1") is None

    def test_cache_stats(self):
        """Test cache statistics."""
        # Initial stats
        stats = self.cache.get_stats()
        assert stats['hits'] == 0
        assert stats['misses'] == 0
        assert stats['evictions'] == 0
        assert stats['size'] == 0
        assert stats['hit_rate'] == 0
        assert stats['total_requests'] == 0
        
        # Add some operations
        self.cache.put("key1", "value1")
        self.cache.get("key1")  # Hit
        self.cache.get("key2")  # Miss
        
        # Get updated stats
        stats = self.cache.get_stats()
        assert stats['hits'] == 1
        assert stats['misses'] == 1
        assert stats['evictions'] == 0
        assert stats['size'] == 1
        assert stats['hit_rate'] == 0.5
        assert stats['total_requests'] == 2
        
        # Test eviction stats
        self.cache.put("key2", "value2")
        self.cache.put("key3", "value3")
        self.cache.put("key4", "value4")  # Should evict key1
        
        stats = self.cache.get_stats()
        assert stats['evictions'] == 1
        assert stats['size'] == 3

    def test_cache_invalidation(self):
        """Test explicit cache invalidation."""
        self.cache.put("key1", "value1")
        assert self.cache.get("key1") == "value1"
        
        # Invalidate
        assert self.cache.invalidate("key1") is True
        assert self.cache.get("key1") is None
        
        # Invalidate non-existent key
        assert self.cache.invalidate("nonexistent") is False

    def test_cache_clear(self):
        """Test clearing entire cache."""
        self.cache.put("key1", "value1")
        self.cache.put("key2", "value2")
        
        self.cache.clear()
        
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") is None
        assert self.cache.get_stats()['size'] == 0

    def test_cleanup_expired(self):
        """Test cleaning up expired entries."""
        # Add entries with different TTL
        self.cache.put("key1", "value1", ttl_seconds=0.1)
        self.cache.put("key2", "value2", ttl_seconds=1.0)
        self.cache.put("key3", "value3")  # No TTL
        
        # Wait for first to expire
        time.sleep(0.15)
        
        # Cleanup
        cleaned = self.cache.cleanup_expired()
        
        assert cleaned == 1  # One expired entry
        assert self.cache.get("key1") is None
        assert self.cache.get("key2") == "value2"
        assert self.cache.get("key3") == "value3"


class TestPatternCache:
    """Test Pattern Cache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.pattern_cache = PatternCache(max_size=10)

    def test_pattern_caching(self):
        """Test caching by content patterns."""
        features = {"word_count": 100, "has_header": True, "language": "en"}
        result = {"confidence": 0.95, "detected_pii": ["email", "phone"]}
        
        # Cache result
        success = self.pattern_cache.cache_pattern_result("document", features, result)
        assert success
        
        # Retrieve result
        cached_result = self.pattern_cache.get_pattern_result("document", features)
        assert cached_result == result

    def test_pattern_key_generation(self):
        """Test pattern key generation consistency."""
        features1 = {"a": 1, "b": 2}
        features2 = {"b": 2, "a": 1}  # Same features, different order
        
        key1 = self.pattern_cache._generate_pattern_key("type1", features1)
        key2 = self.pattern_cache._generate_pattern_key("type1", features2)
        
        # Should generate same key for same features regardless of order
        assert key1 == key2

    def test_pattern_statistics(self):
        """Test pattern-specific statistics."""
        features = {"feature1": "value1"}
        
        # Miss
        result = self.pattern_cache.get_pattern_result("doc_type1", features)
        assert result is None
        
        # Cache and hit
        self.pattern_cache.cache_pattern_result("doc_type1", features, "result1")
        result = self.pattern_cache.get_pattern_result("doc_type1", features)
        assert result == "result1"
        
        # Check stats
        stats = self.pattern_cache.get_pattern_stats()
        assert "doc_type1" in stats
        assert stats["doc_type1"]["requests"] == 2
        assert stats["doc_type1"]["hits"] == 1
        assert stats["doc_type1"]["hit_rate"] == 0.5


class TestModelCache:
    """Test Model Cache functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.model_cache = ModelCache(max_size=10)

    def test_model_prediction_caching(self):
        """Test caching model predictions."""
        input_data = {"text": "Sample text for processing"}
        prediction = {"label": "PII", "confidence": 0.89}
        
        # Cache prediction
        success = self.model_cache.cache_prediction(
            "test_model", input_data, prediction, "v1.0"
        )
        assert success
        
        # Retrieve prediction
        cached_prediction = self.model_cache.get_prediction(
            "test_model", input_data, "v1.0"
        )
        assert cached_prediction == prediction

    def test_input_hashing(self):
        """Test that various inputs are hashed to a consistent string key."""
        # This is testing the private helper, which is fine for unit tests
        hasher = self.model_cache._hash_input

        # Test different data types
        hash1 = hasher({"a": 1, "b": 2})
        hash2 = hasher({"b": 2, "a": 1})  # Order should not matter
        hash3 = hasher([1, 2, 3])
        hash4 = hasher("simple_string")

        assert isinstance(hash1, str) and len(hash1) == 64
        assert hash1 == hash2
        assert hash1 != hash3
        assert hash3 != hash4

    def test_model_invalidation(self):
        """Test invalidating specific model versions."""
        input1 = {"text": "input1"}
        input2 = {"text": "input2"}
        prediction = {"result": "test"}
        
        # Cache predictions for same model, different versions
        self.model_cache.cache_prediction("model1", input1, prediction, "v1.0")
        self.model_cache.cache_prediction("model1", input2, prediction, "v1.0")
        self.model_cache.cache_prediction("model1", input1, prediction, "v2.0")
        
        # Verify all are cached
        assert self.model_cache.get_prediction("model1", input1, "v1.0") == prediction
        assert self.model_cache.get_prediction("model1", input2, "v1.0") == prediction
        assert self.model_cache.get_prediction("model1", input1, "v2.0") == prediction
        
        # Invalidate v1.0
        self.model_cache.invalidate_model("model1", "v1.0")
        
        # v1.0 predictions should be gone, v2.0 should remain
        assert self.model_cache.get_prediction("model1", input1, "v1.0") is None
        assert self.model_cache.get_prediction("model1", input2, "v1.0") is None
        assert self.model_cache.get_prediction("model1", input1, "v2.0") == prediction


@pytest.fixture(scope='module')
def configured_cache():
    """Ensure a clean cache for the module."""
    cache = get_intelligent_cache()
    cache.clear_all()
    return cache


class TestIntelligentCache:
    """Test integration of different cache types in IntelligentCache."""

    def setup_method(self):
        """Set up a clean cache before each test method."""
        # Use the global singleton but clear it to ensure test isolation
        self.cache = get_intelligent_cache()
        self.cache.clear_all()

    def teardown_method(self):
        """Clean up the cache after each test method."""
        self.cache.clear_all()

    def test_general_cache_operations(self):
        """Test general purpose cache within IntelligentCache."""
        self.cache.put("key1", "value1", "general")
        assert self.cache.get("key1", "general") == "value1"
        
        # Non-existent key
        assert self.cache.get("nonexistent", "general") is None

    def test_pattern_cache_integration(self):
        """Test pattern caching via IntelligentCache."""
        features = {"a": 1, "b": 2}
        result = {"pii_detected": True, "confidence": 0.92}
        
        # Cache pattern result
        success = self.cache.cache_pattern_result("invoice", features, result)
        assert success
        
        # Retrieve pattern result
        cached_result = self.cache.get_pattern_result("invoice", features)
        assert cached_result == result

    def test_model_cache_integration(self):
        """Test model cache through main interface."""
        input_data = {"features": [1, 2, 3, 4, 5]}
        prediction = {"class": "sensitive", "probability": 0.85}
        
        # Cache model prediction
        success = self.cache.cache_model_prediction(
            "classifier_v1", input_data, prediction, "1.0"
        )
        assert success
        
        # Retrieve model prediction
        cached_prediction = self.cache.get_model_prediction(
            "classifier_v1", input_data, "1.0"
        )
        assert cached_prediction == prediction

    def test_comprehensive_stats(self):
        """Test the comprehensive statistics report."""
        # Create a dedicated cache instance for this test to ensure stats are isolated
        cache = IntelligentCache({
            'eager_init': True,
            'cache_sizes': {'general': 50, 'patterns': 20, 'models': 10}
        })
        # Add some data for stats testing
        cache.put("key1", "value1", "general")
        cache.get("key1", "general")  # hit
        cache.get("key_miss", "general")  # miss
        cache.cache_pattern_result("type1", {"f": 1}, "res1")
        cache.get_pattern_result("type1", {"f": 1})  # hit
        cache.cache_model_prediction("model1", {"in": 1}, "pred1")
        cache.get_model_prediction("model1", {"in": 1})  # hit

        stats = cache.get_comprehensive_stats()

        # Check top-level structure
        assert "global" in stats
        assert "per_cache" in stats

        # Check global stats - 4 get calls are made
        assert stats["global"]["total_requests"] >= 4
        assert stats["global"]["total_hits"] >= 3
        assert stats["global"]["total_puts"] >= 3

        # Check per-cache structure
        per_cache_stats = stats["per_cache"]
        assert "general" in per_cache_stats
        assert "patterns" in per_cache_stats
        assert "models" in per_cache_stats
        assert per_cache_stats["general"]["size"] > 0

    def test_cache_cleanup(self):
        """Test selective cleanup of expired entries across all caches."""
        # Add entries to different caches
        self.cache.put("key1", "value1", "general", ttl_seconds=0.1)
        self.cache.put("key2", "value2", "general")  # No TTL
        self.cache.cache_pattern_result(
            "test_type", {"key": "p1"}, "pattern1", ttl_seconds=0.1
        )
        self.cache.cache_pattern_result(
            "test_type", {"key": "p2"}, "pattern2"
        ) # No TTL

        # Wait for expiration
        time.sleep(0.15)

        # Run cleanup - calling the private method for a deterministic test
        expired_count = self.cache._cleanup_expired()
        
        # Assert that only expired entries were removed
        assert expired_count == 2
        assert self.cache.get("key1", "general") is None
        assert self.cache.get("key2", "general") == "value2"  # Should remain
        assert self.cache.get_pattern_result("test_type", {"key": "p1"}) is None
        assert self.cache.get_pattern_result("test_type", {"key": "p2"}) == "pattern2" # Should remain

    def test_clear_all_caches(self):
        """Test clearing all caches at once."""
        # Add data to multiple caches
        self.cache.put("key1", "value1", "general")
        self.cache.cache_pattern_result("test_type", {"key": "p1"}, "pattern1")
        self.cache.cache_model_prediction("m1", {"in": "data"}, "model1")

        # Clear all
        self.cache.clear_all()

        # Verify all are empty
        assert self.cache.get("key1", "general") is None
        assert self.cache.get_pattern_result("test_type", {"key": "p1"}) is None
        assert self.cache.get_model_prediction("m1", {"in": "data"}) is None
        
        stats = self.cache.get_comprehensive_stats()
        # FIX: Use .get() to avoid KeyError if a cache type was never used.
        assert stats.get("per_cache", {}).get("general", {}).get("size", 0) == 0
        assert stats.get("per_cache", {}).get("patterns", {}).get("size", 0) == 0
        assert stats.get("per_cache", {}).get("models", {}).get("size", 0) == 0


class TestGlobalCacheFunctions:
    """Test the global convenience functions."""

    def setup_method(self):
        """Ensure a clean cache state for each test."""
        # The get_intelligent_cache() is a singleton, so clear it.
        get_intelligent_cache().clear_all()
    
    def teardown_method(self):
        """Ensure a clean cache state after each test."""
        get_intelligent_cache().clear_all()

    def test_cache_get_put(self):
        """Test global get/put functions."""
        cache_put("key1", "value1", "general")
        result = cache_get("key1", "general")
        assert result == "value1"

    def test_cache_pattern_function(self):
        """Test global cache_pattern function."""
        # Clear cache first
        get_intelligent_cache().clear_all()
        
        # Cache a pattern result first
        get_intelligent_cache().cache_pattern_result(
            "test_type", {"feature": "value"}, "cached_result"
        )
        
        # Use global function
        result = cache_pattern("test_type", {"feature": "value"})
        assert result == "cached_result"

    def test_cache_model_prediction_function(self):
        """Test global cache_model_prediction function."""
        # Clear cache first
        get_intelligent_cache().clear_all()
        
        # Cache a model prediction first
        get_intelligent_cache().cache_model_prediction(
            "test_model", {"input": "data"}, "prediction_result"
        )
        
        # Use global function
        result = cache_model_prediction("test_model", {"input": "data"})
        assert result == "prediction_result"


class TestCacheDecorator:
    """Test the @cache_result decorator."""

    def setup_method(self):
        """Clean the cache for each test."""
        get_intelligent_cache().clear_all()

    def test_function_result_caching(self):
        """Test that a function's result is cached."""
        call_count = 0
        
        @cache_result(cache_type="general", ttl_seconds=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # First call - should execute function
        result1 = expensive_function(1, 2)
        assert result1 == 3
        assert call_count == 1
        
        # Second call with same args - should use cache
        result2 = expensive_function(1, 2)
        assert result2 == 3
        assert call_count == 1  # Function not called again
        
        # Call with different args - should execute function
        result3 = expensive_function(2, 3)
        assert result3 == 5
        assert call_count == 2

    def test_custom_key_function(self):
        """Test decorator with custom key function."""
        call_count = 0
        
        def custom_key(x, y, **kwargs):
            return f"custom_{x}_{y}"
        
        @cache_result(cache_type="general", key_func=custom_key)
        def test_function(x, y, ignored_param=None):
            nonlocal call_count
            call_count += 1
            return x * y
        
        # Calls with different ignored_param should use same cache
        result1 = test_function(2, 3, ignored_param="a")
        result2 = test_function(2, 3, ignored_param="b")
        
        assert result1 == result2 == 6
        assert call_count == 1  # Function called only once


@pytest.mark.serial
def test_thread_safety():
    """
    Test that the cache handles concurrent reads/writes without corruption.
    This test now creates its OWN instance of the cache to prevent interference.
    """
    original_instance = (
        cache_module._global_cache_instance()
        if cache_module._global_cache_instance else None
    )
    try:
        # Use a large cleanup interval to prevent cleanup during the test
        new_cache = cache_module.IntelligentCache(config={
            "max_size": 1000,
            "cleanup_interval": 999999,
            "cache_sizes": {"general": 1000} # Ensure general cache is large enough
        })
        cache_module._global_cache_instance = weakref.ref(new_cache)
        cache = new_cache

        num_threads = 10
        ops_per_thread = 100
        exceptions = []

        def worker(cache_instance, start_index, end_index):
            try:
                for i in range(start_index, end_index):
                    # Use the 'general' cache type explicitly
                    cache_instance.put(f"key-{i}", f"value-{i}", cache_type="general")
                    cache_instance.get(f"key-{i-1}", cache_type="general")
            except Exception as e:
                exceptions.append(e)

        threads = []
        for i in range(num_threads):
            start = i * ops_per_thread
            end = start + ops_per_thread
            thread = threading.Thread(target=worker, args=(cache, start, end))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert not exceptions, f"Exceptions occurred in worker threads: {exceptions}"

        # Use the thread-safe method to get stats
        stats = cache.get_comprehensive_stats().get("global", {})
        
        expected_size = num_threads * ops_per_thread
        actual_size = stats.get("size", 0)
        
        assert actual_size == expected_size, \
            f"Expected size {expected_size}, got {actual_size}. Full stats: {stats}"
        
        # We expect many hits, but the exact number can vary slightly based on thread scheduling
        assert stats.get("total_hits", 0) > 0, f"Expected hits to be > 0. Full stats: {stats}"
    finally:
        if original_instance:
            cache_module._global_cache_instance = weakref.ref(original_instance)
        else:
            cache_module._global_cache_instance = None

if __name__ == "__main__":
    pytest.main([__file__]) 