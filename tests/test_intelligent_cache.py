"""
Unit tests for Intelligent Cache - Session 3
Tests LRU caching, pattern caching, model caching, and TTL functionality.
"""

import pytest
import time
import threading
from unittest.mock import patch, Mock

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
        """Test input data hashing for consistent keys."""
        # Test different input types
        string_input = "test string"
        dict_input = {"key": "value", "number": 42}
        list_input = [1, 2, 3, "test"]
        
        hash1 = self.model_cache._hash_input(string_input)
        hash2 = self.model_cache._hash_input(dict_input)
        hash3 = self.model_cache._hash_input(list_input)
        
        # All should generate valid hashes
        assert isinstance(hash1, str) and len(hash1) == 16
        assert isinstance(hash2, str) and len(hash2) == 16
        assert isinstance(hash3, str) and len(hash3) == 16
        
        # Same input should generate same hash
        hash1_repeat = self.model_cache._hash_input(string_input)
        assert hash1 == hash1_repeat

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
    return get_intelligent_cache()


class TestIntelligentCache:
    """Test the main IntelligentCache component."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache = get_intelligent_cache()
        self.cache.clear_all()  # Ensure clean state for each test

    def test_general_cache_operations(self):
        """Test general purpose cache get/put."""
        # Basic put/get
        assert self.cache.put("key1", "value1", "general")
        assert self.cache.get("key1", "general") == "value1"
        
        # Non-existent key
        assert self.cache.get("nonexistent", "general") is None

    def test_pattern_cache_integration(self):
        """Test pattern cache through main interface."""
        features = {"document_type": "invoice", "page_count": 3}
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

    def test_comprehensive_stats(self, configured_cache):
        """Test the comprehensive statistics gathering."""
        # Add some data to exercise the caches
        configured_cache.put("test_key", "test_value", cache_type="general")
        configured_cache.get("test_key", cache_type="general") # hit
        configured_cache.get("miss_key", cache_type="general") # miss
        
        stats = configured_cache.get_comprehensive_stats()
        
        assert "general_cache" in stats
        assert "pattern_cache" in stats
        assert "model_cache" in stats
        assert "global" in stats

        general_stats = stats["general_cache"]
        assert general_stats['hits'] >= 1
        assert general_stats['misses'] >= 1
        
        global_stats = stats["global"]
        assert "memory_saved_mb" in global_stats

    def test_cache_cleanup(self):
        """Test automatic cache cleanup."""
        # Add entries with short TTL
        self.cache.put("key1", "value1", "general", ttl_seconds=0.1)
        self.cache.put("key2", "value2", "general", ttl_seconds=1.0)
        
        # Wait for expiration
        time.sleep(0.15)
        
        # Force cleanup
        self.cache._cleanup_expired()
        
        # Check results
        assert self.cache.get("key1", "general") is None
        assert self.cache.get("key2", "general") == "value2"

    def test_clear_all_caches(self):
        """Test clearing all cache types."""
        # Add data to all cache types
        self.cache.put("general_key", "value", "general")
        self.cache.cache_pattern_result("type", {"f": "v"}, "result")
        self.cache.cache_model_prediction("model", {"i": "d"}, "pred")
        
        # Clear all
        self.cache.clear_all()
        
        # Check stats immediately after clearing
        stats = self.cache.get_comprehensive_stats()
        assert stats["global"]["total_requests"] == 0
        assert stats["global"]["total_hits"] == 0
        
        # Verify all are empty
        assert self.cache.get("general_key", "general") is None
        assert self.cache.get_pattern_result("type", {"f": "v"}) is None
        assert self.cache.get_model_prediction("model", {"i": "d"}) is None


class TestGlobalCacheFunctions:
    """Test global convenience functions."""

    def setup_method(self):
        """Clear the global cache before each test."""
        get_intelligent_cache().clear_all()

    def test_cache_get_put(self):
        """Test global cache_get and cache_put."""
        assert cache_get("global_key") is None
        assert cache_put("global_key", "my_result", "general", ttl_seconds=300)
        result = cache_get("global_key", "general")
        assert result == "my_result"

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
    """Test the cache_result decorator."""

    def test_function_result_caching(self):
        """Test caching function results with decorator."""
        call_count = 0
        
        @cache_result(cache_type="general", ttl_seconds=60)
        def expensive_function(x, y):
            nonlocal call_count
            call_count += 1
            return x + y
        
        # Clear cache first
        get_intelligent_cache().clear_all()
        
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
        
        # Clear cache first
        get_intelligent_cache().clear_all()
        
        # Calls with different ignored_param should use same cache
        result1 = test_function(2, 3, ignored_param="a")
        result2 = test_function(2, 3, ignored_param="b")
        
        assert result1 == result2 == 6
        assert call_count == 1  # Function called only once


class TestConcurrency:
    """Test concurrency and thread safety."""

    def setup_method(self):
        get_intelligent_cache().clear_all()

    def test_thread_safety(self, configured_cache):
        """Test that cache operations are thread-safe."""
        exceptions = []
        
        def worker(cache, start_index, end_index):
            try:
                for i in range(start_index, end_index):
                    key = f"key_{i}"
                    value = f"value_{i}"
                    cache.put(key, value, cache_type="general", ttl_seconds=10)
                    retrieved = cache.get(key, cache_type="general")
                    assert retrieved == value, f"Failed to retrieve key {key}"
            except Exception as e:
                exceptions.append(e)

        threads = []
        num_threads = 10
        items_per_thread = 10
        for i in range(num_threads):
            start = i * items_per_thread
            end = (i + 1) * items_per_thread
            t = threading.Thread(target=worker, args=(configured_cache, start, end))
            threads.append(t)
            t.start()
        
        for t in threads:
            t.join()

        # Check that no exceptions occurred in worker threads
        assert not exceptions, f"Exceptions occurred in worker threads: {exceptions}"

        # Check final state for consistency
        stats = configured_cache.get_comprehensive_stats()
        general_stats = stats.get("general_cache", {})
        
        # All items should be in the cache
        assert general_stats.get("size") == num_threads * items_per_thread


if __name__ == "__main__":
    pytest.main([__file__]) 