"""
Tests for Memory Optimizer - Priority 3 Session 3
Comprehensive testing of memory optimization, SpaCy management, and garbage collection.
"""

import pytest
import time
import threading
import gc
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any
import psutil

from app.core.memory_optimizer import (
    MemoryMetrics,
    SpaCyOptimizer,
    GarbageCollectionManager,
    MemoryMonitor,
    MemoryOptimizer,
    get_memory_optimizer,
    memory_logger
)
from app.core.memory_utils import (
    optimize_memory,
    get_memory_stats,
    start_memory_monitoring,
    stop_memory_monitoring,
    memory_optimized
)


class TestMemoryMetrics:
    """Test MemoryMetrics data class."""
    
    def test_memory_metrics_creation(self):
        """Test memory metrics creation and properties."""
        metrics = MemoryMetrics(
            timestamp=time.time(),
            rss_mb=100.0,
            vms_mb=200.0,
            percent=75.0,
            available_mb=1000.0,
            process_count=500
        )
        
        assert metrics.rss_mb == 100.0
        assert metrics.vms_mb == 200.0
        assert metrics.total_mb == 300.0
        assert metrics.percent == 75.0
        assert metrics.available_mb == 1000.0
        assert metrics.process_count == 500
        assert metrics.timestamp > 0


class TestSpaCyOptimizer:
    """Test SpaCy optimizer functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.optimizer = SpaCyOptimizer()
    
    @patch('spacy.load')
    def test_model_loading_and_caching(self, mock_spacy_load):
        """Test SpaCy model loading and caching."""
        mock_model = Mock()
        mock_spacy_load.return_value = mock_model
        
        # First call should load model
        result1 = self.optimizer.get_model("en_core_web_sm")
        assert result1 == mock_model
        mock_spacy_load.assert_called_once_with("en_core_web_sm")
        
        # Second call should use cache
        mock_spacy_load.reset_mock()
        result2 = self.optimizer.get_model("en_core_web_sm")
        assert result2 == mock_model
        mock_spacy_load.assert_not_called()
    
    @patch('spacy.load')
    def test_model_eviction(self, mock_spacy_load):
        """Test model eviction when cache is full."""
        mock_models = [Mock() for _ in range(5)]
        mock_spacy_load.side_effect = mock_models
        
        # Fill cache beyond limit
        for i in range(4):
            self.optimizer.get_model(f"model_{i}")
        
        # Check first model is evicted
        stats = self.optimizer.get_model_stats()
        assert len(stats['cached_models']) == 3  # max_models limit
    
    @patch('spacy.load')
    def test_model_clearing(self, mock_spacy_load):
        """Test clearing all cached models."""
        mock_spacy_load.return_value = Mock()
        
        # Load some models
        self.optimizer.get_model("model_1")
        self.optimizer.get_model("model_2")
        
        stats_before = self.optimizer.get_model_stats()
        assert stats_before['model_count'] == 2
        
        # Clear models
        self.optimizer.clear_models()
        
        stats_after = self.optimizer.get_model_stats()
        assert stats_after['model_count'] == 0
        assert len(stats_after['cached_models']) == 0
    
    def test_pipeline_optimization(self):
        """Test SpaCy pipeline optimization."""
        mock_model = Mock()
        mock_model.pipe_names = ['tokenizer', 'tagger', 'parser', 'ner']
        
        disabled = self.optimizer.optimize_pipeline(
            mock_model, 
            ['tagger', 'parser']
        )
        
        assert 'tagger' in disabled
        assert 'parser' in disabled
        mock_model.disable_pipes.assert_called()
    
    def test_model_stats(self):
        """Test model statistics reporting."""
        stats = self.optimizer.get_model_stats()
        
        assert isinstance(stats, dict)
        assert 'cached_models' in stats
        assert 'model_count' in stats
        assert 'max_models' in stats
        assert 'usage_times' in stats
        assert stats['model_count'] == 0
        assert stats['max_models'] == 3


class TestGarbageCollectionManager:
    """Test garbage collection management."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.gc_manager = GarbageCollectionManager()
    
    def test_gc_manager_initialization(self):
        """Test garbage collection manager initialization."""
        assert hasattr(self.gc_manager, '_gc_stats')
        assert hasattr(self.gc_manager, '_collection_threshold')
        assert hasattr(self.gc_manager, '_time_threshold')
        
        stats = self.gc_manager.get_gc_stats()
        assert stats['collections'] == 0
        assert stats['objects_collected'] == 0
    
    def test_should_collect_memory_threshold(self):
        """Test garbage collection trigger based on memory threshold."""
        # Memory growth exceeds threshold
        should_collect = self.gc_manager.should_collect(200.0, 50.0)  # 150MB growth
        assert should_collect is True
        
        # Memory growth below threshold
        should_collect = self.gc_manager.should_collect(120.0, 50.0)  # 70MB growth
        assert should_collect is False
    
    def test_should_collect_time_threshold(self):
        """Test garbage collection trigger based on time threshold."""
        # Simulate old last collection
        self.gc_manager._last_collection = time.time() - 100  # 100 seconds ago
        
        should_collect = self.gc_manager.should_collect(100.0, 90.0)
        assert should_collect is True
    
    @patch('psutil.Process')
    def test_force_collection(self, mock_process):
        """Test forced garbage collection."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        result = self.gc_manager.force_collection()
        
        assert isinstance(result, dict)
        assert 'duration_seconds' in result
        assert 'memory_freed_mb' in result
        assert 'objects_collected' in result
        assert 'total_objects_after' in result
        
        # Check stats updated
        stats = self.gc_manager.get_gc_stats()
        assert stats['collections'] == 1
    
    def test_optimize_gc_settings(self):
        """Test garbage collection settings optimization."""
        # Test heavy processing mode
        self.gc_manager.optimize_gc_settings("heavy")
        assert self.gc_manager._collection_threshold == 50
        assert self.gc_manager._time_threshold == 15
        
        # Test light processing mode
        self.gc_manager.optimize_gc_settings("light")
        assert self.gc_manager._collection_threshold == 200
        assert self.gc_manager._time_threshold == 60
        
        # Test normal mode
        self.gc_manager.optimize_gc_settings("normal")
        assert self.gc_manager._collection_threshold == 100
        assert self.gc_manager._time_threshold == 30


class TestMemoryMonitor:
    """Test memory monitoring functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.monitor = MemoryMonitor(monitoring_interval=0.1)
    
    def teardown_method(self):
        """Clean up after tests."""
        self.monitor.stop_monitoring()
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_get_current_metrics(self, mock_virtual_memory, mock_process):
        """Test current memory metrics collection."""
        # Mock process memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB
        mock_memory_info.vms = 200 * 1024 * 1024  # 200MB
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        # Mock system memory info
        mock_sys_memory = Mock()
        mock_sys_memory.percent = 75.0
        mock_sys_memory.available = 1000 * 1024 * 1024  # 1000MB
        mock_virtual_memory.return_value = mock_sys_memory
        
        metrics = self.monitor.get_current_metrics()
        
        assert isinstance(metrics, MemoryMetrics)
        assert metrics.rss_mb == 100.0
        assert metrics.vms_mb == 200.0
        assert metrics.percent == 75.0
        assert metrics.available_mb == 1000.0
    
    def test_baseline_setting(self):
        """Test memory baseline setting."""
        self.monitor.set_baseline()
        
        assert self.monitor._baseline_memory is not None
        assert self.monitor._baseline_memory > 0
    
    def test_monitoring_start_stop(self):
        """Test monitoring start and stop."""
        assert self.monitor._monitoring is False
        
        self.monitor.start_monitoring()
        assert self.monitor._monitoring is True
        assert self.monitor._monitor_thread is not None
        
        self.monitor.stop_monitoring()
        assert self.monitor._monitoring is False
    
    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_memory_summary(self, mock_virtual_memory, mock_process):
        """Test memory summary generation."""
        # Mock memory info
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024
        mock_memory_info.vms = 200 * 1024 * 1024
        mock_process.return_value.memory_info.return_value = mock_memory_info
        
        mock_sys_memory = Mock()
        mock_sys_memory.percent = 75.0
        mock_sys_memory.available = 1000 * 1024 * 1024
        mock_virtual_memory.return_value = mock_sys_memory
        
        # Set baseline and get summary
        self.monitor.set_baseline()
        summary = self.monitor.get_memory_summary()
        
        assert isinstance(summary, dict)
        assert 'current' in summary
        assert 'baseline_mb' in summary
        assert 'peak_mb' in summary
        assert 'history_count' in summary
    
    def test_memory_monitoring_loop(self):
        """Test memory monitoring loop functionality."""
        self.monitor.start_monitoring()
        
        # Let it run briefly
        time.sleep(0.3)
        
        self.monitor.stop_monitoring()
        
        # Check that some history was collected
        summary = self.monitor.get_memory_summary()
        assert summary['history_count'] > 0


class TestMemoryOptimizer:
    """Test MemoryOptimizer core functionality."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock config to avoid relying on global get_config()
        self.mock_config = {
            'memory_optimizer': {
                'doc_gc_threshold_mb': 50,
                'some_other_setting': True 
            },
            'performance': { # Fallback config
                 'parallel_processing': {} # Ensure this path is fine
            }
        }
        # Patch get_config to return our mock_config
        with patch('app.core.memory_optimizer.get_config', return_value=self.mock_config):
            self.optimizer = MemoryOptimizer() # Uses mock_config now
    
    def teardown_method(self):
        """Clean up after tests."""
        if hasattr(self.optimizer, 'monitor') and self.optimizer.monitor._monitoring:
             self.optimizer.stop_optimization() 
    
    def test_optimizer_initialization(self):
        """Test MemoryOptimizer initialization and component setup."""
        assert self.optimizer.gc_manager is not None, "gc_manager should be initialized"
        assert self.optimizer.monitor is not None, "monitor should be initialized"
        assert self.optimizer.config is not None, "config should be loaded"
        
        # Ensure config is loaded correctly from the mocked config
        assert self.optimizer.config.get('doc_gc_threshold_mb') == 50
        
        # Check that monitoring is not started by default
        assert not self.optimizer.monitor._monitoring, "Monitoring should not be active by default"
    
    def test_start_stop_optimization(self):
        """Test starting and stopping optimization (monitoring)."""
        with patch.object(self.optimizer.monitor, 'start_monitoring') as mock_start_mon, \
             patch.object(self.optimizer.monitor, 'stop_monitoring') as mock_stop_mon:
            
            # Ensure the mock also sets the _monitoring flag for the assertion
            def side_effect_start_monitoring():
                self.optimizer.monitor._monitoring = True
            mock_start_mon.side_effect = side_effect_start_monitoring
            
            self.optimizer.start_optimization()
            mock_start_mon.assert_called_once()
            assert self.optimizer.monitor._monitoring # Should be true after starting
            
            self.optimizer.stop_optimization()
            mock_stop_mon.assert_called_once()
    
    @patch('psutil.virtual_memory') # Mock psutil for monitor
    def test_optimize_memory(self, mock_psutil_vm): # mock_psutil_proc removed from params
        """Test the main memory optimization function."""
        
        mock_mem_info_start = Mock()
        mock_mem_info_start.rss = 100 * 1024 * 1024  # 100MB
        mock_mem_info_start.vms = 200 * 1024 * 1024

        mock_mem_info_end = Mock()
        mock_mem_info_end.rss = 80 * 1024 * 1024   # 80MB (simulating freed memory)
        mock_mem_info_end.vms = 180 * 1024 * 1024
        
        # Create a single mock instance for psutil.Process
        mock_single_process_instance = Mock(spec=psutil.Process)
        # Correct order for side_effect based on calls:
        # 1. monitor.get_current_metrics (start in optimize_memory)
        # 2. gc_manager._get_memory_usage (start in force_collection)
        # 3. gc_manager._get_memory_usage (end in force_collection)
        # 4. monitor.get_current_metrics (end in optimize_memory)
        mock_single_process_instance.memory_info.side_effect = [
            mock_mem_info_start, # Call 1
            mock_mem_info_start, # Call 2
            mock_mem_info_end,   # Call 3
            mock_mem_info_end    # Call 4
        ]

        # Mock system memory for MemoryMonitor.get_current_metrics (used by both monitor calls)
        mock_sys_mem = Mock()
        mock_sys_mem.percent = 50.0
        mock_sys_mem.available = 2000 * 1024 * 1024
        mock_psutil_vm.return_value = mock_sys_mem # This mock is from the @patch decorator

        # Patch psutil.Process to return our single controlled instance
        # Also mock gc.get_objects for consistent object collection simulation
        with patch('psutil.Process', return_value=mock_single_process_instance) as mock_process_constructor, \
             patch('gc.get_objects', side_effect=[
                 [object()] * 120, # For monitor.get_current_metrics (start)
                 [object()] * 100, # For gc_manager.force_collection (objects_before)
                 [object()] * 50,  # For gc_manager.force_collection (objects_after)
                 [object()] * 40   # For monitor.get_current_metrics (end)
             ]):
            result = self.optimizer.optimize_memory()
        
        assert isinstance(result, dict)
        assert 'start_memory_mb' in result
        assert 'end_memory_mb' in result
        assert 'memory_saved_mb' in result
        assert 'duration_seconds' in result
        assert 'memory_freed_mb' in result # From gc_manager's perspective
        assert 'objects_collected' in result

        assert result['start_memory_mb'] == 100.0
        # End memory for the monitor is the 4th item in side_effect (mock_mem_info_end)
        assert result['end_memory_mb'] == 80.0 
        assert result['memory_saved_mb'] == pytest.approx(20.0)
        # gc_manager's memory_freed_mb is (call 2 rss - call 3 rss) = 100 - 80 = 20
        assert result['memory_freed_mb'] == pytest.approx(20.0) 
        assert result['objects_collected'] >= 0

    def test_optimization_stats(self):
        """Test comprehensive optimization statistics reporting."""
        self.optimizer.start_optimization()
        # Mock parts of optimize_memory to avoid complex psutil mocking here if already tested above
        with patch.object(self.optimizer.monitor, 'get_current_metrics', return_value=MemoryMetrics(time.time(), 100,200,50,1000,100)), \
             patch.object(self.optimizer.gc_manager, 'force_collection', return_value={'duration_seconds':0.1, 'memory_freed_mb':10, 'objects_collected':5}):
            self.optimizer.optimize_memory() 
        self.optimizer.stop_optimization()

        stats = self.optimizer.get_optimization_stats()
        
        assert isinstance(stats, dict)
        assert 'gc_stats' in stats
        assert 'memory_summary' in stats
        # Removed: assert 'spacy_stats' in stats
        # Removed: assert 'last_optimization' in stats
        # Removed: assert 'auto_optimize_enabled' in stats
        
        assert isinstance(stats['gc_stats'], dict)
        assert isinstance(stats['memory_summary'], dict)

    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_optimized_processing_context(self, mock_psutil_vm, mock_psutil_proc):
        """Test the optimized_processing context manager."""
        # Mock psutil calls as they are used by gc_manager and monitor
        mock_mem_info = Mock()
        mock_mem_info.rss = 100 * 1024 * 1024
        mock_psutil_proc.return_value.memory_info.return_value = mock_mem_info
        mock_sys_mem = Mock()
        mock_sys_mem.percent = 50.0
        mock_sys_mem.available = 2000 * 1024 * 1024
        mock_psutil_vm.return_value = mock_sys_mem

        with patch.object(self.optimizer.gc_manager, 'optimize_gc_settings') as mock_optimize_settings, \
             patch.object(self.optimizer.gc_manager, 'force_collection') as mock_force_collection, \
             patch('gc.set_threshold') as mock_gc_set_threshold, \
             patch('gc.get_threshold', return_value=(700,10,10)) as mock_gc_get_threshold:
            
            with self.optimizer.optimized_processing(processing_mode="heavy"):
                # Simulate some processing
                pass
            
            mock_optimize_settings.assert_called_once_with("heavy")
            mock_force_collection.assert_called_once()
            assert mock_gc_set_threshold.call_count >= 1 # Called for optimize_gc_settings and restore
            mock_gc_get_threshold.assert_called_once() # To save original thresholds


class TestGlobalFunctions:
    """Test global convenience functions."""
    
    def test_get_memory_optimizer(self):
        """Test global memory optimizer access."""
        # Reset global _memory_optimizer before testing to ensure a clean state
        with patch('app.core.memory_optimizer._memory_optimizer', None):
            optimizer1 = get_memory_optimizer() # This is app.core.memory_optimizer.get_memory_optimizer
            optimizer2 = get_memory_optimizer()
            
            # Should return same instance
            assert optimizer1 is optimizer2
            assert isinstance(optimizer1, MemoryOptimizer)
    
    @patch('app.core.memory_utils.get_memory_optimizer') # Corrected patch target
    def test_optimize_memory_function(self, mock_get_optimizer_in_utils):
        """Test global optimize_memory function."""
        mock_optimizer_instance = Mock(spec=MemoryOptimizer)
        mock_optimizer_instance.optimize_memory.return_value = {'result': 'mocked_optimize_memory_success'} # Distinct return
        mock_get_optimizer_in_utils.return_value = mock_optimizer_instance
        
        result = optimize_memory() # This is app.core.memory_utils.optimize_memory
        
        assert result == {'result': 'mocked_optimize_memory_success'} # Assert against mock's return
        mock_optimizer_instance.optimize_memory.assert_called_once()
    
    @patch('app.core.memory_utils.get_memory_optimizer') # Corrected patch target
    def test_get_memory_stats_function(self, mock_get_optimizer_in_utils):
        """Test global get_memory_stats function."""
        mock_optimizer_instance = Mock(spec=MemoryOptimizer)
        mock_optimizer_instance.get_optimization_stats.return_value = {'stats': 'mocked_get_stats_success'} # Distinct return
        mock_get_optimizer_in_utils.return_value = mock_optimizer_instance
        
        result = get_memory_stats() # This is app.core.memory_utils.get_memory_stats
        
        assert result == {'stats': 'mocked_get_stats_success'} # Assert against mock's return
        mock_optimizer_instance.get_optimization_stats.assert_called_once()
    
    @patch('app.core.memory_utils.get_memory_optimizer') # Corrected patch target
    def test_start_memory_monitoring_function(self, mock_get_optimizer_in_utils):
        """Test global start_memory_monitoring function."""
        mock_optimizer_instance = Mock(spec=MemoryOptimizer)
        mock_get_optimizer_in_utils.return_value = mock_optimizer_instance
        
        start_memory_monitoring() # This is app.core.memory_utils.start_memory_monitoring
        
        mock_optimizer_instance.start_optimization.assert_called_once()
    
    @patch('app.core.memory_utils.get_memory_optimizer') # Corrected patch target
    def test_stop_memory_monitoring_function(self, mock_get_optimizer_in_utils):
        """Test global stop_memory_monitoring function."""
        mock_optimizer_instance = Mock(spec=MemoryOptimizer)
        mock_get_optimizer_in_utils.return_value = mock_optimizer_instance
        
        stop_memory_monitoring() # This is app.core.memory_utils.stop_memory_monitoring
        
        mock_optimizer_instance.stop_optimization.assert_called_once()


class TestMemoryOptimizedDecorator:
    """Test memory optimization decorator."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Create a mock config to avoid relying on global get_config()
        self.mock_config = {
            'memory_optimizer': {
                'doc_gc_threshold_mb': 50,
                'some_other_setting': True 
            },
            'performance': { 
                 'parallel_processing': {}
            }
        }
        # Patch get_config to return our mock_config for the MemoryOptimizer instance
        # This patch is specific to the instantiation of MemoryOptimizer if it calls get_config()
        # For the decorator itself, the relevant patch is for get_memory_optimizer in memory_utils
        with patch('app.core.memory_optimizer.get_config', return_value=self.mock_config):
             # This optimizer instance is used if the context manager's __enter__ needs to return it
             # or if other direct interactions are needed.
            self.optimizer = MemoryOptimizer()

    @patch('app.core.memory_utils.get_memory_optimizer') # Corrected patch target
    def test_memory_optimized_decorator(self, mock_get_optimizer_in_utils):
        """Test memory optimization decorator functionality."""
        mock_optimizer_instance = Mock(spec=MemoryOptimizer)
        mock_context_manager = MagicMock()
        # Configure the __enter__ and __exit__ methods for the context manager mock
        mock_optimizer_instance.optimized_processing.return_value = mock_context_manager
        # mock_context_manager itself is what is returned by __enter__
        mock_context_manager.__enter__.return_value = self.optimizer # or a new mock if needed
        mock_context_manager.__exit__.return_value = None # __exit__ should return None or bool

        mock_get_optimizer_in_utils.return_value = mock_optimizer_instance
        
        @memory_optimized("heavy") # Decorator from app.core.memory_utils
        def test_function(value):
            return value * 2
        
        result = test_function(5)
        
        assert result == 10
        mock_optimizer_instance.optimized_processing.assert_called_once_with("heavy")
        mock_context_manager.__enter__.assert_called_once()
        mock_context_manager.__exit__.assert_called_once()
    
    def test_memory_optimized_decorator_default_mode(self):
        """Test memory optimization decorator with default mode."""
        @memory_optimized()
        def test_function():
            return "test"
        
        # Should not raise errors
        result = test_function()
        assert result == "test"


class TestIntegration:
    """Integration tests for memory optimization components."""

    def setup_method(self):
        # Ensure a clean global optimizer for each integration test
        self.optimizer_patcher = patch('app.core.memory_optimizer._memory_optimizer', None)
        self.optimizer_patcher.start() # Start the patch
        # Now that _memory_optimizer is None, get_memory_optimizer() will create a new instance
        self.optimizer = get_memory_optimizer() 
        self.optimizer.config = {'doc_gc_threshold_mb': 1} # simple config for tests

    def teardown_method(self):
        if hasattr(self.optimizer, 'monitor') and self.optimizer.monitor._monitoring:
            # Ensure monitoring is stopped on the instance we used in the test
            self.optimizer.stop_optimization() 
        
        # Stop the patcher that was started in setup_method
        self.optimizer_patcher.stop() 

    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    def test_full_optimization_cycle(self, mock_psutil_vm, mock_psutil_proc):
        """Test a full cycle: start, optimize, get stats, stop."""
        # Mock psutil calls
        mock_mem_info = Mock()
        mock_mem_info.rss = 100 * 1024 * 1024
        mock_psutil_proc.return_value.memory_info.return_value = mock_mem_info
        mock_sys_mem = Mock()
        mock_sys_mem.percent = 50.0
        mock_sys_mem.available = 2000 * 1024 * 1024
        mock_psutil_vm.return_value = mock_sys_mem

        self.optimizer.start_optimization()
        assert self.optimizer.monitor._monitoring

        with patch('gc.get_objects', side_effect=[[object()] * 10, [object()] * 5]):
            opt_result = self.optimizer.optimize_memory()
        assert 'start_memory_mb' in opt_result

        stats = self.optimizer.get_optimization_stats()
        assert 'gc_stats' in stats
        assert 'memory_summary' in stats

        self.optimizer.stop_optimization()
        assert not self.optimizer.monitor._monitoring

    def test_concurrent_operations(self):
        """Test thread safety of optimizer operations."""
        # This test will rely on the internal locks of components like LRUCache, GCManager, Monitor
        # if they are called indirectly. Here we mostly test MemoryOptimizer's own methods if they had shared state.
        # For now, MemoryOptimizer methods like optimize_memory, get_stats are themselves synchronous
        # and rely on underlying components being thread-safe.
        
        # Example: if optimize_memory modified shared state on self.optimizer directly without lock
        # this test would be more relevant for self.optimizer itself.
        # We can, however, test if concurrent calls to get_memory_optimizer() behave as expected (singleton)

        optimizers = []
        def worker():
            # Simulate getting and using optimizer concurrently
            opt = get_memory_optimizer() 
            optimizers.append(opt)
            opt.get_optimization_stats() # Access some state

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads: t.start()
        for t in threads: t.join()

        assert len(optimizers) == 5
        # All threads should get the same global optimizer instance
        if optimizers:
            first_optimizer_id = id(optimizers[0])
            assert all(id(opt) == first_optimizer_id for opt in optimizers)
        
        # Ensure the global optimizer is reset for subsequent tests
        # Correct way to start and stop a patcher inline:
        temp_patcher = patch('app.core.memory_optimizer._memory_optimizer', None)
        temp_patcher.start()
        temp_patcher.stop()

    @patch('psutil.Process')
    @patch('psutil.virtual_memory')
    @patch.object(memory_logger, 'warning') # Patch the logger used by MemoryMonitor
    def test_memory_threshold_handling(self, mock_logger_warning, mock_psutil_vm, mock_psutil_proc):
        """Test how MemoryMonitor (via MemoryOptimizer) logs threshold breaches."""
        # Mock config for MemoryOptimizer to ensure it gets 'performance' or 'memory_optimizer'
        # The setup_method for TestIntegration already re-initializes optimizer and sets a basic config.
        
        # Setup monitor part of the optimizer
        self.optimizer.monitor.monitoring_interval = 0.01 # Fast monitoring
        self.optimizer.monitor._warning_threshold = 70
        self.optimizer.monitor._critical_threshold = 85
        
        # Mock psutil calls for MemoryMonitor.get_current_metrics
        mock_proc_mem_info = Mock()
        mock_proc_mem_info.rss = 100 * 1024 * 1024 
        mock_proc_mem_info.vms = 200 * 1024 * 1024
        mock_psutil_proc.return_value.memory_info.return_value = mock_proc_mem_info
        
        mock_sys_mem_info = Mock()
        mock_sys_mem_info.available = 2000 * 1024 * 1024
        
        # Initial state: below thresholds
        mock_sys_mem_info.percent = 60.0
        mock_psutil_vm.return_value = mock_sys_mem_info
        
        # Removed lines related to optimizer._auto_optimize, _optimization_interval, _last_optimization

        self.optimizer.start_optimization() # Starts the monitor
        time.sleep(0.05) # Let monitor run once (initial metrics)

        # Simulate high memory usage (warning)
        mock_sys_mem_info.percent = 75.0 # Above warning (70), below critical (85)
        mock_psutil_vm.return_value = mock_sys_mem_info
        time.sleep(0.05) # Let monitor run again
        
        # Check if logger.warning was called by MemoryMonitor for high memory
        high_mem_calls = [call for call in mock_logger_warning.call_args_list if "High memory usage detected" in call.args[0]]
        assert len(high_mem_calls) > 0, "MemoryMonitor should log a warning for high memory usage"

        # Simulate critical memory usage
        mock_logger_warning.reset_mock()
        mock_sys_mem_info.percent = 90.0 # Above critical (85)
        mock_psutil_vm.return_value = mock_sys_mem_info
        time.sleep(0.05) # Let monitor run again

        critical_mem_calls = [call for call in mock_logger_warning.call_args_list if "Critical memory usage detected" in call.args[0]]
        assert len(critical_mem_calls) > 0, "MemoryMonitor should log a warning for critical memory usage"
            
        # Removed assertions for mock_force_gc.assert_called() as it's not auto-triggered by MemoryOptimizer

        self.optimizer.stop_optimization()


if __name__ == "__main__":
    pytest.main([__file__]) 