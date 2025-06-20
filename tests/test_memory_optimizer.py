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
        self.mock_config = {
            'doc_gc_threshold_mb': 50,
        }
        # Patch get_config at the source to ensure the optimizer gets it
        with patch('app.core.memory_optimizer.get_config', return_value={'memory_optimizer': self.mock_config}):
            self.optimizer = MemoryOptimizer()

    def teardown_method(self):
        """Clean up after tests."""
        if self.optimizer.monitor and self.optimizer.monitor._monitoring:
             self.optimizer.stop_optimization() 
    
    def test_optimizer_initialization(self):
        """Test MemoryOptimizer initialization."""
        assert isinstance(self.optimizer.gc_manager, GarbageCollectionManager)
        assert isinstance(self.optimizer.monitor, MemoryMonitor)
        assert self.optimizer.config == self.mock_config
        assert not self.optimizer.monitor._monitoring

    def test_start_stop_optimization(self):
        """Test starting and stopping optimization (monitoring)."""
        with patch.object(self.optimizer.monitor, 'start_monitoring') as mock_start, \
             patch.object(self.optimizer.monitor, 'stop_monitoring') as mock_stop:
            self.optimizer.start_optimization()
            mock_start.assert_called_once()
            
            self.optimizer.stop_optimization()
            mock_stop.assert_called_once()
            
    @patch('app.core.memory_optimizer.MemoryMonitor.get_current_metrics')
    @patch('app.core.memory_optimizer.GarbageCollectionManager.force_collection')
    def test_optimize_memory(self, mock_force_collection, mock_get_metrics):
        """Test the main memory optimization function."""
        mock_get_metrics.side_effect = [
            MemoryMetrics(time.time(), 100, 200, 50, 1000, 100), # Start
            MemoryMetrics(time.time(), 80, 180, 40, 1200, 80)   # End
        ]
        mock_force_collection.return_value = {'memory_freed_mb': 20.0, 'objects_collected': 20}
        
        result = self.optimizer.optimize_memory()
        
        mock_force_collection.assert_called_once()
        assert mock_get_metrics.call_count == 2
        assert result['start_memory_mb'] == 100.0
        assert result['end_memory_mb'] == 80.0
        assert result['memory_saved_mb'] == pytest.approx(20.0)
        assert result['memory_freed_mb'] == 20.0

    def test_optimization_stats(self):
        """Test comprehensive optimization statistics reporting."""
        with patch.object(self.optimizer.gc_manager, 'get_gc_stats', return_value={'collections': 1}) as mock_gc_stats, \
             patch.object(self.optimizer.monitor, 'get_memory_summary', return_value={'peak_mb': 150}) as mock_mem_summary:
            
            stats = self.optimizer.get_optimization_stats()
            
            mock_gc_stats.assert_called_once()
            mock_mem_summary.assert_called_once()
            
            assert stats['gc_stats'] == {'collections': 1}
            assert stats['memory_summary'] == {'peak_mb': 150}

    @patch('app.core.memory_optimizer.gc')
    def test_optimized_processing_context(self, mock_gc):
        """Test the optimized_processing context manager."""
        mock_gc.get_threshold.return_value = (700, 10, 10)
        
        with patch.object(self.optimizer, 'tune_garbage_collector') as mock_tune_gc, \
             patch.object(self.optimizer, 'force_gc_collection') as mock_force_gc:
            
            with self.optimizer.optimized_processing(processing_mode="heavy"):
                pass 
            
            mock_tune_gc.assert_called_once_with("heavy")
            mock_force_gc.assert_called_once()
            mock_gc.set_threshold.assert_called_with(700, 10, 10)


class TestGlobalOptimizerAccess:
    """Test the singleton access pattern for the optimizer."""
    
    def test_get_memory_optimizer_singleton(self):
        """Test that get_memory_optimizer returns a singleton instance."""
        with patch('app.core.memory_optimizer._memory_optimizer', None):
            optimizer1 = get_memory_optimizer()
            optimizer2 = get_memory_optimizer()
            assert optimizer1 is optimizer2
            assert isinstance(optimizer1, MemoryOptimizer)


if __name__ == "__main__":
    pytest.main([__file__]) 