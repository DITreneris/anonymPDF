"""
Comprehensive tests for memory utilities module.
Tests cover convenience functions, decorators, and memory optimization integration.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from app.core.memory_utils import (
    optimize_memory,
    get_memory_stats,
    start_memory_monitoring,
    stop_memory_monitoring,
    memory_optimized
)


class TestMemoryUtilityFunctions:
    """Test convenience functions."""

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_optimize_memory(self, mock_get_optimizer):
        """Test optimize_memory function."""
        mock_optimizer = Mock()
        mock_optimizer.optimize_memory.return_value = {"freed_mb": 50}
        mock_get_optimizer.return_value = mock_optimizer
        
        result = optimize_memory()
        
        mock_get_optimizer.assert_called_once()
        mock_optimizer.optimize_memory.assert_called_once()
        assert result == {"freed_mb": 50}

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_get_memory_stats(self, mock_get_optimizer):
        """Test get_memory_stats function."""
        mock_optimizer = Mock()
        expected_stats = {"total_memory": 8192}
        mock_optimizer.get_optimization_stats.return_value = expected_stats
        mock_get_optimizer.return_value = mock_optimizer
        
        result = get_memory_stats()
        
        mock_get_optimizer.assert_called_once()
        assert result == expected_stats

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_start_memory_monitoring(self, mock_get_optimizer):
        """Test start_memory_monitoring function."""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        
        start_memory_monitoring()
        
        mock_optimizer.start_optimization.assert_called_once()

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_stop_memory_monitoring(self, mock_get_optimizer):
        """Test stop_memory_monitoring function."""
        mock_optimizer = Mock()
        mock_get_optimizer.return_value = mock_optimizer
        
        stop_memory_monitoring()
        
        mock_optimizer.stop_optimization.assert_called_once()


class TestMemoryOptimizedDecorator:
    """Test memory_optimized decorator."""

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_decorator_default_mode(self, mock_get_optimizer):
        """Test decorator with default mode."""
        mock_optimizer = Mock()
        # Fix context manager protocol for Mock objects
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=False)
        mock_optimizer.optimized_processing.return_value = mock_context_manager
        mock_get_optimizer.return_value = mock_optimizer
        
        @memory_optimized()
        def test_func(x):
            return x * 2
        
        result = test_func(5)
        
        assert result == 10
        mock_optimizer.optimized_processing.assert_called_with("normal")

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_decorator_custom_mode(self, mock_get_optimizer):
        """Test decorator with custom mode."""
        mock_optimizer = Mock()
        # Fix context manager protocol for Mock objects
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=False)
        mock_optimizer.optimized_processing.return_value = mock_context_manager
        mock_get_optimizer.return_value = mock_optimizer
        
        @memory_optimized(processing_mode="aggressive")
        def test_func():
            return "result"
        
        result = test_func()
        
        assert result == "result"
        mock_optimizer.optimized_processing.assert_called_with("aggressive")

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_decorator_preserves_metadata(self, mock_get_optimizer):
        """Test decorator preserves function metadata."""
        mock_optimizer = Mock()
        mock_context = Mock()
        mock_optimizer.optimized_processing.return_value = mock_context
        mock_get_optimizer.return_value = mock_optimizer
        
        @memory_optimized()
        def original_function():
            """Original docstring."""
            return "result"
        
        assert original_function.__name__ == "original_function"
        assert original_function.__doc__ == "Original docstring."

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_decorator_handles_missing_attributes(self, mock_get_optimizer):
        """Test decorator with functions missing __name__ or __doc__."""
        mock_optimizer = Mock()
        # Fix context manager protocol for Mock objects
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=False)
        mock_optimizer.optimized_processing.return_value = mock_context_manager
        mock_get_optimizer.return_value = mock_optimizer
        
        mock_func = Mock()
        if hasattr(mock_func, '__name__'):
            delattr(mock_func, '__name__')
        if hasattr(mock_func, '__doc__'):
            delattr(mock_func, '__doc__')
        mock_func.return_value = "result"
        
        decorated_func = memory_optimized()(mock_func)
        result = decorated_func()
        
        assert result == "result"


class TestIntegrationScenarios:
    """Integration test scenarios for memory utilities."""

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_multiple_convenience_functions_same_optimizer(self, mock_get_optimizer):
        """Test that multiple convenience functions use the same optimizer instance."""
        # Setup mock
        mock_optimizer = Mock()
        mock_optimizer.optimize_memory.return_value = {}
        mock_optimizer.get_optimization_stats.return_value = {}
        mock_get_optimizer.return_value = mock_optimizer
        
        # Call multiple functions
        optimize_memory()
        get_memory_stats()
        start_memory_monitoring()
        stop_memory_monitoring()
        
        # Should use same optimizer instance
        assert mock_get_optimizer.call_count == 4

    @patch('app.core.memory_utils.get_memory_optimizer')
    def test_decorator_and_functions_integration(self, mock_get_optimizer):
        """Test integration between decorator and utility functions."""
        # Setup mock
        mock_optimizer = Mock()
        # Fix context manager protocol for Mock objects
        mock_context_manager = Mock()
        mock_context_manager.__enter__ = Mock(return_value=mock_context_manager)
        mock_context_manager.__exit__ = Mock(return_value=False)
        mock_optimizer.optimized_processing.return_value = mock_context_manager
        mock_optimizer.optimize_memory.return_value = {"status": "success"}
        mock_get_optimizer.return_value = mock_optimizer
        
        @memory_optimized()
        def process_data():
            # Within the decorated function, call utility functions
            optimize_memory()
            return "processed"
        
        result = process_data()
        
        assert result == "processed"
        # Both the decorator and the utility function should have accessed the optimizer
        assert mock_get_optimizer.call_count >= 2

    def test_real_integration_with_actual_memory_optimizer(self):
        """Integration test with real memory optimizer (if available)."""
        try:
            # Test with real optimizer
            stats = get_memory_stats()
            assert isinstance(stats, dict)
            
            # Test decorator with real optimizer
            @memory_optimized()
            def simple_function():
                return "test"
            
            result = simple_function()
            assert result == "test"
            
        except ImportError:
            # Skip if memory optimizer is not available
            pytest.skip("Memory optimizer not available for integration test")


def test_module_imports():
    """Test that all expected functions are importable from the module."""
    from app.core.memory_utils import (
        optimize_memory,
        get_memory_stats, 
        start_memory_monitoring,
        stop_memory_monitoring,
        memory_optimized
    )
    
    # Verify all functions are callable
    assert callable(optimize_memory)
    assert callable(get_memory_stats)
    assert callable(start_memory_monitoring)
    assert callable(stop_memory_monitoring)
    assert callable(memory_optimized) 