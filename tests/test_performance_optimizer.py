"""
Unit tests for Performance Optimizer - Session 3
Tests parallel processing, document streaming, and batch processing functionality.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch
import threading

from app.core.performance_optimizer import (
    ParallelProcessor,
    DocumentStreamer,
    BatchEngine,
    LoadBalancer,
    ProcessingTask,
    ProcessingResult,
    ProcessingMode,
    get_parallel_processor,
    get_batch_engine
)


class TestDocumentStreamer:
    """Test DocumentStreamer functionality."""

    def test_stream_document(self):
        """Test document streaming with chunks."""
        streamer = DocumentStreamer(chunk_size=10)  # Small chunks for testing
        
        # Create temporary test file
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
            test_data = b"Hello World! This is a test document for streaming."
            temp_file.write(test_data)
            temp_path = Path(temp_file.name)
        
        try:
            # Stream the document
            chunks = list(streamer.stream_document(temp_path))
            
            # Verify chunks
            assert len(chunks) > 0
            reconstructed = b''.join(chunks)
            assert reconstructed == test_data
            
        finally:
            temp_path.unlink()  # Clean up
    
    def test_estimate_processing_time(self):
        """Test processing time estimation."""
        streamer = DocumentStreamer()
        
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
            temp_file.write(b"x" * (1024 * 1024))  # 1MB file
            temp_path = Path(temp_file.name)
        
        try:
            estimated_time = streamer.estimate_processing_time(temp_path)
            assert estimated_time > 0
            assert isinstance(estimated_time, float)
            
        finally:
            temp_path.unlink()


class TestLoadBalancer:
    """Test LoadBalancer functionality."""

    def test_get_optimal_workers(self):
        """Test optimal worker calculation."""
        balancer = LoadBalancer()
        
        # Test with small tasks
        workers = balancer.get_optimal_workers(task_count=5, avg_task_size_mb=0.1)
        assert workers > 0
        assert workers <= 8  # Max workers limit
        
        # Test with large tasks
        workers_large = balancer.get_optimal_workers(task_count=2, avg_task_size_mb=10.0)
        assert workers_large > 0
        
    def test_adjust_for_current_load(self):
        """Test load adjustment functionality."""
        balancer = LoadBalancer()
        
        # Test normal case
        adjusted = balancer.adjust_for_current_load(4)
        assert adjusted > 0
        assert adjusted <= 4


class TestParallelProcessor:
    """Test ParallelProcessor functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.processor = ParallelProcessor()
        
        # Mock processor function
        def mock_processor(data, **kwargs):
            time.sleep(0.01)  # Simulate processing time
            return f"processed_{data}"
        
        self.mock_processor = mock_processor

    def test_determine_processing_mode(self):
        """Test processing mode determination."""
        # Single small task
        task = ProcessingTask("test1", "data1", estimated_size=1024)
        mode = self.processor._determine_processing_mode([task])
        assert mode == ProcessingMode.SEQUENTIAL
        
        # Single large task
        large_task = ProcessingTask("test2", "data2", estimated_size=15*1024*1024)
        mode = self.processor._determine_processing_mode([large_task])
        assert mode == ProcessingMode.PARALLEL_THREADS
        
        # Multiple tasks
        tasks = [
            ProcessingTask("test3", "data3", estimated_size=1024),
            ProcessingTask("test4", "data4", estimated_size=1024),
            ProcessingTask("test5", "data5", estimated_size=1024)
        ]
        mode = self.processor._determine_processing_mode(tasks)
        assert mode == ProcessingMode.PARALLEL_THREADS

    def test_process_single_document(self):
        """Test single document processing."""
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False) as temp_file:
            temp_file.write(b"test document content")
            temp_path = Path(temp_file.name)
        
        try:
            def simple_processor(file_path, **kwargs):
                time.sleep(0.001)  # Simulate a small amount of work
                return f"processed {file_path.name}"
            
            result = self.processor.process_single_document(temp_path, simple_processor)
            
            assert isinstance(result, ProcessingResult)
            assert result.success is True
            assert result.duration > 0
            assert "processed" in result.result
            
        finally:
            temp_path.unlink()

    def test_process_batch_sequential(self):
        """Test batch processing in sequential mode."""
        tasks = [
            ProcessingTask("task1", "data1", estimated_size=100),
            ProcessingTask("task2", "data2", estimated_size=100)
        ]
        
        results = self.processor.process_batch(tasks, self.mock_processor)
        
        assert len(results) == 2
        assert all(isinstance(r, ProcessingResult) for r in results)
        assert all(r.success for r in results)
        assert all("processed_" in r.result for r in results)

    def test_process_batch_parallel_threads(self):
        """Test batch processing with parallel threads."""
        tasks = [
            ProcessingTask(f"task{i}", f"data{i}", estimated_size=1024)
            for i in range(5)
        ]
        
        results = self.processor.process_batch(tasks, self.mock_processor)
        
        assert len(results) == 5
        assert all(isinstance(r, ProcessingResult) for r in results)
        assert all(r.success for r in results)

    def test_execute_task(self):
        """Test individual task execution."""
        task = ProcessingTask("test_task", "test_data")
        
        result = self.processor._execute_task(task, self.mock_processor)
        
        assert isinstance(result, ProcessingResult)
        assert result.task_id == "test_task"
        assert result.success is True
        assert result.duration > 0

    def test_execute_task_with_error(self):
        """Test task execution with error handling."""
        def error_processor(data, **kwargs):
            raise ValueError("Test error")
        
        task = ProcessingTask("error_task", "error_data")
        result = self.processor._execute_task(task, error_processor)
        
        assert isinstance(result, ProcessingResult)
        assert result.success is False
        assert result.error is not None
        assert "Test error" in result.error


class TestBatchEngine:
    """Test BatchEngine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.batch_engine = BatchEngine()
        
        def mock_processor(data, **kwargs):
            time.sleep(0.01)
            return f"batch_processed_{data}"
        
        self.mock_processor = mock_processor

    def test_submit_batch(self):
        """Test batch submission."""
        tasks = [
            ProcessingTask("batch_task1", "data1"),
            ProcessingTask("batch_task2", "data2")
        ]
        
        batch_id = self.batch_engine.submit_batch(
            "test_batch",
            tasks,
            self.mock_processor,
            priority=1
        )
        
        assert batch_id == "test_batch"
        
        # Check batch status
        status = self.batch_engine.get_batch_status("test_batch")
        assert status is not None
        assert status['status'] == 'queued'
        assert status['priority'] == 1

    def test_process_next_batch(self):
        """Test batch processing."""
        tasks = [
            ProcessingTask("process_task1", "data1"),
            ProcessingTask("process_task2", "data2")
        ]
        
        # Submit batch
        self.batch_engine.submit_batch(
            "process_batch",
            tasks,
            self.mock_processor
        )
        
        # Process the batch
        processed_batch_id = self.batch_engine.process_next_batch()
        
        assert processed_batch_id == "process_batch"
        
        # Check results
        results = self.batch_engine.get_batch_results("process_batch")
        assert results is not None
        assert len(results) == 2
        assert all(r.success for r in results)

    def test_get_queue_stats(self):
        """Test queue statistics."""
        stats = self.batch_engine.get_queue_stats()
        
        assert isinstance(stats, dict)
        assert 'queued_batches' in stats
        assert 'active_batches' in stats
        assert 'completed_batches' in stats
        assert 'total_batches' in stats

    def test_process_empty_queue(self):
        """Test processing when queue is empty."""
        result = self.batch_engine.process_next_batch()
        assert result is None


class TestProcessingDataClasses:
    """Test data classes and structures."""

    def test_processing_task_creation(self):
        """Test ProcessingTask creation."""
        task = ProcessingTask(
            task_id="test_task",
            data="test_data",
            priority=2,
            estimated_size=1024,
            metadata={"type": "test"}
        )
        
        assert task.task_id == "test_task"
        assert task.data == "test_data"
        assert task.priority == 2
        assert task.estimated_size == 1024
        assert task.metadata["type"] == "test"

    def test_processing_task_default_metadata(self):
        """Test ProcessingTask with default metadata."""
        task = ProcessingTask("test", "data")
        assert task.metadata == {}

    def test_processing_result_creation(self):
        """Test ProcessingResult creation."""
        result = ProcessingResult(
            task_id="result_task",
            result="test_result",
            success=True,
            duration=1.5,
            memory_used=128.0,
            metadata={"processed": True}
        )
        
        assert result.task_id == "result_task"
        assert result.result == "test_result"
        assert result.success is True
        assert result.duration == 1.5
        assert result.memory_used == 128.0
        assert result.metadata["processed"] is True


class TestGlobalInstances:
    """Test global instance functions."""

    def test_get_parallel_processor(self):
        """Test global parallel processor instance."""
        processor1 = get_parallel_processor()
        processor2 = get_parallel_processor()
        
        assert processor1 is processor2  # Same instance
        assert isinstance(processor1, ParallelProcessor)

    def test_get_batch_engine(self):
        """Test global batch engine instance."""
        engine1 = get_batch_engine()
        engine2 = get_batch_engine()
        
        assert engine1 is engine2  # Same instance
        assert isinstance(engine1, BatchEngine)


class TestIntegration:
    """Integration tests for the complete system."""

    def test_end_to_end_processing(self):
        """Test complete end-to-end processing workflow."""
        processor = get_parallel_processor()
        
        def integration_processor(data, **kwargs):
            time.sleep(0.01)
            return {"processed": data, "timestamp": time.time()}
        
        # Create test tasks
        tasks = [
            ProcessingTask(f"integration_task_{i}", f"data_{i}", estimated_size=1024)
            for i in range(3)
        ]
        
        # Process batch
        results = processor.process_batch(tasks, integration_processor)
        
        # Verify results
        assert len(results) == 3
        assert all(r.success for r in results)
        assert all("processed" in r.result for r in results)
        assert all(r.duration > 0 for r in results)

    def test_performance_improvement_simulation(self):
        """Simulate performance improvements with parallel processing."""
        processor = ParallelProcessor()
        
        def slow_processor(data, **kwargs):
            time.sleep(0.05)  # Simulate slow processing
            return f"slow_processed_{data}"
        
        # Sequential processing simulation
        tasks = [ProcessingTask(f"seq_task_{i}", f"data_{i}") for i in range(4)]
        
        start_time = time.time()
        results = processor._process_sequential(tasks, slow_processor)
        sequential_time = time.time() - start_time
        
        # Parallel processing simulation
        start_time = time.time()
        results_parallel = processor._process_parallel_threads(tasks, slow_processor, 2)
        parallel_time = time.time() - start_time
        
        # Verify both produce same results
        assert len(results) == len(results_parallel) == 4
        assert all(r.success for r in results)
        assert all(r.success for r in results_parallel)
        
        # Parallel should be faster (allowing for some overhead)
        assert parallel_time < sequential_time * 0.8  # At least 20% improvement


if __name__ == "__main__":
    pytest.main([__file__]) 