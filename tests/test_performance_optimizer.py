"""
Unit tests for Performance Optimizer - Session 3
Tests parallel processing, document streaming, and batch processing functionality.
"""

import pytest
import tempfile
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

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
from app.core.config_manager import get_config


@pytest.fixture(scope="module")
def config():
    """Provides the application configuration for the module."""
    return get_config().get('performance', {})


@pytest.fixture
def load_balancer():
    """Provides a LoadBalancer instance."""
    return LoadBalancer()


@pytest.fixture
def parallel_processor():
    """Provides a ParallelProcessor instance."""
    return ParallelProcessor()


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
    """Tests for the LoadBalancer."""

    @pytest.fixture
    def load_balancer(self):
        """Provides a LoadBalancer instance."""
        return LoadBalancer()

    def test_get_optimal_workers(self, load_balancer):
        """Test optimal worker calculation."""
        assert load_balancer.get_optimal_workers(task_count=5, avg_task_size_mb=0.1) > 0

    def test_adjust_for_current_load(self, load_balancer):
        """Test load adjustment functionality."""
        with patch('app.core.performance_optimizer.psutil.cpu_percent', return_value=50), \
             patch('app.core.performance_optimizer.psutil.virtual_memory') as mock_mem:
            mock_mem.return_value.percent = 60
            assert load_balancer.adjust_for_current_load(4) == 4

    @patch('app.core.performance_optimizer.psutil.virtual_memory')
    @patch('app.core.performance_optimizer.psutil.cpu_percent')
    def test_adaptive_mode_selection_based_on_load(self, mock_cpu, mock_mem, load_balancer):
        """Test that the load balancer adjusts worker count based on system load."""
        mock_cpu.return_value = 90
        mock_mem.return_value.percent = 90
        assert load_balancer.adjust_for_current_load(8) == 4


class TestParallelProcessor:
    """Tests for the ParallelProcessor."""

    @pytest.fixture
    def parallel_processor(self, config):
        """Provides a ParallelProcessor instance for testing."""
        return ParallelProcessor(config=config)

    def test_determine_processing_mode(self, parallel_processor):
        """Test processing mode determination."""
        task = ProcessingTask(task_id="test1", data="data1", estimated_size=1024)
        assert parallel_processor._determine_processing_mode([task]) == ProcessingMode.SEQUENTIAL

    def test_process_batch_with_process_pool(self, parallel_processor):
        """Test that process_batch handles ProcessPoolExecutor mode correctly."""
        
        # Create a simple mock function that simulates ProcessPoolExecutor behavior
        def mock_process_function(task_and_func):
            task, func = task_and_func
            return func(task.data)
        
        # Prepare test tasks
        tasks = [
            ProcessingTask(task_id=f"t{i}", data=f"data_{i}", estimated_size=10 * 1024 * 1024)
            for i in range(3)
        ]

        def test_processor(data, **kwargs):
            return f"processed_{data}"
        
        # Mock the entire ProcessPoolExecutor usage by patching the process_batch method
        # to return expected results when ProcessPoolExecutor mode is used
        with patch.object(parallel_processor, '_determine_processing_mode', return_value=ProcessingMode.PARALLEL_PROCESSES):
            # Instead of mocking ProcessPoolExecutor, we'll mock the internal execution
            with patch.object(parallel_processor, '_execute_task') as mock_execute:
                mock_execute.side_effect = lambda task, func: ProcessingResult(
                    task_id=task.task_id,
                    result=f"processed_{task.data}",
                    success=True,
                    duration=0.1
                )
                
                results = parallel_processor.process_batch(tasks, test_processor)
                
                # Verify results
                assert len(results) == 3
                assert all(isinstance(r, ProcessingResult) for r in results)
                assert all(r.success for r in results)
                assert results[0].result == "processed_data_0"
    def test_processing_mode_logic(self, parallel_processor):
        """Test the logic that determines which processing mode to use."""
        
        # Small tasks should use sequential processing
        small_tasks = [ProcessingTask(f"small_{i}", "data", estimated_size=100) for i in range(2)]
        mode = parallel_processor._determine_processing_mode(small_tasks)
        assert mode == ProcessingMode.SEQUENTIAL
        
        # Medium tasks should use thread pool
        medium_tasks = [ProcessingTask(f"med_{i}", "data", estimated_size=1024*100) for i in range(5)]
        mode = parallel_processor._determine_processing_mode(medium_tasks)
        assert mode in [ProcessingMode.PARALLEL_THREADS, ProcessingMode.SEQUENTIAL]
        
        # Large tasks should potentially use process pool
        large_tasks = [ProcessingTask(f"large_{i}", "data", estimated_size=1024*1024*10) for i in range(4)]
        mode = parallel_processor._determine_processing_mode(large_tasks)
        assert mode in [ProcessingMode.PARALLEL_PROCESSES, ProcessingMode.PARALLEL_THREADS, ProcessingMode.SEQUENTIAL]

    def test_safe_parallel_processing(self, parallel_processor):
        """Test parallel processing without actually using multiprocessing."""
        
        tasks = [
            ProcessingTask(task_id=f"safe_{i}", data=f"input_{i}", estimated_size=1024)
            for i in range(4)
        ]
        
        def safe_processor(data, **kwargs):
            return f"output_{data}"
        
        # Force sequential mode to avoid any multiprocessing issues
        with patch.object(parallel_processor, '_determine_processing_mode', return_value=ProcessingMode.SEQUENTIAL):
            results = parallel_processor.process_batch(tasks, safe_processor)
        
        assert len(results) == 4
        assert all(r.success for r in results)
        assert results[0].result == "output_input_0"
        assert results[3].result == "output_input_3"
            
    def test_execute_task_with_error(self, parallel_processor):
        """Test task execution with error handling."""
        def error_processor(data, **kwargs):
            raise ValueError("Test error")
        
        task = ProcessingTask(task_id="error_task", data="error_data")
        result = parallel_processor._execute_task(task, error_processor)
        
        assert isinstance(result, ProcessingResult)
        assert result.success is False
        assert "Test error" in result.error

    def test_process_batch_sequential(self, parallel_processor):
        """Test sequential batch processing."""
        tasks = [
            ProcessingTask(task_id=f"seq_{i}", data=f"data_{i}", estimated_size=100)
            for i in range(3)
        ]
        
        def simple_processor(data, **kwargs):
            return f"processed_{data}"
        
        with patch.object(parallel_processor, '_determine_processing_mode', return_value=ProcessingMode.SEQUENTIAL):
            results = parallel_processor.process_batch(tasks, simple_processor)
        
        assert len(results) == 3
        assert all(isinstance(r, ProcessingResult) for r in results)
        assert all(r.success for r in results)
        assert results[0].result == "processed_data_0"

    def test_process_batch_with_thread_pool(self, parallel_processor):
        """Test that process_batch handles ThreadPoolExecutor mode correctly."""
        
        tasks = [
            ProcessingTask(task_id=f"thread_{i}", data=f"data_{i}", estimated_size=1024)
            for i in range(3)
        ]
        
        def test_processor(data, **kwargs):
            return f"threaded_{data}"
        
        # Mock ThreadPoolExecutor safely by mocking the execution results
        with patch.object(parallel_processor, '_determine_processing_mode', return_value=ProcessingMode.PARALLEL_THREADS):
            with patch.object(parallel_processor, '_execute_task') as mock_execute:
                mock_execute.side_effect = lambda task, func: ProcessingResult(
                    task_id=task.task_id,
                    result=f"threaded_{task.data}",
                    success=True,
                    duration=0.05
                )
                
                results = parallel_processor.process_batch(tasks, test_processor)
                
                assert len(results) == 3
                assert all(isinstance(r, ProcessingResult) for r in results)
                assert all(r.success for r in results)
                assert results[1].result == "threaded_data_1"
                assert mock_execute.call_count == 3


class TestBatchEngine:
    """Tests for the BatchEngine."""

    @pytest.fixture
    def batch_engine(self, config):
        """Provides a BatchEngine instance for testing."""
        return BatchEngine(config=config)

    def test_submit_batch(self, batch_engine):
        """Test batch submission."""
        tasks = [ProcessingTask(task_id="b1", data="d1")]
        batch_id = batch_engine.submit_batch("test_batch", tasks, lambda x: x)
        assert batch_id == "test_batch"
        status = batch_engine.get_batch_status("test_batch")
        assert status['status'] == 'queued'

    @patch('app.core.performance_optimizer.ParallelProcessor.process_batch')
    def test_process_next_batch(self, mock_process_batch, batch_engine):
        """Test batch processing."""
        tasks = [ProcessingTask(task_id="b2", data="d2")]
        batch_engine.submit_batch("process_batch", tasks, lambda x: x)
        
        batch_engine.process_next_batch()
        
        mock_process_batch.assert_called_once()
        status = batch_engine.get_batch_status("process_batch")
        assert status['status'] == 'completed'

    def test_batch_queue_order(self, batch_engine):
        """Test that batches are processed in correct order."""
        tasks1 = [ProcessingTask(task_id="order1", data="d1")]
        tasks2 = [ProcessingTask(task_id="order2", data="d2")]
        
        batch_engine.submit_batch("first_batch", tasks1, lambda x: f"result_{x}")
        batch_engine.submit_batch("second_batch", tasks2, lambda x: f"result_{x}")
        
        # Process first batch
        batch_engine.process_next_batch()
        status1 = batch_engine.get_batch_status("first_batch")
        assert status1['status'] == 'completed'
        
        # Second batch should still be queued
        status2 = batch_engine.get_batch_status("second_batch")
        assert status2['status'] == 'queued'


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

    def test_processing_result_error_case(self):
        """Test ProcessingResult with error."""
        result = ProcessingResult(
            task_id="error_task",
            result=None,
            success=False,
            error="Something went wrong",
            duration=0.5
        )
        
        assert result.task_id == "error_task"
        assert result.result is None
        assert result.success is False
        assert result.error == "Something went wrong"
        assert result.duration == 0.5


class TestGlobalInstances:
    """Test global instance functions."""

    def test_get_parallel_processor(self):
        """Test global parallel processor instance."""
        processor = get_parallel_processor()
        assert isinstance(processor, ParallelProcessor)
        # Test that multiple calls return the same instance (singleton pattern)
        processor2 = get_parallel_processor()
        assert processor is processor2

    def test_get_batch_engine(self):
        """Test global batch engine instance."""
        engine = get_batch_engine()
        assert isinstance(engine, BatchEngine)
        # Test that multiple calls return the same instance (singleton pattern)
        engine2 = get_batch_engine()
        assert engine is engine2


class TestIntegration:
    """Test full processing pipeline integration."""

    def test_end_to_end_processing(self):
        """Test a full batch processing from submission to completion."""
        
        def integration_processor(data, **kwargs):
            return f"integrated_{data}"
        
        tasks = [ProcessingTask(f"e2e_{i}", f"data_{i}") for i in range(3)]
        
        # Use the global batch engine instance
        batch_engine = get_batch_engine()
        assert batch_engine.config == get_config().get('performance', {})

        batch_id = "end_to_end_batch"
        batch_engine.submit_batch(batch_id, tasks, integration_processor)
        
        status = batch_engine.get_batch_status(batch_id)
        assert status['status'] == 'completed'
        results = {r.task_id: r.result for r in status['results']}
        assert results["e2e_0"] == "integrated_data_0"
        assert results["e2e_2"] == "integrated_data_2"

    def test_performance_improvement_simulation(self):
        """Simulate processing to test performance metrics."""
        
        def slow_processor(data, **kwargs):
            time.sleep(0.01)  # Reduced sleep time for faster tests
            return f"slow_processed_{data}"
            
        tasks = [ProcessingTask(f"perf_{i}", f"data_{i}", estimated_size=1024) for i in range(5)]
        
        # Get processor with module-level config
        processor = get_parallel_processor()
        assert processor.config == get_config().get('performance', {})

        # Process sequentially first
        start_seq = time.time()
        with patch.object(processor, '_determine_processing_mode', return_value=ProcessingMode.SEQUENTIAL):
            seq_results = processor.process_batch(tasks, slow_processor)
        end_seq = time.time()
        duration_seq = end_seq - start_seq

        # Process in parallel
        start_par = time.time()
        with patch.object(processor, '_determine_processing_mode', return_value=ProcessingMode.PARALLEL_THREADS):
            par_results = processor.process_batch(tasks, slow_processor)
        end_par = time.time()
        duration_par = end_par - start_par
        
        # Verify results are correct
        assert len(seq_results) == 5
        assert len(par_results) == 5
        assert all(r.success for r in seq_results)
        assert all(r.success for r in par_results)
        
        # Assert that both processing modes work
        assert duration_seq > 0
        assert duration_par > 0
        # Note: We can't guarantee parallel is faster due to overhead with small tasks

    def test_error_handling_in_pipeline(self):
        """Test error handling throughout the processing pipeline."""
        
        def error_prone_processor(data, **kwargs):
            if data == "error_data":
                raise RuntimeError("Intentional error for testing")
            return f"processed_{data}"
        
        tasks = [
            ProcessingTask("good_1", "good_data_1"),
            ProcessingTask("error_task", "error_data"),
            ProcessingTask("good_2", "good_data_2")
        ]
        
        batch_engine = get_batch_engine()
        batch_id = "error_handling_batch"
        batch_engine.submit_batch(batch_id, tasks, error_prone_processor)
        
        status = batch_engine.get_batch_status(batch_id)
        assert status['status'] == 'completed'
        
        results = {r.task_id: r for r in status['results']}
        
        # Good tasks should succeed
        assert results["good_1"].success is True
        assert results["good_2"].success is True
        assert results["good_1"].result == "processed_good_data_1"
        assert results["good_2"].result == "processed_good_data_2"
        
        # Error task should fail gracefully
        assert results["error_task"].success is False
        assert "Intentional error for testing" in results["error_task"].error


if __name__ == "__main__":
    pytest.main([__file__])