import asyncio
from pathlib import Path
import time
import sys
import os
import logging # Added for explicit logger configuration if needed

# Add project root to sys.path to allow imports from app
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

# Ensure 'processed_benchmarks' directory exists for output files
# This will be created in the workspace root where benchmark_runner.py is located
output_base_dir = project_root / "processed_benchmarks"
output_base_dir.mkdir(exist_ok=True)

try:
    from app.services.pdf_processor import PDFProcessor
    from app.core.logging import pdf_logger
    from app.core.intelligent_cache import get_intelligent_cache
    # Added for true batch benchmarking
    from app.core.performance_optimizer import PerformanceOptimizer, ProcessingTask, get_performance_optimizer 
except ImportError as e:
    print(f"Error importing project modules: {e}")
    print(f"Ensure your PYTHONPATH is set up correctly or run this script from the project root.")
    print(f"Current sys.path: {sys.path}")
    print(f"Current working directory: {os.getcwd()}")
    sys.exit(1)

async def run_benchmark_on_directory(pdf_directory_path: str, processor: PDFProcessor, run_id: str):
    """
    Processes all PDF files in a given directory using the PDFProcessor.
    """
    pdf_dir = Path(pdf_directory_path)
    if not pdf_dir.is_dir():
        pdf_logger.error(f"Provided path is not a directory: {pdf_directory_path}")
        return

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pdf_logger.warning(f"No PDF files found in directory: {pdf_directory_path}")
        return

    pdf_logger.info(f"--- Starting Benchmark Run: {run_id} for directory: {pdf_directory_path} ---")
    pdf_logger.info(f"Found {len(pdf_files)} PDF files to process.")

    overall_start_time = time.perf_counter()
    total_files_processed = 0
    
    # Create a subdirectory for this specific run's output files
    run_output_dir = output_base_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)

    for pdf_file_path in pdf_files:
        pdf_logger.info(f"Benchmarking file: {pdf_file_path.name} (from {run_id}) - First Pass")
        try:
            output_path = run_output_dir / f"anonymized_{pdf_file_path.name}"
            
            success, report = processor.anonymize_pdf(pdf_file_path, output_path)

            if success:
                pdf_logger.info(f"Successfully processed (benchmark): {pdf_file_path.name} to {output_path} - First Pass")
                # total_files_processed incremented only on the second pass to count unique files
            else:
                error_detail = report.get('details', 'No details') if isinstance(report, dict) else str(report)
                pdf_logger.error(f"Failed to process (benchmark): {pdf_file_path.name} - First Pass. Error: {report.get('error', 'Unknown error')}, Details: {error_detail}")

        except Exception as e:
            pdf_logger.error(f"Exception during benchmark processing (First Pass) for {pdf_file_path.name}: {e}", exc_info=True)
        
        # Second pass for the same file to test caching
        pdf_logger.info(f"Benchmarking file: {pdf_file_path.name} (from {run_id}) - Second Pass (Cache Test)")
        try:
            # Output path can be the same, it will overwrite. This is fine for cache testing.
            output_path_second_pass = run_output_dir / f"anonymized_{pdf_file_path.name}_second_pass_test" 
            # Using a slightly different name for the second pass output to avoid confusion if needed, 
            # though overwriting the original is also fine for benchmark purposes.
            # For simplicity and direct cache testing of anonymize_pdf's components, 
            # we'll use the same output_path as the main benchmark would, effectively overwriting.

            success_second_pass, report_second_pass = processor.anonymize_pdf(pdf_file_path, output_path) # Using original output_path

            if success_second_pass:
                pdf_logger.info(f"Successfully processed (benchmark): {pdf_file_path.name} to {output_path} - Second Pass (Cache Test)")
                total_files_processed +=1 # Count unique file after successful second pass
            else:
                error_detail_second_pass = report_second_pass.get('details', 'No details') if isinstance(report_second_pass, dict) else str(report_second_pass)
                pdf_logger.error(f"Failed to process (benchmark): {pdf_file_path.name} - Second Pass (Cache Test). Error: {report_second_pass.get('error', 'Unknown error')}, Details: {error_detail_second_pass}")

        except Exception as e:
            pdf_logger.error(f"Exception during benchmark processing (Second Pass - Cache Test) for {pdf_file_path.name}: {e}", exc_info=True)

    overall_duration = time.perf_counter() - overall_start_time
    pdf_logger.info(f"--- Benchmark Run: {run_id} for directory {pdf_directory_path} COMPLETED ---")
    pdf_logger.info(f"Total files successfully processed in this run: {total_files_processed} / {len(pdf_files)}")
    pdf_logger.info(f"Overall wall-clock time for this directory run ({run_id}): {overall_duration:.4f} seconds.")

async def main_benchmark():
    """
    Main function to run benchmarks on specified directories.
    """
    # If pdf_logger is not outputting to console by default, uncomment and configure:
    # if not any(isinstance(h, logging.StreamHandler) for h in pdf_logger.handlers):
    #     console_handler = logging.StreamHandler(sys.stdout)
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - BENCHMARK - %(message)s')
    #     console_handler.setFormatter(formatter)
    #     pdf_logger.addHandler(console_handler)
    #     pdf_logger.setLevel(logging.INFO) # Ensure INFO level is captured for benchmark logs
    #     pdf_logger.propagate = False # Avoid duplicate logs if root logger also has a handler

    pdf_logger.info("Initializing PDFProcessor for benchmarking...")
    pdf_processor = PDFProcessor() 

    # Using raw strings for Windows paths
    test_set_1_path = r"C:\\Users\\tomas\\Desktop\\0001 DUOMENU ANALITIKA\\038_AnonymPDF\\01_Live_test\\01_Live_test_Data_20250602"
    test_set_2_path = r"C:\\Users\\tomas\\Desktop\\0001 DUOMENU ANALITIKA\\038_AnonymPDF\\01_Live_test\\02_Live_test_Data_20250602"

    pdf_logger.info(f"Starting benchmark for Test Set 1 from: {test_set_1_path}")
    await run_benchmark_on_directory(test_set_1_path, pdf_processor, "TestSet1_Optimized")
    
    pdf_logger.info(f"Starting benchmark for Test Set 2 from: {test_set_2_path}")
    await run_benchmark_on_directory(test_set_2_path, pdf_processor, "TestSet2_Optimized")

    pdf_logger.info("All benchmark runs completed.")

    # Get and print cache statistics
    pdf_logger.info("--- Cache Performance Statistics ---")
    cache_instance = get_intelligent_cache()
    comprehensive_stats = cache_instance.get_comprehensive_stats()

    for cache_name, stats in comprehensive_stats.items():
        pdf_logger.info(f"Stats for {cache_name}:")
        if isinstance(stats, dict):
            # General cache and other LRUCache instances will have hits, misses, hit_rate
            if 'hits' in stats and 'misses' in stats and 'total_requests' in stats:
                hits = stats.get('hits', 0)
                misses = stats.get('misses', 0)
                total_requests = stats.get('total_requests', 0)
                hit_rate = stats.get('hit_rate', 0)
                evictions = stats.get('evictions', 0)
                current_size = stats.get('size',0)
                
                pdf_logger.info(f"  Hits: {hits}")
                pdf_logger.info(f"  Misses: {misses}")
                pdf_logger.info(f"  Total Requests: {total_requests}")
                pdf_logger.info(f"  Hit Rate: {hit_rate:.2%}")
                pdf_logger.info(f"  Evictions: {evictions}")
                pdf_logger.info(f"  Current Size: {current_size}")

            # PatternCache and ModelCache might have different stat structures
            elif 'pattern_details' in stats: # For PatternCache
                pdf_logger.info(f"  Pattern Cache Details:")
                for content_type, pattern_stat in stats['pattern_details'].items():
                    p_hits = pattern_stat.get('hits',0)
                    p_requests = pattern_stat.get('requests',0)
                    p_hit_rate = (p_hits / p_requests * 100) if p_requests > 0 else 0
                    pdf_logger.info(f"    ContentType '{content_type}': Hits={p_hits}, Requests={p_requests}, HitRate={p_hit_rate:.2f}%")
            
            elif 'model_details' in stats: # For ModelCache
                pdf_logger.info(f"  Model Cache Details:")
                # Similar detailed logging for model cache if needed
                # For now, just log the top-level model stats if they exist
                m_hits = stats.get('total_hits', 0) # Assuming these keys exist based on ModelCache implementation
                m_misses = stats.get('total_misses', 0)
                m_requests = m_hits + m_misses
                m_hit_rate = (m_hits / m_requests * 100) if m_requests > 0 else 0
                pdf_logger.info(f"    Overall Model Stats: Hits={m_hits}, Requests={m_requests}, HitRate={m_hit_rate:.2f}%")

            else: # Fallback for other dict-based stats
                for key, value in stats.items():
                    pdf_logger.info(f"  {key}: {value}")
        else:
            pdf_logger.info(f"  {stats}") # If stats is not a dict for some reason
    pdf_logger.info("--- End of Cache Statistics ---")

    pdf_logger.info("Cleaning up temporary files...")
    pdf_processor.cleanup()
    pdf_logger.info("Cleanup finished.")

# --- True Batch Benchmark Functions ---

def simple_anonymization_task_processor(task_data_pdf_path_str, # This is task.data
                                        # These come from **kwargs passed to optimize_batch_processing
                                        pdf_processor_instance_kwarg: PDFProcessor, 
                                        output_dir_base_kwarg: Path, 
                                        current_run_id_kwarg: str):
    """
    Wrapper function for PerformanceOptimizer to process a single PDF anonymization task.
    'task_data_pdf_path_str' is the path to the PDF file (from ProcessingTask.data).
    Other arguments are passed via **kwargs from optimize_batch_processing.
    """
    pdf_file_path = Path(task_data_pdf_path_str)
    
    run_output_dir = output_dir_base_kwarg / current_run_id_kwarg # e.g., processed_benchmarks/TestSet1_TrueBatch_Optimized
    run_output_dir.mkdir(parents=True, exist_ok=True) # Ensure it exists
    # Using a distinct name prefix to avoid clashes if the same output dir is used by other means
    output_path = run_output_dir / f"anonymized_TB_{pdf_file_path.name}" 

    # pdf_logger.debug(f"[Task Wrapper for TrueBatch] Processing {pdf_file_path.name} -> {output_path}")
    success, report = pdf_processor_instance_kwarg.anonymize_pdf(pdf_file_path, output_path)
    
    if success:
        return {"status": "success", "file": pdf_file_path.name, "redactions": report.get("totalRedactions", 0)}
    else:
        return {"status": "failure", "file": pdf_file_path.name, "error": report.get("error", "unknown")}

async def run_true_batch_on_directory(
    pdf_directory_path: str, 
    pdf_processor: PDFProcessor, 
    optimizer: PerformanceOptimizer, 
    run_id: str,
    sequential_mode: bool = False
):
    """
    Processes all PDF files in a given directory as a single batch
    using PerformanceOptimizer.optimize_batch_processing.
    Can run in normal (optimized) mode or sequential mode (max_workers=1).
    """
    pdf_dir = Path(pdf_directory_path)
    if not pdf_dir.is_dir():
        pdf_logger.error(f"Provided path is not a directory: {pdf_directory_path}")
        return 0.0 # Return 0 duration for error cases

    pdf_files = list(pdf_dir.glob("*.pdf"))
    if not pdf_files:
        pdf_logger.warning(f"No PDF files found in directory: {pdf_directory_path}")
        return 0.0

    mode_description = "Sequential (max_workers=1)" if sequential_mode else "Optimized Parallel"
    pdf_logger.info(f"--- Starting True Batch Benchmark Run: {run_id} for directory: {pdf_directory_path} ({mode_description}) ---")
    pdf_logger.info(f"Found {len(pdf_files)} PDF files to process as a batch.")

    tasks = []
    for i, pdf_file in enumerate(pdf_files):
        try:
            tasks.append(
                ProcessingTask(
                    task_id=f"{run_id}_{pdf_file.stem}_{i}", 
                    data=str(pdf_file), # Pass path as string, wrapper will convert
                    estimated_size=pdf_file.stat().st_size # For load balancer if it uses it
                )
            )
        except FileNotFoundError:
            pdf_logger.error(f"File not found while creating task: {pdf_file}. Skipping this file for the batch.")
            continue # Skip if file disappeared

    if not tasks:
        pdf_logger.warning(f"No valid tasks created for batch run: {run_id}. Aborting this batch.")
        return 0.0

    original_max_workers = None
    # Get the ParallelProcessor instance associated with the given PerformanceOptimizer
    # It's assumed PerformanceOptimizer has an attribute 'parallel_processor'
    if not hasattr(optimizer, 'parallel_processor'):
        pdf_logger.error("PerformanceOptimizer does not have 'parallel_processor' attribute. Cannot force sequential mode.")
        return 0.0
        
    parallel_proc_instance = optimizer.parallel_processor

    if sequential_mode:
        original_max_workers = parallel_proc_instance.max_workers
        pdf_logger.info(f"BENCHMARK (True Batch): Forcing sequential mode by setting ParallelProcessor.max_workers to 1 (was {original_max_workers}) for run {run_id}")
        parallel_proc_instance.max_workers = 1
    
    overall_start_time = time.perf_counter()
    
    batch_results = []
    try:
        batch_results = optimizer.optimize_batch_processing(
            tasks=tasks, 
            processor_func=simple_anonymization_task_processor,
            # These kwargs are passed to simple_anonymization_task_processor
            pdf_processor_instance_kwarg=pdf_processor,
            output_dir_base_kwarg=output_base_dir, # Global 'output_base_dir' from script
            current_run_id_kwarg=run_id
        )
    except Exception as e:
        pdf_logger.error(f"Exception during optimize_batch_processing for {run_id}: {e}", exc_info=True)
        # Ensure max_workers is reset even if batch processing fails
        if sequential_mode and original_max_workers is not None:
            parallel_proc_instance.max_workers = original_max_workers
            pdf_logger.info(f"BENCHMARK (True Batch): Restored ParallelProcessor.max_workers to {original_max_workers} for run {run_id} after exception.")
        return 0.0 # Error, return 0 duration

    overall_duration = time.perf_counter() - overall_start_time

    if sequential_mode and original_max_workers is not None:
        parallel_proc_instance.max_workers = original_max_workers
        pdf_logger.info(f"BENCHMARK (True Batch): Restored ParallelProcessor.max_workers to {original_max_workers} for run {run_id}")

    successful_tasks_count = sum(1 for res in batch_results if res.success and res.result and res.result.get("status") == "success")
    failed_tasks_count = len(tasks) - successful_tasks_count

    pdf_logger.info(f"--- True Batch Benchmark Run: {run_id} for directory {pdf_directory_path} COMPLETED ({mode_description}) ---")
    pdf_logger.info(f"Total tasks processed in this batch: {successful_tasks_count} successful, {failed_tasks_count} failed / {len(tasks)}")
    if batch_results: # Log details of failed tasks if any
        for res in batch_results:
            if not res.success or (res.result and res.result.get("status") == "failure"):
                pdf_logger.warning(f"  Task failed: ID={res.task_id}, File={res.result.get('file', 'N/A') if res.result else 'N/A'}, Error={res.result.get('error', res.error) if res.result else res.error}")

    pdf_logger.info(f"Overall wall-clock time for this true batch run ({run_id}): {overall_duration:.4f} seconds.")
    
    return overall_duration

async def main_true_batch_benchmark():
    """
    Main function to run true batch benchmarks using PerformanceOptimizer.
    This will measure the speedup of batch processing.
    """
    pdf_logger.info("Initializing PDFProcessor and PerformanceOptimizer for True Batch benchmarking...")
    pdf_processor = PDFProcessor() 
    optimizer = get_performance_optimizer() # Get the global optimizer

    # Ensure the optimizer has its parallel_processor ready
    if not hasattr(optimizer, 'parallel_processor') or optimizer.parallel_processor is None:
        pdf_logger.error("Global PerformanceOptimizer's parallel_processor is not initialized. True batch benchmark cannot run.")
        # Attempt to initialize it if it's a getter that does lazy init
        # This depends on the implementation of get_performance_optimizer and its components
        # For now, we assume get_performance_optimizer() returns a ready-to-use optimizer.
        # If not, one might need to call a method like optimizer._load_components() if it exists
        # or ensure PDFProcessor() call triggers this.
        # Based on get_performance_optimizer, it should be initialized.

    test_set_1_path_str = r"C:\\\\Users\\\\tomas\\\\Desktop\\\\0001 DUOMENU ANALITIKA\\\\038_AnonymPDF\\\\01_Live_test\\\\01_Live_test_Data_20250602"
    test_set_2_path_str = r"C:\\\\Users\\\\tomas\\\\Desktop\\\\0001 DUOMENU ANALITIKA\\\\038_AnonymPDF\\\\01_Live_test\\\\02_Live_test_Data_20250602"

    # --- Test Set 1 ---
    run_id_ts1_opt = "TestSet1_TrueBatch_Optimized"
    pdf_logger.info(f"Starting True Batch benchmark (Optimized) for Test Set 1 from: {test_set_1_path_str}")
    duration_ts1_opt = await run_true_batch_on_directory(test_set_1_path_str, pdf_processor, optimizer, run_id_ts1_opt, sequential_mode=False)
    
    run_id_ts1_seq = "TestSet1_TrueBatch_Sequential"
    pdf_logger.info(f"Starting True Batch benchmark (Sequential) for Test Set 1 from: {test_set_1_path_str}")
    # Important: Re-initialize pdf_processor and optimizer for a clean state if tests modify them in ways not reset
    # For this test, we only modify optimizer.parallel_processor.max_workers, which is reset.
    # Cache should ideally be cleared between distinct benchmark types (e.g. optimized vs sequential) if we want to measure from-cold performance.
    # However, for comparing optimized vs sequential parallel processing, consistent cache state (primed or empty) is key.
    # The file-by-file benchmark (main_benchmark) will prime the cache. So these true batch runs will benefit from a warm cache for text extraction.
    # This is acceptable as we are comparing the parallel execution strategy, not raw text extraction speed again.
    duration_ts1_seq = await run_true_batch_on_directory(test_set_1_path_str, pdf_processor, optimizer, run_id_ts1_seq, sequential_mode=True)

    if duration_ts1_seq > 0.0001 and duration_ts1_opt > 0.0001: # Avoid division by zero or tiny numbers
        speedup_ts1 = duration_ts1_seq / duration_ts1_opt
        pdf_logger.info(f"RESULTS Test Set 1 True Batch Speedup (Sequential Time / Optimized Time): {speedup_ts1:.2f}x ({duration_ts1_seq:.2f}s / {duration_ts1_opt:.2f}s)")
    else:
        pdf_logger.warning(f"Could not calculate Test Set 1 True Batch speedup. Opt time: {duration_ts1_opt:.4f}s, Seq time: {duration_ts1_seq:.4f}s")

    # --- Test Set 2 ---
    run_id_ts2_opt = "TestSet2_TrueBatch_Optimized"
    pdf_logger.info(f"Starting True Batch benchmark (Optimized) for Test Set 2 from: {test_set_2_path_str}")
    duration_ts2_opt = await run_true_batch_on_directory(test_set_2_path_str, pdf_processor, optimizer, run_id_ts2_opt, sequential_mode=False)
    
    run_id_ts2_seq = "TestSet2_TrueBatch_Sequential"
    pdf_logger.info(f"Starting True Batch benchmark (Sequential) for Test Set 2 from: {test_set_2_path_str}")
    duration_ts2_seq = await run_true_batch_on_directory(test_set_2_path_str, pdf_processor, optimizer, run_id_ts2_seq, sequential_mode=True)
    
    if duration_ts2_seq > 0.0001 and duration_ts2_opt > 0.0001:
        speedup_ts2 = duration_ts2_seq / duration_ts2_opt
        pdf_logger.info(f"RESULTS Test Set 2 True Batch Speedup (Sequential Time / Optimized Time): {speedup_ts2:.2f}x ({duration_ts2_seq:.2f}s / {duration_ts2_opt:.2f}s)")
    else:
        pdf_logger.warning(f"Could not calculate Test Set 2 True Batch speedup. Opt time: {duration_ts2_opt:.4f}s, Seq time: {duration_ts2_seq:.4f}s")

    pdf_logger.info("All True Batch benchmark runs completed.")


async def run_all_benchmarks():
    """Runs all benchmark suites and then prints cache stats and cleans up."""
    # 1. Run File-by-File benchmarks (for cache validation and per-file timings)
    await main_benchmark()

    # 2. Run True Batch benchmarks (for parallel processing speedup)
    await main_true_batch_benchmark()

    # 3. Get and print cache statistics (after all processing)
    pdf_logger.info("--- Cache Performance Statistics (After All Benchmarks) ---")
    cache_instance = get_intelligent_cache()
    comprehensive_stats = cache_instance.get_comprehensive_stats()

    for cache_name, stats in comprehensive_stats.items():
        pdf_logger.info(f"Stats for {cache_name}:")
        if isinstance(stats, dict):
            if 'hits' in stats and 'misses' in stats and 'total_requests' in stats:
                hits = stats.get('hits', 0)
                misses = stats.get('misses', 0)
                total_requests = stats.get('total_requests', 0)
                hit_rate = stats.get('hit_rate', 0)
                evictions = stats.get('evictions', 0)
                current_size = stats.get('size',0)
                
                pdf_logger.info(f"  Hits: {hits}")
                pdf_logger.info(f"  Misses: {misses}")
                pdf_logger.info(f"  Total Requests: {total_requests}")
                pdf_logger.info(f"  Hit Rate: {hit_rate:.2%}")
                pdf_logger.info(f"  Evictions: {evictions}")
                pdf_logger.info(f"  Current Size: {current_size}")
            elif 'pattern_details' in stats: 
                pdf_logger.info(f"  Pattern Cache Details:")
                for content_type, pattern_stat in stats['pattern_details'].items():
                    p_hits = pattern_stat.get('hits',0)
                    p_requests = pattern_stat.get('requests',0)
                    p_hit_rate = (p_hits / p_requests * 100) if p_requests > 0 else 0
                    pdf_logger.info(f"    ContentType '{content_type}': Hits={p_hits}, Requests={p_requests}, HitRate={p_hit_rate:.2f}%")
            elif 'model_details' in stats: 
                pdf_logger.info(f"  Model Cache Details:")
                m_hits = stats.get('total_hits', 0) 
                m_misses = stats.get('total_misses', 0)
                m_requests = m_hits + m_misses
                m_hit_rate = (m_hits / m_requests * 100) if m_requests > 0 else 0
                pdf_logger.info(f"    Overall Model Stats: Hits={m_hits}, Requests={m_requests}, HitRate={m_hit_rate:.2f}%")
            else: 
                for key, value in stats.items():
                    pdf_logger.info(f"  {key}: {value}")
        else:
            pdf_logger.info(f"  {stats}") 
    pdf_logger.info("--- End of Cache Statistics ---")

    # 4. Cleanup (using the last PDFProcessor instance from main_benchmark, or create a new one if needed)
    # This assumes pdf_processor variable from main_benchmark() is not what we need,
    # rather the one initialized at the start of main_true_batch_benchmark or main_benchmark.
    # It is safer to re-initialize one for cleanup if there's doubt about state.
    # For simplicity, let's assume the PDFProcessor has no state that impacts cleanup
    # or that the one from true_batch is fine.
    # The cleanup method in PDFProcessor is simple, just removes files from its temp_dir.
    pdf_logger.info("Cleaning up temporary files (using a new PDFProcessor instance for cleanup)...")
    final_cleanup_processor = PDFProcessor()
    final_cleanup_processor.cleanup() # Perform cleanup
    pdf_logger.info("Cleanup finished.")

if __name__ == "__main__":
    # Configure logging for benchmark_runner if it's not already handled by StructuredLogger's default console output
    # This is usually not needed if StructuredLogger is working as expected.
    # Example:
    # if not any(isinstance(h, logging.StreamHandler) for h in pdf_logger.handlers):
    #     ch = logging.StreamHandler(sys.stdout) # Changed to stdout
    #     ch.setLevel(logging.INFO)
    #     formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    #     ch.setFormatter(formatter)
    #     pdf_logger.addHandler(ch)
    #     pdf_logger.logger.propagate = False # Access underlying standard logger

    asyncio.run(run_all_benchmarks()) 