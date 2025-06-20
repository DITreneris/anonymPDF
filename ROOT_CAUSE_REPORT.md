# Root Cause Analysis: Pytest Suite Failures

**Execution Date:** 2025-06-20  
**Log File:** `56_log.txt`

## 1. Executive Summary

The test suite is experiencing a critical failure cascade, with **22 failed tests, 4 setup errors, and 5 warnings**. The overall test coverage is **64.47%**, failing the required 80% quality gate.

The failures are clustered into five primary root causes:
1.  **Fixture Configuration Errors:** Blocking entire test classes from running.
2.  **Core Logic Regressions (`AttributeError`):** A breaking change in a core component affects multiple integration and system tests.
3.  **Incorrect PII Detection Logic:** Numerous assertion failures in PII detection, especially for Lithuanian language contexts and name validation.
4.  **Data Model & API Mismatches:** Inconsistencies between data structures and component APIs, likely due to dependency updates (e.g., Pydantic v1 to v2).
5.  **Stateful Database Logic Failures:** Tests involving database state are not isolated, leading to incorrect assertions.

This report details each cluster with specific evidence. The action plan will prioritize fixes based on impact and severity, starting with the blockers.

---

## 2. Failure Clusters (Deep Diagnostic)

### Cluster 1: Fixture Not Found (Blocker)

*   **Root Cause:** Fundamental test setup issue where tests are invoking fixtures (`client`, `adaptive_coordinator`) that are not available in their scope. This typically happens when a `conftest.py` file is missing, not correctly located, or the fixtures are defined in a scope not accessible to these specific tests.
*   **Severity:** Critical Blocker.
*   **Evidence:**
    *   `tests/api/test_pdf_endpoint.py:12` - `ERROR at setup ... test_post_lithuanian_pdf_for_processing_success`: `fixture 'client' not found`
    *   `tests/api/test_pdf_endpoint.py:39` - `ERROR at setup ... test_post_non_pdf_file_returns_4xx`: `fixture 'client' not found`
    *   `tests/system/test_adaptive_workflow.py:111` - `ERROR at setup ... test_feedback_api_is_disabled_for_now`: `fixture 'client' not found`
    *   `tests/system/test_adaptive_workflow.py:119` - `ERROR at setup ... test_adaptive_workflow_learns_new_pattern`: `fixture 'adaptive_coordinator' not found`
*   **Next Step:** Investigate the project's `conftest.py` structure. The `client` fixture for FastAPI testing and the `adaptive_coordinator` fixture need to be made available to the `tests/api` and `tests/system` test paths.

### Cluster 2: `AttributeError` in Performance Metrics

*   **Root Cause:** A method, likely `start_file_processing`, was renamed or removed from the `FileProcessingMetrics` class, but its usage in `app/services/pdf_processor.py` was not updated. This is a classic breaking API change.
*   **Severity:** High. This single error causes multiple integration and system tests to fail.
*   **Evidence:**
    *   `AttributeError: 'FileProcessingMetrics' object has no attribute 'start_file_processing'`
    *   `FAILED tests/test_pdf_processor.py::TestPDFProcessorIntegration::test_process_pdf_success`
    *   `FAILED tests/test_pdf_processor.py::TestPDFProcessorIntegration::test_process_pdf_failure_on_invalid_content`
    *   `FAILED tests/system/test_real_time_monitor_integration.py::test_monitoring_end_to_end`
*   **Next Step:** Inspect the `FileProcessingMetrics` class to find the new method name and refactor the call at `app/services/pdf_processor.py:603`.

### Cluster 3: Incorrect PII Detection & Validation Logic

*   **Root Cause:** The PII detection and validation logic has either regressed or does not handle specific edge cases correctly. This includes issues with salutation handling ("Ponas"), contextual location detection, name validation rules, and an API change in the redaction report format.
*   **Severity:** High. These are core functionality failures.
*   **Evidence:**
    *   `FAILED tests/test_lithuanian_pii.py`: Multiple assertion errors, e.g., failing to find a phone number (`+370...`), incorrect name extraction (`'Linas Vaitkus' in {'Ponas Linas Vaitkus...'}`), and missed location detection.
    *   `FAILED tests/test_lithuanian_pii.py`: `AssertionError: assert 'summary' in {'categories': ...}` indicates the redaction report structure changed.
    *   `FAILED tests/test_validation_utils.py`: `validate_person_name` incorrectly approves invalid names (e.g., "A", "Nr. 123", "John123").
*   **Next Step:** Address the logic failures one by one. Refactor `validate_person_name` to be stricter. Adjust the Lithuanian PII detection to handle salutations and context correctly. Update tests to match the new redaction report structure.

### Cluster 4: Data Model and API Mismatches

*   **Root Cause:** Interfaces between components have diverged. The structure of detection result objects has changed (`KeyError: 'pattern_name'`), and Pydantic model methods have been updated (`AttributeError: 'AdaptivePattern' object has no attribute 'dict'`), likely from a library upgrade.
*   **Severity:** Medium. Affects specific advanced features.
*   **Evidence:**
    *   `FAILED tests/test_priority2_enhancements.py`: `KeyError: 'pattern_name'` on detection results.
    *   `FAILED tests/adaptive/test_pattern_db.py`: `AttributeError: 'AdaptivePattern' object has no attribute 'dict'`.
    *   `FAILED tests/test_pdf_processor.py`: `AssertionError: Expected 'extract_text_from_pdf' to be called once. Called 0 times.`
*   **Next Step:** Refactor `tests/test_priority2_enhancements.py` to use the new data structure for detections. Replace `.dict()` with `.model_dump()` for Pydantic models in `tests/adaptive/test_pattern_db.py`. Investigate the control flow in `anonymize_pdf_flow` to fix the mock call.

### Cluster 5: Stateful Database Logic Failures

*   **Root Cause:** Tests interacting with the adaptive pattern database are not properly isolated. State from one test is leaking into the next, causing assertion failures on counts of active/inactive patterns and breaking the feedback learning loop validation.
*   **Severity:** Medium. Undermines confidence in the adaptive learning system.
*   **Evidence:**
    *   `FAILED tests/adaptive/test_pattern_db.py`: `assert 2 == 1` and `assert 1 == 0` on pattern counts.
    *   `FAILED tests/system/test_adaptive_workflow.py`: `AssertionError: Adaptive pattern was not found...` across three parameterized runs.
*   **Next Step:** Ensure that database fixtures have a `yield` statement and perform cleanup (e.g., `db.rollback()`, `db.close()`, or table truncation) after each test execution to guarantee test isolation.

---
## 3. Warnings Analysis

*   **`InconsistentVersionWarning` (sklearn) & XGBoost serialization warning:** Models were created with older library versions (`sklearn==1.4.0`, an old `xgboost`). Loading them in the current environment (`sklearn==1.7.0`) is risky and can lead to silent errors.
*   **Action:** A model retraining and re-serialization sprint using consistent, current library versions must be planned post-haste. This is not part of the immediate 8-hour fix but is a critical reliability action item. 