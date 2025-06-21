# GO GREEN: One-Day Test Suite Recovery Plan

This plan is engineered for maximum impact and velocity. We will work in focused, 2-hour sprints to systematically eliminate failure clusters.

## Impact & Priority Matrix

| Root Cause Cluster                                 | Failure Count | Blocker Severity | Fix Complexity | Priority | Est. Time (Hours) |
| -------------------------------------------------- | ------------- | ---------------- | -------------- | -------- | ----------------- |
| 1. `DetectionContext` Model Mismatch (`AttributeError`) | 12            | **CRITICAL**     | Low            | **P0**   | 2.0               |
| 2. `AdvancedPatternRefinement` Output (`KeyError`) | 2             | High             | Medium         | **P1**   | 1.5               |
| 3. Logic/Assertion Bugs (`KeyError`, `AssertionError`) | 2             | Medium           | Medium         | **P2**   | 2.5               |

---

## The 8-Hour War Room

### **Sprint 1 (Hours 0-2): Stabilize the Core - Fix `DetectionContext`**
*The goal is to fix the P0 `AttributeError` and turn the majority of tests from FAILED to PASSED.*

-   [ ] **Task 1.1:** Isolate a single failing test from Cluster 1.
    -   **Command:** `pytest tests/test_lithuanian_pii.py::TestLithuanianIntegration::test_comprehensive_lithuanian_document_processing -v`
-   [ ] **Task 1.2:** Analyze the `DetectionContext` object definition in `app/core/context_analyzer.py` to confirm the correct attribute names.
-   [ ] **Task 1.3:** **Refactor** `app/services/pdf_processor.py` at line `359` in `deduplicate_with_confidence` to use the correct attributes from `DetectionContext`.
-   [ ] **Task 1.4:** **Refactor** the `deduplicate_with_confidence` call in `tests/test_pdf_processor.py` at line `130` to pass the correct arguments.
-   [ ] **Task 1.5:** Run the isolated test from Task 1.1. **Validate** it passes.
-   [ ] **Task 1.6:** Run the entire `tests/test_pdf_processor.py` and `tests/test_lithuanian_pii.py` modules. **Validate** all `AttributeError` failures are resolved.
-   [ ] **Task 1.7:** Commit the fix with a clear message: `fix(core): Correct DetectionContext attribute usage in PDFProcessor`.

### **Sprint 2 (Hours 2-4): Fix High-Priority Data Contracts**
*The goal is to fix the P1 `KeyError` from the `AdvancedPatternRefinement` and harden the logic bugs from P2.*

-   [ ] **Task 2.1:** Isolate the failing `KeyError` test.
    -   **Command:** `pytest tests/test_priority2_enhancements.py::TestAdvancedPatternRefinement::test_enhanced_email_detection -v`
-   [ ] **Task 2.2:** Investigate `app/core/context_analyzer.py` to find the new data structure returned by `find_enhanced_patterns`.
-   [ ] **Task 2.3:** **Refactor** the assertions in `tests/test_priority2_enhancements.py` at lines `134` and `154` to work with the new data structure.
-   [ ] **Task 2.4:** Address the `KeyError: 'NAMES'` from Cluster 3 in `tests/test_pdf_processor.py:156`. **Refactor** the assertion to use the correct key for the redaction report category.
-   [ ] **Task 2.5:** Run the full `tests/test_priority2_enhancements.py` module. **Validate** all tests pass.
-   [ ] **Task 2.6:** Commit the fixes: `fix(tests): Update tests for new AdvancedPatternRefinement and report schemas`.

### **Sprint 3 (Hours 4-6): Mop-Up & Hardening**
*The goal is to eliminate the final logic bugs and harden the suite against flakiness.*

-   [ ] **Task 3.1:** Tackle the final `AssertionError` in `tests/test_pdf_processor.py:224`.
-   [ ] **Task 3.2:** Debug the `anonymize_pdf_flow` to understand why the mock for `extract_text_from_pdf` is not being called. **Refactor** the test setup or the application logic to ensure the call is made under test conditions.
-   [ ] **Task 3.3:** **Audit** the fixtures in `tests/conftest.py`. Look for any shared state that could "bleed" between tests. Ensure all fixtures that create data (e.g., database entries) are function-scoped unless explicitly designed to be session-wide.
-   [ ] **Task 3.4:** Run the entire test suite locally again. **Validate** all 275 tests now pass.
-   [ ] **Task 3.5:** Commit the final fixes: `fix(tests): Correct logic in anonymize_pdf_flow test and harden fixtures`.

### **Sprint 4 (Hours 6-8): Confidence Rebuild & CI Validation**
*The goal is to prove the fix is complete, produces no regressions, and get a green build on CI.*

-   [ ] **Task 4.1:** Run the full test suite in parallel to check for race conditions.
    -   **Command:** `pytest -n auto`
-   [ ] **Task 4.2:** Generate a new coverage report.
    -   **Command:** `pytest --cov=app --cov-report=term-missing`
-   [ ] **Task 4.3:** Compare the new coverage report to the one in the log (`58_log.txt`). **Validate** that coverage has not decreased.
-   [ ] **Task 4.4:** Push the branch to the remote repository and open a pull request.
-   [ ] **Task 4.5:** **Monitor** the CI pipeline.
-   [ ] **Task 4.6:** On success, capture the "GO GREEN" screenshot and attach it to the pull request. Merge the PR.

---

## **Next Step**

I will begin **Sprint 1, Task 1.1** immediately. I will isolate the first failure to begin the rapid reproduction and fixing cycle. 