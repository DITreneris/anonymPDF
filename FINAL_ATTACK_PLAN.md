# Final Attack Plan: Road to Green

## 1. Triage & Failure Cluster Analysis

The 10 failures are not independent. They fall into four distinct clusters, each pointing to a core architectural flaw.

| Cluster | Severity | Components Involved | Failure Summary |
| :--- | :--- | :--- | :--- |
| **#1: Advanced Patterns** | **BLOCKER** | `context_analyzer.py`, `patterns.yaml` | The core regex engine is returning zero results. This is the foundation of all other detection. |
| **#2: PDF Processor** | **Critical** | `pdf_processor.py` | The processor fails to use detected PII to create physical redactions and has broken data contracts. |
| **#3: Lithuanian Logic** | **Critical** | `lithuanian_enhancements.py`, `pdf_processor.py` | Greedy name matching and a complete failure to detect locations, even after previous fixes. |
| **#4: System State** | **High** | `test_pattern_learner.py`, `test_real_time_monitor.py`, `conftest.py` | Tests that rely on database state or inter-service communication are failing. |

---

## 2. Root Cause & Step-by-Step Solution

### **Sprint 1: Fix The Foundation (Cluster #1 & #2)**

#### **Task 1.1: Fix Advanced Pattern Engine**
*   **Root Cause:** The `AdvancedPatternRefinement.find_enhanced_patterns` method in `context_analyzer.py` is failing silently. The `pattern_map` is likely not being populated correctly, or the logic is flawed, causing it to return an empty list for all inputs.
*   **Step 1: Read the Test.** Examine `tests/test_priority2_enhancements.py` to understand the expected inputs.
*   **Step 2: Debug `AdvancedPatternRefinement`**. Add logging to the `__init__` and `find_enhanced_patterns` methods to verify the `pattern_map` is loaded and that the main loop is executing.
*   **Step 3: Implement Fix.** Correct the logic to ensure patterns are found and returned.

#### **Task 1.2: Fix PDF Processor Data Flow**
*   **Root Cause:**
    1.  A data contract is broken: The system expects the key `'emails'` but the pattern in `patterns.yaml` is named `'email'`.
    2.  The `anonymize_pdf` function in `pdf_processor.py` is not correctly using the `personal_info` dictionary to find and redact text in the PDF.
*   **Step 1: Fix Key Mismatch.** In `config/patterns.yaml`, rename the `email` pattern to `emails`.
*   **Step 2: Trace Data Flow.** Debug `anonymize_pdf`, tracing how the `all_pii_texts` set is populated and used in the `page.search_for(pii_text)` loop. The issue is likely that the text from PII detection doesn't exactly match the text in the PDF, causing the search to fail. We may need to make the search more robust.

### **Sprint 2: Fix Language-Specific Logic (Cluster #3)**

#### **Task 2.1: Fix Greedy Lithuanian Names**
*   **Root Cause:** The `lithuanian_name_with_title` regex in `lithuanian_enhancements.py` is still too greedy. My previous negative lookahead fix was insufficient.
*   **Step 1: Rewrite Name Regex.** Rewrite the pattern with a more restrictive structure, for example, by explicitly not matching across conjunctions (`ir`, `bei`, etc.).

#### **Task 2.2: Fix Missing Lithuanian Locations**
*   **Root Cause:** The `find_personal_info` method is failing to use our regex-based city detector for Lithuanian text, likely because my previous manual injection was flawed. It is falling back to the default spaCy NER which cannot handle grammatical cases.
*   **Step 1: Correctly Integrate City Detection.** Re-implement the logic in `find_personal_info` to ensure that for Lithuanian text, our `lithuanian_city_simple` pattern is *always* run and its results are correctly added to the `context_aware_detections` list.

### **Sprint 3: Harden The Suite (Cluster #4 & Final Validation)**

#### **Task 3.1: Stabilize State-Dependent Tests**
*   **Root Cause:** The `pattern_learner` and `real_time_monitor` tests depend on a clean database state for each run. Fixtures defined in `conftest.py` are likely failing to tear down and reset the database correctly between test runs.
*   **Step 1: Analyze Fixtures.** Examine all database-related fixtures in `conftest.py` and the failing test files.
*   **Step 2: Implement Clean-up.** Ensure that every test that writes to a database has a `yield` statement followed by code that deletes the test database file or clears the relevant tables.

#### **Task 3.2: Full Suite Validation**
*   **Step 1: Run Full Suite.** Execute `pytest --cache-clear`.
*   **Step 2: GO GREEN.** Address any final, minor failures. Commit all changes. Delete the plan files.

---

**Execution starts now.** I will begin with Sprint 1, Task 1.1. No more failures. 