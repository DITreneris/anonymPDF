# Scorched Earth Plan: Road to Green

## 1. Unforgiving Triage & Failure Cluster Analysis

The 9 persistent failures are symptoms of four deep-seated architectural diseases.

| Cluster | Severity | Disease | Evidence (Failing Tests) |
| :--- | :--- | :--- | :--- |
| **#1: Core Engine Failure** | **BLOCKER** | The `AdvancedPatternRefinement` engine is fundamentally broken, returning zero results for enhanced patterns. | `test_priority2_enhancements.py::test_enhanced_personal_code_detection` |
| **#2: Data Integrity & PDF Corruption** | **CRITICAL** | The `PDFProcessor` has a broken data contract (`email`/`emails`) and its redaction logic is corrupting PDF objects. | `test_pdf_processor.py::test_process_pdf_success`, `test_pdf_processor.py::test_anonymize_pdf_flow` |
| **#3: Catastrophic Language Logic** | **CRITICAL** | The Lithuanian-specific logic is pathologically greedy on names/addresses and completely blind to grammatical cases for locations. | All 3 failures in `test_lithuanian_pii.py` |
| **#4: Test State Pollution** | **HIGH** | Database fixtures are not being torn down, causing state to bleed between tests and create false negatives. | `test_pattern_learner.py` (2 failures), `test_real_time_monitor_integration.py` (1 failure) |

---

## 2. Surgical Strikes: Step-by-Step Eradication

### **Sprint 1: Rebuild the Core (Cluster #1 & #2)**

*   **Target 1: `AdvancedPatternRefinement` (`context_analyzer.py`)**
    *   **Root Cause:** The `find_enhanced_patterns` method does not use the patterns from the `LithuanianLanguageEnhancer`. It only uses the base patterns from `ConfigManager`, which is why the Lithuanian-specific "personal code with label" pattern is never found.
    *   **Solution:**
        1.  I will inject the `LithuanianLanguageEnhancer` into `AdvancedPatternRefinement`'s constructor.
        2.  I will modify `find_enhanced_patterns` to iterate over both the base patterns *and* the `enhanced_lithuanian_patterns`, merging the results.

*   **Target 2: `PDFProcessor` (`pdf_processor.py`) & Tests**
    *   **Root Cause (`emails` key):** The test is using a mocked `ConfigManager` that has not been updated with the `emails` key change. The fix is not in the core code, but in the test itself.
    *   **Root Cause (PDF Corruption):** My previous word-by-word redaction logic (`inst | next_word_rect[0]`) is an incorrect use of the PyMuPDF API, causing the `Point` error.
    *   **Solution:**
        1.  I will inspect `tests/test_pdf_processor.py` and fix the mock object to use the correct `emails` key.
        2.  I will gut my previous flawed redaction logic in `anonymize_pdf` and replace it with the correct PyMuPDF method for creating a bounding box that encompasses all words in a multi-word PII string.

### **Sprint 2: Eradicate Flawed Language Logic (Cluster #3)**

*   **Target 1: Greedy Patterns (`lithuanian_enhancements.py`)**
    *   **Root Cause:** My previous regex fixes were weak. The patterns are still too greedy.
    *   **Solution:** I will rewrite the `lithuanian_name_with_title` and `lithuanian_address_full` patterns from scratch to be highly constrained, non-greedy, and to explicitly stop at sentence-ending punctuation or conjunctions.

*   **Target 2: Missing Locations (`pdf_processor.py`)**
    *   **Root Cause:** The spaCy NER model is identifying "Vilniaus" with low confidence, and the `deduplicate_with_confidence` logic discards it.
    *   **Solution:** I will modify the logic to ensure that if a Lithuanian city is detected by our robust regex, its confidence score is artificially boosted to `0.9` (VERY_HIGH), guaranteeing it wins against any conflicting, weaker spaCy detection.

### **Sprint 3: Sterilize the Environment (Cluster #4)**

*   **Target: Test Fixtures (`conftest.py`, `tests/adaptive/test_pattern_learner.py`)**
    *   **Root Cause:** The fixtures that create test databases are not cleaning up after themselves.
    *   **Solution:** I will locate every fixture responsible for creating a `.db` file. I will rewrite them using a `yield` statement inside a `try...finally` block to guarantee that the database file is deleted (`os.remove(db_path)`) after the test has completed, regardless of whether it passed or failed.

---

**Execution starts now.** I will begin with Sprint 1. No more communication until I have a fix for Cluster #1. 