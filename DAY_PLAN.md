# 8-Hour "GO GREEN" Action Plan

## Mission: Full Test Suite Pass by EOD. Zero Regressions.

---

### **Impact & Priority Matrix**

| Root Cause Cluster                        | Failure Count | Severity | Fix Complexity | Priority | Estimated Fix Time |
| ----------------------------------------- | :-----------: | :------: | :------------: | :------: | :----------------: |
| 1. Flawed Lithuanian PII Detection        |       4       | Critical |      Low       |  **P0**  |      1.5 hrs       |
| 2. Inconsistent Redaction/Detection Logic |       2       |   High   |   Medium     |  **P1**  |      2.0 hrs       |
| 3. Broken Advanced Pattern Matching       |       2+      |   High   |   Medium     |  **P1**  |      1.5 hrs       |
| 4. System & Environment Instability       |       3+      | Blocker  |      High      |  **P2**  |      2.0 hrs       |

---

## **Sprint Breakdown (Hour-by-Hour)**

### **Sprint 1: Stabilize Core PII (Hours 0-2)**

-   [ ] **Task 1.1 (Repro & Isolate):**
    -   [ ] Lock `tests/test_lithuanian_pii.py` to run exclusively.
    -   Command: `pytest tests/test_lithuanian_pii.py --cache-clear -v`
    -   Confirm failure `TestLithuanianIntegration.test_anti_overredaction_of_common_words` locally.
-   [ ] **Task 1.2 (Root-Cause Fix):**
    -   [ ] **Hypothesis:** The `LITHUANIAN_PHONE_NUMBER` regex in `config/patterns.yaml` is greedy and incorrect.
    -   [ ] **Action:** Modify the regex to correctly capture the full international format `+370 XXX XXXXX`. The current pattern `\+370\s\d{3}\s\d{5}` seems plausible but might have issues with how it's defined in YAML or interpreted by the engine. Let's start by inspecting `patterns.yaml`.
    -   [ ] **Action:** Correct the pattern, ensuring it's a single, non-breaking expression.
-   [ ] **Task 1.3 (Validate Fix):**
    -   [ ] Rerun `pytest tests/test_lithuanian_pii.py`.
    -   [ ] **Goal:** Pass all tests in `test_lithuanian_pii.py`.
-   [ ] **Task 1.4 (Commit):**
    -   [ ] `git commit -m "fix(PII): Correct Lithuanian phone number regex for full match"`

**Next Step:** Proceed to Sprint 2 to fix the inconsistent redaction logic.

---

### **Sprint 2: Fix Data Contracts & Redaction Application (Hours 2-4)**

-   [ ] **Task 2.1 (Repro & Isolate - Key Mismatch):**
    -   [ ] Run `pytest tests/test_pdf_processor.py::TestPDFProcessorIntegration::test_process_pdf_success -v`
    -   [ ] Confirm `AssertionError: assert None == 1`
-   [ ] **Task 2.2 (Root-Cause Fix - Key Mismatch):**
    -   [ ] **Hypothesis:** `AdvancedPatternRefinement` returns `email` (lowercase) as the category, but the test expects `emails`.
    -   [ ] **Action:** Locate the `email` pattern in `config/patterns.yaml` and change its name/key to `emails` for consistency, OR change the test assertion. The [memory I have about this class][[memory:2066266056353795392]] states the pattern name should be used as the category, so changing `patterns.yaml` is the correct fix.
-   [ ] **Task 2.3 (Repro & Isolate - No Annotations):**
    -   [ ] Run `pytest tests/test_pdf_processor.py::TestPDFProcessorIntegration::test_anonymize_pdf_flow -v`
-   [ ] **Task 2.4 (Root-Cause Fix - No Annotations):**
    -   [ ] **Hypothesis:** The redaction data from `pii_detection` is not being passed correctly to the `fitz` annotation step in `pdf_processor.py`.
    -   [ ] **Action:** Debug `pdf_processor.py`. Trace the `detections` from the PII engine to the loop that creates `page.add_redact_annot()`. Ensure the coordinates are valid and the call is being made for every reported redaction.
-   [ ] **Task 2.5 (Commit):**
    -   [ ] `git commit -m "fix(Processor): Align email pattern key to 'emails'"`
    -   [ ] `git commit -m "fix(Processor): Ensure redaction data creates PDF annotations"`

**Next Step:** Harden the advanced pattern engine in Sprint 3.

---

### **Sprint 3: Harden Advanced Engine & System (Hours 4-6)**

-   [ ] **Task 3.1 (Repro & Isolate):**
    -   [ ] Run `pytest tests/test_priority2_enhancements.py -v`.
-   [ ] **Task 3.2 (Root-Cause Fix):**
    -   [ ] **Hypothesis:** A recent change to `context_analyzer.py` or the structure of `patterns.yaml` broke the `find_enhanced_patterns` method.
    -   [ ] **Action:** Review the logic in `app/core/context_analyzer.py`. Based on memory, this class iterates a flat dictionary of patterns. Verify the patterns are being loaded correctly and the regex match is using the capturing group as intended.
    -   [ ] **Action:** Fix the logic to ensure it correctly iterates patterns and extracts matches.
-   [ ] **Task 3.3 (System Stability):**
    -   [ ] **Hypothesis:** The remaining system test failures are due to Windows-specific encoding issues.
    -   [ ] **Action:** Examine `tests/system/test_real_time_monitor_integration.py`. Add `encoding="utf-8"` to all `open()` calls within the test and any underlying file I/O in the monitor itself if it's not already present.
-   [ ] **Task 3.4 (Commit):**
    -   [ ] `git commit -m "fix(Engine): Repair logic in AdvancedPatternRefinement"`
    -   [ ] `git commit -m "fix(System): Enforce UTF-8 encoding in file I/O for Windows compat"`

**Next Step:** Full validation and confidence rebuild in Sprint 4.

---

### **Sprint 4: Confidence Rebuild & Final Validation (Hours 6-8)**

-   [ ] **Task 4.1 (Full Suite Execution):**
    -   [ ] Run the entire test suite: `pytest --cache-clear`
    -   [ ] **Goal:** 100% pass rate.
-   [ ] **Task 4.2 (Address Lingering Flakiness):**
    -   [ ] If any tests fail, analyze immediately.
    -   [ ] If it's a race condition or async timing issue, add explicit waits. **No `xfail` or `retry` without justification.**
-   [ ] **Task 4.3 (Code Cleanup & Final Push):**
    -   [ ] Remove any debugging statements (`print`, etc.).
    -   [ ] `git push origin main`
-   [ ] **Task 4.4 (Mission Complete):**
    -   [ ] Trigger CI run.
    -   [ ] **Deliverable:** Monitor the run and confirm an all-green pipeline.
    -   [ ] Delete `ROOT_CAUSE_REPORT.md` and `DAY_PLAN.md` as they are ephemeral work products.

**Next Step:** Begin Sprint 1. 