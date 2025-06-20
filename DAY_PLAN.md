# Killer One-Day Coding Plan - Progress Update

**Overall Status: Sprint 4 in progress. Core blockers and major regressions resolved. Focusing on logic and final validation.**

---

### Sprint 1 – Rapid Repro & Blocker Fix (Hours 0-2) - COMPLETE
**Goal:** Resolve the `FixtureNotFound` errors to unblock the entire test suite.
- **[X] Task 1.1 - 1.4 (COMPLETE):** Successfully identified missing `client` and `adaptive_coordinator` fixtures.
  - **Decision:** Added the required fixtures to `tests/conftest.py`.
  - **New Blocker:** Uncovered a `Redis` connection error during API tests.
  - **Decision:** Mocked the Celery `.delay()` call in `tests/api/test_pdf_endpoint.py` using `monkeypatch` to isolate the test from the Redis dependency.
- **Outcome:** All setup errors resolved. API and System tests are now running.

---

### Sprint 2 – Root-Cause Fix (Hours 2-4) - COMPLETE
**Goal:** Eliminate the `AttributeError` and the Pydantic `AttributeError`, which were clear regressions.
- **[X] Task 2.1 (Fix `FileProcessingMetrics`) - COMPLETE:**
  - **Decision:** Identified that `start_file_processing` was renamed to `track_file_processing` in `app/core/performance.py`. The implementation was also changed to a context manager.
  - **Fix:** Refactored the call in `app/services/pdf_processor.py` to use a `try...finally` block to ensure the new performance tracking method is called correctly.
- **[X] Task 2.2 (Fix Pydantic/Dataclass `dict` method) - COMPLETE:**
  - **Initial Assumption:** Believed the `AttributeError: 'AdaptivePattern' object has no attribute 'dict'` was due to a Pydantic v2 upgrade.
  - **Investigation:** The fix `.model_dump()` also failed, revealing that `AdaptivePattern` was a standard dataclass, not a Pydantic model.
  - **Decision:** Corrected the code in `tests/adaptive/test_pattern_db.py` to use `asdict` from the `dataclasses` module.
- **Outcome:** All `AttributeError` regressions are fixed.

---

### Sprint 3 – Suite Hardening (Hours 4-6) - COMPLETE
**Goal:** Fix state leakage in database tests and address the `KeyError` on detection results.
- **[X] Task 3.1 (Isolate DB Tests) - COMPLETE:**
    - **Initial Fix:** Changed the `db_session` fixture scope in `tests/conftest.py` from `session` to `function` and added a `rollback()` to improve test isolation. This was not sufficient.
    - **Root Cause:** Discovered a logic bug in `app/core/adaptive/pattern_db.py` where the `is_active` flag was not being correctly passed to the `INSERT` statement, causing state to leak.
    - **Decision:** Patched the SQL statements in `add_or_update_pattern` to correctly handle the `is_active` flag.
- **Outcome:** All database state leakage issues are resolved. `tests/adaptive/test_pattern_db.py` is now passing.

---

### Sprint 4 – Confidence Rebuild (Hours 6-8) - IN PROGRESS
**Goal:** Tackle the core logic failures and run the full suite to validate green status.
- **[X] Task 4.1 (Fix `validate_person_name`) - COMPLETE:**
  - **Decision:** The function in `app/core/validation_utils.py` was missing several key checks.
  - **Fix:** Added validation to reject names that are too short (`<2` chars), too long (`>100` chars), contain digits, or start with excluded prefixes (`Nr.`, `Tel.`).
  - **Outcome:** All tests in `tests/test_validation_utils.py` are now passing.

- **[ ] Task 4.2 (Fix Lithuanian PII detection) - IN PROGRESS:**
  - **Investigation:** The tests in `tests/test_lithuanian_pii.py` were failing because the core detection function was missing.
  - **Decision:** Implemented the `find_enhanced_lithuanian_patterns` function in `app/core/lithuanian_enhancements.py` to iterate through the compiled regexes and find all matches.
  - **Current Status:** The fix has been implemented, but the corresponding test suite (`tests/test_lithuanian_pii.py`) has **not been validated yet**. This is the next immediate step.

- **[ ] Task 4.3 (Full Suite Run):**
  - Execute `pytest --cov=app --cov-report=term-missing --cov-fail-under=80`.
  - Analyze the output. All tests should pass.
  - Note the final coverage report.

- **[ ] Task 4.4 (Final Validation):**
  - If all tests are green, the mission is complete. Prepare commit messages for the patches.