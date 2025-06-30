# ONE-DAY FIX PLAN - 8 HOUR SPRINT SCHEDULE
**Target:** Green test suite by EOD  
**Start Time:** 09:30 UTC  
**End Time:** 17:30 UTC  
**Deadline:** CI pipeline green with >80% coverage

---

## SPRINT OVERVIEW
- **Sprint 1 (09:30-11:30):** Critical bug fix - Context analyzer  
- **Sprint 2 (11:45-13:45):** Lithuanian patterns alignment  
- **Sprint 3 (14:30-16:30):** PDF processing & pattern learning  
- **Sprint 4 (16:30-17:30):** Full validation & CI verification  

---

## SPRINT 1: CRITICAL BUG FIX (09:30-11:30)
**Goal:** Fix UnboundLocalError → 8/15 failures resolved (53% improvement)

### Hour 1: Reproduce & Isolate (09:30-10:30)
- [ ] **09:30-09:35** Reproduce specific failing test
  ```bash
  pytest tests/test_priority2_enhancements.py::TestContextualValidator::test_confidence_calculation_person_name -v
  ```
- [ ] **09:35-09:45** Examine `app/core/context_analyzer.py:362` in detail
- [ ] **09:45-10:00** Trace method flow and identify intended logic
- [ ] **10:00-10:15** Document exact fix needed in code comments
- [ ] **10:15-10:30** Create unit test for the fix

### Hour 2: Fix Implementation (10:30-11:30)
- [ ] **10:30-10:45** Fix the UnboundLocalError in `calculate_confidence()` method
- [ ] **10:45-11:00** Run failing tests to verify fix
  ```bash
  pytest tests/test_priority2_enhancements.py::TestContextualValidator -v
  ```
- [ ] **11:00-11:15** Run full priority2 enhancement test suite
- [ ] **11:15-11:30** **COMMIT:** "Fix UnboundLocalError in context_analyzer.py calculate_confidence method"

**Sprint 1 Success Criteria:**
- ✅ 8 UnboundLocalError tests pass
- ✅ No new test regressions introduced
- ✅ Code commit with clear message

---

## SPRINT 2: LITHUANIAN PATTERNS (11:45-13:45)
**Goal:** Align pattern expectations → 11/15 failures resolved (73% improvement)

### Hour 1: Pattern Analysis (11:45-12:45)
- [ ] **11:45-12:00** Reproduce Lithuanian pattern failures
  ```bash
  pytest tests/test_lithuanian_pii.py::TestLithuanianPiiPatterns::test_lithuanian_car_plate_pattern -v
  pytest tests/test_priority2_enhancements.py::TestLithuanianLanguageEnhancer::test_enhanced_lithuanian_patterns -v
  ```
- [ ] **12:00-12:15** Examine `app/core/lithuanian_enhancements.py`
- [ ] **12:15-12:30** Check pattern configuration in `config/patterns.yaml`
- [ ] **12:30-12:45** Identify missing/incorrect patterns

### Hour 2: Pattern Fixes (12:45-13:45)
- [ ] **12:45-13:00** Fix car plate pattern logic
- [ ] **13:00-13:15** Add missing `lithuanian_address_full` pattern
- [ ] **13:15-13:30** Fix "Vilniaus" location detection issue
- [ ] **13:30-13:45** **COMMIT:** "Align Lithuanian pattern detection with test expectations"

**Sprint 2 Success Criteria:**
- ✅ Lithuanian pattern tests pass
- ✅ Car plate detection works as expected
- ✅ Address patterns correctly implemented

---

## SPRINT 3: PDF & PATTERN LEARNING (14:30-16:30)
**Goal:** Complete test suite fixes → 15/15 failures resolved (100% pass)

### Hour 1: PDF Processing (14:30-15:30)
- [ ] **14:30-14:45** Reproduce PDF redaction failures
  ```bash
  pytest tests/test_pdf_processor.py::TestPDFProcessorIntegration -v
  ```
- [ ] **14:45-15:00** Check redaction annotation logic in PDF processor
- [ ] **15:00-15:15** Fix lithuanian_personal_codes detection mapping
- [ ] **15:15-15:30** Fix redaction annotation creation

### Hour 2: Pattern Learning & Monitoring (15:30-16:30)
- [ ] **15:30-15:45** Fix adaptive pattern learner threshold logic
  ```bash
  pytest tests/adaptive/test_pattern_learner.py::TestDiscoverAndValidatePatterns -v
  ```
- [ ] **15:45-16:00** Fix real-time monitoring integration
  ```bash
  pytest tests/system/test_real_time_monitor_integration.py -v
  ```
- [ ] **16:00-16:15** Verify all fixes work in isolation
- [ ] **16:15-16:30** **COMMIT:** "Fix PDF processing, pattern learning, and monitoring integration"

**Sprint 3 Success Criteria:**
- ✅ PDF redaction annotations created properly
- ✅ Pattern learning thresholds work correctly
- ✅ Real-time monitoring logs metrics

---

## SPRINT 4: VERIFICATION & CI (16:30-17:30)
**Goal:** Confirm green pipeline and >80% coverage

### Full Validation (16:30-17:30)
- [ ] **16:30-16:40** Run complete test suite locally
  ```bash
  pytest --cov=app --cov-report=term-missing
  ```
- [ ] **16:40-16:50** Verify >80% test coverage achieved
- [ ] **16:50-17:00** Push to CI and monitor pipeline
- [ ] **17:00-17:10** Address any CI-specific failures
- [ ] **17:10-17:20** Generate final coverage report
- [ ] **17:20-17:30** **FINAL COMMIT:** "Complete test suite fixes - all tests passing"

**Sprint 4 Success Criteria:**
- ✅ 272/272 tests passing (100% pass rate)
- ✅ >80% test coverage maintained
- ✅ CI pipeline green
- ✅ No linter errors

---

## CHECKPOINT SCHEDULE

| Time | Checkpoint | Expected Status |
|------|------------|-----------------|
| 11:30 | Sprint 1 Complete | 8/15 failures fixed |
| 13:45 | Sprint 2 Complete | 11/15 failures fixed |
| 16:30 | Sprint 3 Complete | 15/15 failures fixed |
| 17:30 | Final Verification | CI green, >80% coverage |

---

## EMERGENCY ESCALATION

**If behind schedule by >30 minutes:**
1. Focus on P0 and P1 fixes only (11/15 failures)
2. Create follow-up tickets for P2/P3 items
3. Ensure CI pipeline is at least functional

**If unexpected issues arise:**
1. Document thoroughly in commit messages
2. Create detailed issue tickets
3. Communicate timeline impact immediately

---

## SUCCESS DELIVERABLES

✅ **ROOT_CAUSE_REPORT.md** - Complete diagnostic analysis  
✅ **DAY_PLAN.md** - This actionable 8-hour plan  
🔄 **Code fixes** - All critical bugs resolved  
🔄 **CI proof** - Green pipeline with screenshot  
🔄 **Coverage report** - >80% maintained  

**Final Status Target:** 272/272 tests passing, 0 failures, CI green

---

**Plan Created:** 2025-06-30 09:15 UTC  
**Ready to Execute:** YES - All blockers identified and actionable 