# Changelog

All notable changes to this project will be documented in this file.

## [2.0.0] - 2025-05-28
### Added
- Major UI/UX overhaul: modern two-pane layout, professional design system
- Unified statistics and information panel (StatisticsPanel)
- Visual redaction statistics and PII category breakdown
- Download and export report buttons
- Language detection display
- Responsive design and improved accessibility
- Version display in frontend and backend

### Fixed
- Critical bug: blank page due to TypeScript errors
- Upload area now fully functional with drag & drop and validation
- Backend now returns redaction report in API response
- Improved error handling and visual hierarchy

## [1.1.0] - 2024-12-15
### Added
- Enhanced Lithuanian PII detection patterns
- City/location intelligence (60+ cities, districts, neighborhoods)
- Improved error dialogs and error boundaries
- Comprehensive test suite (44 tests)

## [1.0.0] - 2024-11-01
### Added
- Initial release: PDF upload, anonymization, and download
- Basic PII detection (names, emails, phone numbers, codes)
- SQLite database, FastAPI backend, React frontend
- Structured logging and dependency validation 