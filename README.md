# AnonymPDF - Local Desktop PDF Anonymization Tool

![Version](https://img.shields.io/badge/version-2.0.0-blue.svg)

A robust local desktop application for automatically anonymizing PDF documents by detecting and redacting personal information (PII) before sharing or analysis. Built with enterprise-grade logging, error handling, and modern UI/UX design.

---

**Versioning Policy:** This project uses [Semantic Versioning](https://semver.org/). Major UI/UX or workflow changes increment the MAJOR version. Minor features and fixes increment the MINOR or PATCH version.

## 🎯 Current Status: Phase 1, 2 & 3 Complete ✅

**Latest Update:** May 28, 2025  
**Implementation Status:** Foundation, Error Handling & UI/UX COMPLETED  
**Next Phase:** Advanced Features & Accessibility  

### ✅ Key Features Implemented
- **Modern Two-Pane UI:** Responsive layout with integrated statistics and educational content
- **Complete Upload Workflow:** Drag & drop PDF upload with real-time progress tracking
- **Redaction Statistics Display:** Visual breakdown of detected PII categories and redaction counts
- **Advanced PII Detection (35+ Types):** Names, emails, phone numbers, Lithuanian personal codes, VAT codes, IBANs, addresses, cities, business certificates
- **Lithuanian Location Intelligence:** 60+ cities, districts, and neighborhoods with smart matching
- **Real PDF Redaction:** Content permanently removed using PyMuPDF (not just covered)
- **Multi-language Support:** English (`en_core_web_sm`) and Lithuanian (`lt_core_news_sm`) NLP models
- **Enterprise Logging:** Structured logging with file rotation and contextual information
- **Professional Error Handling:** Material-UI error dialogs with recovery suggestions
- **Dependency Validation:** Startup checks with clear installation guidance
- **Database Migration System:** Automatic schema updates and data migration
- **React Error Boundaries:** Application-wide error recovery

## 🏗️ Architecture

**Deployment Model:** Local desktop application (single-user)  
**Database:** SQLite for local data storage  
**Backend:** FastAPI (Python) serving localhost  
**Frontend:** React 19.1 with TypeScript and Material-UI v7.1.0  

### Backend Core
- **PDF Processing:** PyMuPDF (fitz) for content redaction
- **PII Detection:** spaCy NER + custom regex patterns
- **Logging:** Structured logging with rotation (10MB files, 5 backups)
- **Error Handling:** Comprehensive error tracking and recovery
- **Dependencies:** Automatic validation with clear error messages
- **API Schema:** Complete redaction report integration

### Frontend Core
- **UI Framework:** React 19.1 with TypeScript (strict mode)
- **Components:** Material-UI v7.1.0 with modern design system
- **Layout:** Responsive two-pane layout (45%/55% split)
- **Error Handling:** Error boundaries with fallback UI and recovery options
- **File Handling:** Drag-and-drop upload with progress tracking
- **Statistics Display:** Integrated redaction statistics and educational content

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Node.js 16+
- npm or yarn

### 1. Backend Setup

```bash
# Clone and navigate to project
git clone <repository-url>
cd AnonymPDF

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows PowerShell:
.\venv\Scripts\Activate.ps1
# Windows CMD:
venv\Scripts\activate.bat
# Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download required spaCy models
python -m spacy download en_core_web_sm
python -m spacy download lt_core_news_sm

# Start backend server
uvicorn app.main:app --reload
```

Backend will be available at: `http://localhost:8000`

### 2. Frontend Setup

```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install

# Start development server (Vite)
npm run dev
```

Frontend will be available at: `http://localhost:5173` (Vite default)

## 📁 Project Structure

```
AnonymPDF/
├── app/                          # Backend application
│   ├── core/                     # Core utilities
│   │   ├── logging.py           # ✅ Structured logging system
│   │   └── dependencies.py      # ✅ Dependency validation
│   ├── db/                       # Database layer
│   │   ├── database.py          # Database configuration
│   │   └── migrations.py        # ✅ Migration system
│   ├── models/                   # Database models
│   │   └── pdf_document.py      # ✅ Updated schema with redaction_report
│   ├── api/endpoints/            # API endpoints
│   │   └── pdf.py              # ✅ Enhanced with logging & report integration
│   ├── services/                 # Business logic
│   │   └── pdf_processor.py     # ✅ Enhanced processing
│   ├── schemas/                  # Pydantic schemas
│   │   └── pdf.py              # ✅ Updated with redaction_report field
│   └── main.py                  # ✅ App entry with validation
├── frontend/                     # Frontend application
│   ├── src/
│   │   ├── components/
│   │   │   ├── layout/
│   │   │   │   └── TwoPaneLayout.tsx    # ✅ Responsive two-pane layout
│   │   │   ├── upload/
│   │   │   │   └── FileUploadZone.tsx   # ✅ Enhanced drag-drop upload
│   │   │   ├── ui/
│   │   │   │   └── StatisticsPanel.tsx  # ✅ NEW: Combined stats & education
│   │   │   ├── ErrorBoundary.tsx        # ✅ Error boundary
│   │   │   ├── ErrorDialog.tsx          # ✅ Enhanced error dialog
│   │   │   └── RedactionReport.tsx      # ✅ Legacy report display
│   │   ├── utils/
│   │   │   └── errorHandler.ts          # ✅ Error handling utilities
│   │   └── App.tsx                      # ✅ Modern UI with state management
│   ├── package.json             # Vite configuration
│   └── tsconfig.app.json        # ✅ Strict TypeScript
├── logs/                         # ✅ Application logs
│   └── anonympdf.log            # Structured log output
├── uploads/                      # Temporary file storage
├── processed/                    # Anonymized files
└── requirements.txt             # Python dependencies
```

## 🎨 User Interface Features

### ✅ Modern Two-Pane Layout
- **Left Pane (45%):** Upload controls, file status, and processing feedback
- **Right Pane (55%):** Combined redaction statistics and educational content
- **Responsive Design:** Adapts to different screen sizes
- **Professional Styling:** Single brand accent color (#005FCC) with neutral design

### ✅ Enhanced Upload Experience
- **Drag & Drop:** Intuitive file upload with visual feedback
- **File Validation:** PDF-only, 50MB limit, corruption detection
- **Progress Tracking:** Real-time upload and processing progress
- **Error Handling:** Clear error messages with recovery suggestions

### ✅ Redaction Statistics Display
- **Visual Statistics Cards:** Total redactions and PII categories count
- **Detailed Breakdown:** Expandable list of detected PII categories
- **Language Detection:** Shows detected document language
- **Action Buttons:** Download anonymized PDF and export report
- **Educational Content:** Always-visible "How It Works" information

### ✅ Design System
- **Color Palette:** Single cobalt blue accent with neutral grays
- **Typography:** Inter font with proper hierarchy
- **Spacing:** 8px scale with consistent padding
- **Cards:** White background with subtle shadows and borders
- **Accessibility:** Enhanced contrast for better readability

## 🚀 Current Functionality Status

### ✅ Fully Working Features
1. **Complete Upload Workflow:**
   - Drag & drop PDF upload with validation
   - Real-time progress tracking (upload percentage)
   - Comprehensive error handling with recovery suggestions

2. **PDF Processing:**
   - Automatic language detection (Lithuanian/English)
   - Advanced PII detection (35+ pattern types)
   - Real content redaction using PyMuPDF
   - Processing typically completes in 2-3 seconds

3. **Results Display:**
   - **Visual Statistics:** Total redactions and PII categories count
   - **Detailed Breakdown:** Expandable list showing detected categories
   - **Language Detection:** Shows detected document language
   - **Download Functionality:** Download anonymized PDF file
   - **Report Export:** Export detailed JSON report

4. **User Experience:**
   - Professional two-pane responsive layout
   - Combined statistics and educational content
   - "Process Another File" workflow reset
   - Context-aware error dialogs

### 📊 Example Processing Results
Based on real testing with Lithuanian documents:
- **43 Total Redactions** detected and removed
- **16 PII Categories** identified (names, addresses, phone numbers, etc.)
- **Lithuanian Language** automatically detected
- **Processing Time:** ~2.2 seconds for 183KB PDF

### 🔧 Technical Status
- **Build Status:** ✅ Successful TypeScript compilation
- **API Integration:** ✅ Complete backend-frontend communication
- **Error Handling:** ✅ Comprehensive error recovery
- **Component Architecture:** ✅ Modular, maintainable design

## 🔧 Available Scripts

### Backend
```bash
uvicorn app.main:app --reload     # Start development server
uvicorn app.main:app             # Start production server
```

### Frontend
```bash
npm run dev                      # Start Vite development server
npm run build                    # Build for production
npm run preview                  # Preview production build
npm run lint                     # Run ESLint
```

## 🛡️ PII Detection Patterns (45+ Types)

### 🧠 AI-Powered Detection
- **Names:** spaCy PERSON entity detection (EN & LT models)
- **Organizations:** spaCy ORG entity detection
- **Locations:** spaCy GPE entities + comprehensive Lithuanian city database

### 📍 Lithuanian Location Intelligence
- **60+ Cities & Districts:** Vilnius, Kaunas, Klaipėda, Šiauliai, Panevėžys, Alytus, etc.
- **Administrative Divisions:** Vilniaus raj., Kauno sav., Telšių aps.
- **Neighborhoods:** Antakalnis, Žirmūnai, Lazdynai, Fabijoniškės, etc.
- **Smart Matching:** Word-boundary detection to avoid false positives

### 📞 Contact Information
- **Email Addresses:** Standard email pattern detection
- **Lithuanian Phone Numbers:**
  - Formatted: `+370 600 55678`
  - Prefixed: `Tel. nr.: +370 600 55678`
  - Compact: `+37060055678`
- **International Phones:** Generic international formats

### 🆔 Identity Documents
- **Lithuanian Passports:** `LT1234567` (2 letters + 7 digits)
- **Driver's Licenses:** `AB123456C` (1-2 letters + 6-7 digits + optional letter)
- **Lithuanian Personal Codes:** Asmens Kodas (`38901234567`)

### 🏥 Healthcare & Medical
- **Health Insurance Numbers:** 6-12 digit health insurance/resident certificate numbers
- **Blood Groups:** `A+`, `A-`, `B+`, `B-`, `AB+`, `AB-`, `O+`, `O-`
- **Medical Record Numbers:** 6-10 digit medical record identifiers

### 🚗 Automotive
- **Lithuanian Car Plates:** `ABC-123`, `DEF 456` (3 letters + 3 digits)

### 🏛️ Government & Business IDs
- **VAT Codes:**
  - Labeled: `PVM kodas: LT100001738313`
  - Standalone: `LT100001738313`
- **Business Certificates:** `AF123456-1`, 9-digit business codes
- **Legal Entity Codes:** 8-9 digit legal entity identifiers

### 💳 Enhanced Financial Information
- **Lithuanian IBANs:** `LT123456789012345678`
- **EU IBANs:** Universal IBAN format for all EU countries (`DE89370400440532013000`)
- **SWIFT/BIC Codes:** `CBVILT2X` (8-11 character bank codes)
- **Enhanced Credit Cards:**
  - Visa: `4532123456789012`
  - MasterCard: `5555555555554444`
  - American Express: `378282246310005`
  - Discover: `6011123456789012`
- **Generic Credit Cards:** Legacy pattern support
- **SSNs:** Social Security Number patterns

### 🏠 Address & Location Data
- **Street Addresses:**
  - Prefixed: `Adresas: Paupio g. 50-136`
  - Generic: `Gedimino pr. 25`, `Vilniaus g. 123-45`
- **Postal Codes:** `LT-11341`, `LT-01103`
- **Address Components:** Street types (g., pr., al.)

### 📅 Temporal Information
- **Date Formats:**
  - ISO format: `2024-01-15`
  - European format: `1989.03.15`

## 📊 Quality Standards Achieved

### ✅ Code Quality
- Zero print statements in production code
- 100% TypeScript strict mode compliance
- Comprehensive error handling throughout
- Structured logging for all operations

### ✅ Reliability
- Graceful dependency validation on startup
- Application recovery from component failures
- Data integrity maintained across operations
- Clear error messages with recovery suggestions

### ✅ User Experience
- Professional error dialogs with context-aware suggestions
- Intuitive error recovery workflows
- Reliable processing for files up to 10MB
- Real-time feedback during processing

## 🔍 API Documentation

When the backend is running, comprehensive API documentation is available:
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

## 📝 Logging

The application uses structured logging with:
- **Log Location:** `logs/anonympdf.log`
- **Rotation:** 10MB files, 5 backups
- **Levels:** DEBUG, INFO, WARNING, ERROR
- **Context:** File names, processing times, error details, user actions

## 🚨 Error Handling

### Error Types
- **Validation Errors:** File format, size, corruption issues
- **Processing Errors:** PDF parsing, PII detection failures
- **System Errors:** Internal server errors, dependency issues
- **Network Errors:** Connection timeouts, API failures
- **Timeout Errors:** Long-running operation failures

### Recovery Features
- Context-aware recovery suggestions
- Retry mechanisms for retryable errors
- Fallback UI for component failures
- Detailed error logging for debugging

## 🔄 Development Roadmap

### ✅ Phase 1: Foundation (COMPLETED)
- Structured logging framework
- Database schema refactoring
- Dependency validation system
- TypeScript strict mode

### ✅ Phase 2: Enhanced Error Handling (COMPLETED)
- Material-UI error dialogs
- React error boundaries
- Comprehensive error recovery

### ✅ Phase 3: UI/UX Enhancement (COMPLETED)
- Modern two-pane responsive layout
- Enhanced file upload with drag & drop
- Redaction statistics display
- Professional design system
- Integrated educational content

### 🔄 Phase 4: Advanced Features (CURRENT)
- Enhanced accessibility (WCAG AA compliance)
- File preview system with PDF thumbnails
- Advanced progress feedback with stages
- Guided onboarding for new users

### 🔄 Phase 5: Performance & Polish (FUTURE)
- Performance optimization and bundle splitting
- Installation package creation
- Analytics and monitoring
- Internationalization support

## 🤝 Contributing

1. Follow TypeScript strict mode requirements
2. Use structured logging (no print statements)
3. Include comprehensive error handling
4. Add unit tests for new features
5. Ensure user-friendly error messages

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.

---

**Built for local desktop deployment with enterprise-grade reliability and user experience.**