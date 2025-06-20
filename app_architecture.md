# AnonymPDF: Application Architecture Analysis

This document provides a comprehensive architectural analysis of the AnonymPDF Python application. The findings are based on a static analysis of the codebase, focusing on structure, data flow, quality, and maintainability.

## 1. High-Level Architectural Overview

AnonymPDF is a monolithic, full-stack application built with Python and FastAPI. It is designed as a self-contained web service that provides an API for anonymizing PDF documents and includes an integrated frontend. The architecture suggests a design suitable for both local/desktop deployment and small-scale server-based operation.

### 1.1. Architectural Layers

The application follows a classic N-tier architecture pattern, logically separating concerns into distinct layers:

- **Presentation Layer:** A static, single-page application (SPA) frontend (likely React/TypeScript, based on `tsconfig.json` and `.tsx` files in `frontend/`) served directly by the backend.
- **API Layer (`app/api`):** A set of FastAPI routers that define the external RESTful interface for all application functionalities, including PDF processing, analytics, and monitoring.
- **Service Layer (`app/services`):** Orchestrates the core business logic, acting as a mediator between the API endpoints and the core domain logic.
- **Core/Domain Layer (`app/core`):** Contains the heart of the application's logic, including PII detection, text extraction, NLP processing, and data validation. This layer is the most complex.
- **Data Persistence Layer (`app/database`, `app/db`):** Manages all interactions with the database using SQLAlchemy as an ORM. The underlying database is SQLite, indicating a serverless, file-based persistence model.
- **Task-Queue Layer (`app/worker.py`):** An asynchronous task processing system (likely Celery) for handling long-running operations like PDF anonymization, decoupled from the synchronous API requests.

### 1.2. Major Components

- **FastAPI Application (`app/main.py`):** The main entry point. It initializes the application, configures middleware (CORS), mounts the frontend, and includes the API routers.
- **PDF Processing Engine (`app/pdf_processor.py`, `app/services/pdf_processor.py`):** A core component responsible for the primary feature of extracting text from and redacting PDF documents.
- **Multi-lingual NLP Engine (`app/core/nlp_loader.py`):** Utilizes `spaCy` with models for both English (`en_core_web_sm`) and Lithuanian (`lt_core_news_sm`) to perform Named Entity Recognition (NER) for PII detection.
- **Adaptive Learning System (`app/core/adaptive`):** A sophisticated subsystem designed to improve PII detection over time. It includes components for A/B testing, pattern learning, and dynamic rule adjustments. This is a key feature that differentiates the application.
- **Database Manager (`app/database.py`, `app/db/migrations.py`):** Manages the SQLite database connection, schema creation, and a simple, custom migration system.
- **Asynchronous Worker (`app/worker.py`):** Manages out-of-band processing tasks, crucial for preventing API timeouts during heavy computations.

### 1.3. External Dependencies

The application relies on a set of well-defined external libraries and systems:

- **Web Framework:** FastAPI
- **Database:** SQLite (via SQLAlchemy ORM)
- **NLP:** spaCy (with `en_core_web_sm` and `lt_core_news_sm` models)
- **PDF Parsing:** PyMuPDF (`fitz`) and `pdfminer.six`
- **Language Detection:** `langdetect`
- **File System:** Requires local directory access for `uploads`, `processed`, `temp`, and `logs`.

### 1.4. Architectural Patterns

- **Monolithic Deployment:** The frontend and backend are tightly coupled and deployed as a single unit. The `build-windows.ps1` and `AnonymPDF.spec` files suggest the target deployment is a single executable for Windows.
- **Layered (N-Tier) Architecture:** As described in section 1.1.
- **Repository Pattern:** The use of SQLAlchemy abstracts database interactions, resembling the Repository pattern for data access.
- **Dependency Injection:** FastAPI's dependency injection system is used to provide components like database sessions to the API layer.
- **Asynchronous Task Execution (via Task Queue):** Decouples long-running tasks from the HTTP request-response cycle.

## 2. Data Flow

Data moves through the system in a few primary workflows:

1.  **PDF Anonymization Flow (Synchronous Request, Asynchronous Execution):**
    - A user uploads a PDF via the frontend to the `/api/v1/pdf/upload` endpoint.
    - The API endpoint saves the file to the `uploads/` directory and creates a record in the `pdf_documents` SQLite database table.
    - A task is dispatched to the asynchronous worker (e.g., Celery) with the document's ID.
    - The API immediately returns a task ID or document ID to the user.
    - The worker picks up the task, reads the PDF from `uploads/`, uses the NLP and pattern-matching engines to extract and identify PII, and generates a redacted version of the PDF in the `processed/` directory.
    - The `pdf_documents` table is updated with the status (`completed` or `failed`), a path to the processed file, and a `redaction_report`.

2.  **Analytics & Monitoring Flow:**
    - Endpoints under `/api/v1/analytics` and `/api/v1/monitoring` read directly from the application's SQLite databases (e.g., `analytics.db`, `monitoring.db`) to provide performance metrics, processing statistics, and system health data to the frontend.

3.  **Feedback & Adaptive Learning Flow:**
    - Users can submit feedback on redaction quality via a `/api/v1/feedback` endpoint.
    - This feedback is stored and used by the `AdaptiveLearningCoordinator` (`app/core/adaptive/coordinator.py`) to fine-tune PII detection patterns stored in the `patterns.db` SQLite database. This creates a feedback loop to improve the system's accuracy over time.

---
*This document is still under construction. The following sections will be populated after further analysis.*

## 3. Detailed Component Analysis

This section provides a more detailed walkthrough of the key modules and classes identified in the codebase.

### 3.1. API Endpoints (`app/api/endpoints/`)

The API layer is well-structured, with each file in this directory corresponding to a distinct domain of functionality.

- **`pdf.py`:** This is the primary endpoint for the core service.
    - `POST /api/v1/pdf/upload`: Handles file uploads. It performs initial validation (file type), saves the file to a local `uploads/` directory, creates a `PDFDocument` record in the database, and dispatches an asynchronous task to a Celery worker via `process_pdf_task.delay()`. This non-blocking approach is a key architectural strength, allowing the system to handle time-consuming PDF processing without timing out API clients. **(Evidence: `app/api/endpoints/pdf.py`, lines 30-87)**
    - Other endpoints provide resource access for retrieving document status, download links for processed files, and the final redaction report.
- **`analytics.py`:** Exposes endpoints for retrieving application performance and usage metrics.
- **`monitoring.py`:** Provides simple health checks.
- **`feedback.py`:** Allows users to submit feedback, which is a crucial input for the adaptive learning system.

### 3.2. Asynchronous Worker (`app/worker.py`)

This component is responsible for executing the long-running PDF processing tasks.

- **Technology:** It uses **Celery** with a **Redis** broker/backend. This is explicitly configured at the top of the file. **(Evidence: `app/worker.py`, lines 12-18)**
- **Task `process_pdf_task`:** This is the sole task defined in the file.
    - It creates its own database session (`SessionLocal()`), which is a best practice for task queues to ensure session safety.
    - It invokes `pdf_processor_instance.process_pdf()` to perform the actual work. A global instance is used for performance.
    - It meticulously updates the `pdf_documents` table with the processing status (`PROCESSING`, `COMPLETED`, `FAILED`) and saves the final redacted file to the `processed/` directory. **(Evidence: `app/worker.py`, lines 71-115)**

### 3.3. Service Layer: `PDFProcessor` (`app/services/pdf_processor.py`)

This class is the orchestrator for the entire PII detection and redaction process. It is arguably the most critical and complex component in the application.

- **Responsibilities:**
    1.  **Configuration Management:** Loads patterns and settings from a `ConfigManager`.
    2.  **NLP Model Loading:** Safely loads multiple spaCy models with robust fallbacks for different deployment environments (including PyInstaller).
    3.  **Text Extraction:** Calls `extract_text_enhanced` to get text from PDFs.
    4.  **Language Detection:** Determines the document language (`detect_language`).
    5.  **PII Orchestration:** The `find_personal_info` method combines results from spaCy NER, custom regex patterns, and specialized Lithuanian detection functions.
    6.  **Contextual Validation:** It uses a suite of "Priority 2" helper classes (`ContextualValidator`, `AdvancedPatternRefinement`, etc.) to analyze the context of detected entities, assign confidence scores, and reduce false positives. This is the system's most advanced feature. **(Evidence: `app/services/pdf_processor.py`, lines 25-45)**
    7.  **Redaction:** It generates a redaction report and uses `fitz` (PyMuPDF) to create the redacted PDF.

## 4. Logic & Quality Evaluation

This section assesses the quality, maintainability, and correctness of the codebase.

### 4.1. Code Quality & Maintainability

- **High Complexity in Core Service:** The `PDFProcessor` class, while powerful, exhibits signs of being a "God Class." It has over 600 lines of code and manages a vast number of responsibilities. The `find_personal_info` method is particularly long and complex, making it difficult to debug and maintain. **(Evidence: `app/services/pdf_processor.py`)**
    - **Recommendation (Refactoring):** Refactor `PDFProcessor`. The PII detection logic within `find_personal_info` could be extracted into a dedicated `PiiDetectionPipeline` class. This new class would be responsible for composing the different detection strategies (NER, regex, contextual validation) as a series of steps, making the process more modular and testable. **Effort: Large, Impact: High**
- **Good Separation of Concerns (in some areas):** The separation of API endpoints, the use of a service layer, and the delegation to an async worker are all signs of a well-thought-out architecture. The "Priority 2" contextual analysis components are also well-encapsulated.
- **Duplicated Code (`get_pdf_processor`):** There are two `get_pdf_processor` functions, one in `app/dependencies.py` and one in `app/worker.py`. While they serve different contexts (API vs. worker), this duplication could lead to inconsistencies.
    - **Recommendation (Refactoring):** Centralize the `PDFProcessor` instantiation logic. A single factory function could be created and imported in both places, configured with a parameter to handle the minor differences between API and worker environments. **Effort: Small, Impact: Medium**
- **Hard-coded Configuration:** Some configuration values, such as lists of excluded terms (`GEOGRAPHIC_EXCLUSIONS`, `DOCUMENT_TERMS`), are hard-coded directly in Python files. **(Evidence: `app/core/validation_utils.py`)**
    - **Recommendation (Configuration):** Move all such lists and magic values into the `config/` YAML files. This would make them easier to modify without changing the code. **Effort: Medium, Impact: Medium**

### 4.2. Algorithmic Correctness & Scalability (ML)

- **Advanced PII Detection:** The system goes far beyond simple regex matching. The use of spaCy for NER combined with contextual validators that analyze surrounding text is a sophisticated approach. The introduction of confidence scores is a best practice for managing uncertainty in ML-based systems.
- **Scalability Bottleneck (CPU-Bound):** The entire PII detection process is CPU-bound and runs within a single worker process for each task. While the async architecture allows the system to handle *concurrent* requests, the processing time for a single large document will not improve.
    - **Recommendation (Architectural Evolution):** For very large-scale deployments, the PII detection pipeline itself could be broken down into smaller, distributable tasks. For example, text extraction could be one step, NER another, and regex matching a third, potentially running in parallel. This would be a significant architectural change towards a microservices-style pipeline. **Effort: Large, Impact: High**
- **Stateful NLP Models:** The spaCy models are loaded once into memory in the `PDFProcessor` instance for performance. This is a good optimization for a single-worker environment but requires careful memory management. The current memory footprint is acceptable for the chosen models.

### 4.3. Test Coverage

- The `cov_report.txt` file indicates a coverage of 64%, which is insufficient for a production system, especially given the complexity of the logic. The low coverage means there is a high risk of regressions.
    - **Recommendation (Testing):** Increase test coverage to at least 85-90%. Priority should be given to the complex logic in `app/services/pdf_processor.py` and `app/core/validation_utils.py`. Unit tests should be written for each detection and validation function in isolation. **Effort: Large, Impact: High**

## 5. Growth & Improvement Potential

This section outlines recommendations for future development.

| Recommendation                               | Category                   | Description                                                                                                                                                             | Effort | Impact |
| -------------------------------------------- | -------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ | ------ |
| Refactor `PDFProcessor`                      | Refactoring                | Extract PII detection logic into a new, modular `PiiDetectionPipeline` class to reduce complexity and improve testability.                                                | L      | High   |
| Increase Test Coverage                       | Testing / CI/CD            | Increase unit and integration test coverage to >85%, focusing on the core processing and validation logic to reduce the risk of regressions.                                | L      | High   |
| Centralize `PDFProcessor` Instantiation      | Refactoring                | Create a single factory for the `PDFProcessor` instance to remove duplicated code between the web and worker processes.                                                   | S      | Med    |
| Externalize Hard-coded Rules                 | Configuration              | Move lists of terms and other magic values from code files (`validation_utils.py`) to the YAML configuration files to improve maintainability.                          | M      | Med    |
| Evolve to a Micro-pipeline                 | Architectural Evolution    | For future scalability, break the PII detection process into a series of independent, communicating services (e.g., text extraction, NER, pattern matching).              | L      | High   |
| Introduce Typed Interfaces                   | Code Quality               | While FastAPI uses Pydantic, the internal function signatures could benefit from more explicit `TypedDict` or `dataclass` usage for complex dictionary objects.          | M      | Med    |
| Formalize Database Migrations                | CI/CD / DevOps             | Replace the manual migration script with a dedicated tool like **Alembic**. This provides robust, reversible, and auto-generated database schema migrations.             | M      | High   |

## 6. Glossary

- **PII:** Personally Identifiable Information. The core data the application seeks to find and redact.
- **NER:** Named Entity Recognition. An NLP technique used to identify entities like people, organizations, and locations in text.
- **Redaction:** The process of removing or obscuring information from a document.
- **Adaptive Learning:** The system's ability to improve its detection accuracy over time based on user feedback and observed patterns.
- **Confidence Score:** A value assigned to a PII detection indicating the system's confidence in its correctness.

---
*Final review pending.* 