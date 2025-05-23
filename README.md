# AnonymPDF

A web-based tool for automatically anonymizing PDF documents by removing personal data (names, personal codes, contacts, etc.) before sending them to AI analysis platforms.

## Features

- PDF document upload and processing
- Multi-language support (Lithuanian and English for text analysis)
- Automatic detection and redaction of personal information using NLP and regex patterns.
- Real PDF redaction using PyMuPDF (sensitive content is removed, not just covered).
- User-friendly web interface built with React and Material-UI.
- Handles documents by processing them and providing an anonymized version for download.

## Technologies Used

**Backend:**
- Python 3.9+
- FastAPI: For building the REST API.
- Uvicorn: ASGI server for FastAPI.
- spaCy: For Natural Language Processing (NER).
  - `en_core_web_sm` (English model)
  - `lt_core_news_sm` (Lithuanian model)
- PyMuPDF (fitz): For PDF parsing and redaction.
- SQLAlchemy: For database interaction (with SQLite by default).
- Pydantic: For data validation.

**Frontend:**
- Node.js and npm/yarn
- React
- TypeScript
- Material-UI: For UI components.
- Axios: For API communication.
- React Dropzone: For file uploads.

**Database:**
- SQLite (default, via `anonympdf.db`)

## Setup

### Backend

1. **Clone the repository (if you haven't already):**
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Linux/macOS
   source venv/bin/activate
   # On Windows (PowerShell)
   .\venv\Scripts\Activate.ps1
   # On Windows (CMD)
   venv\Scripts\activate.bat
   ```

3. **Install backend dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download spaCy NLP models:**
   ```bash
   python -m spacy download en_core_web_sm
   python -m spacy download lt_core_news_sm
   ```

5. **Run the backend server:**
   The backend runs on `http://localhost:8000` by default.
   ```bash
   uvicorn app.main:app --reload
   ```

### Frontend

1. **Navigate to the frontend directory:**
   ```bash
   cd frontend # Corrected path
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   # or if you use yarn: yarn install
   ```

3. **Run the frontend development server:**
   The frontend typically runs on `http://localhost:3000`.
   ```bash
   npm start
   # or if you use yarn: yarn start
   ```

After starting both backend and frontend servers, open your browser and navigate to the frontend URL (e.g., `http://localhost:3000`).

## Development

The project follows a modular structure:

- `app/`: Main backend application code
  - `api/`: API endpoints (FastAPI routers)
  - `db/`: Database setup and session management (e.g., `database.py`)
  - `models/`: SQLAlchemy database models (e.g., `pdf_document.py`)
  - `pdf_processor.py`: Module containing low-level PDF redaction logic using PyMuPDF (e.g., the `redact_pdf` function).
  - `schemas/`: Pydantic schemas for data validation and serialization.
  - `services/`: Business logic and core processing classes.
    - `pdf_processor.py`: Contains the `PDFProcessor` class orchestrating text extraction, PII detection (spaCy, regex), and invoking redaction.
  - `main.py`: FastAPI application entry point.
- `frontend/`: Frontend application code (React, TypeScript, Material-UI) # Corrected path
  - `src/`: Source files
    - `components/`: Reusable React components
    - `App.tsx`: Main application component
- `tests/`: Test files for the backend.
- `uploads/`: Directory for uploaded original files (temporary storage during processing).
- `processed/`: Directory for anonymized PDF files.
- `temp/`: Temporary directory for backend processing if needed.

## API Documentation

Once the backend server is running, API documentation is available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE.md) file for details.
(Note: Create a `LICENSE.md` file with the MIT License text if it doesn't exist)