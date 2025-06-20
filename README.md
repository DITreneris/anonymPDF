# AnonymPDF

AnonymPDF is an intelligent tool designed to automatically anonymize PDF documents by identifying and redacting Personally Identifiable Information (PII). It leverages a hybrid approach of rules-based detection, Natural Language Processing (NLP), and Machine Learning models to ensure sensitive data is protected.

## Key Features

-   **High-Accuracy PII Redaction:** Reliably finds and redacts a wide range of sensitive information, including names, addresses, contact details, and national identification numbers.
-   **Adaptive Learning:** The system continuously improves its redaction accuracy. When users provide feedback on missed or incorrect redactions, the application learns new patterns and automatically applies them in future processing, reducing the need for manual configuration updates.
-   **Performance Optimized:** Engineered for speed and efficiency, the application uses asynchronous processing and optimized data handling to support high-throughput operations.
-   **Configurable Processing:** Flexible configuration allows for customized redaction rules, multi-language support (including enhanced support for Lithuanian), and fine-tuning of the ML models.
-   **REST API:** A simple RESTful API allows for easy integration into existing workflows and applications.
-   **Real-time Monitoring:** A live dashboard provides insights into system performance, including processing times, CPU/memory usage, and redaction statistics, helping to identify bottlenecks and ensure system health.

## Getting Started

### Prerequisites

-   Python 3.11+
-   `pip` for package management
-   Node.js and `npm` (for frontend development)

### Installation

1.  Clone the repository:
    ```bash
    git clone <repository-url>
    cd AnonymPDF
    ```

2.  Install the backend dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3.  Install the frontend dependencies:
    ```bash
    cd frontend
    npm install
    cd ..
    ```

### Running the Application

**1. Backend Server:**

To start the FastAPI server, run the following command from the project root:

```bash
uvicorn app.main:app --reload
```

The API documentation will be available at `http://127.0.0.1:8000/docs`.

**2. Frontend Development Server:**

To run the frontend UI in development mode with hot-reloading:
```bash
cd frontend
npm run dev
```
This will open the application in your browser, typically at `http://localhost:5173`.

## Testing

The application includes a comprehensive test suite for both the backend and frontend.

### Backend Tests

The backend tests use `pytest`. To run the full suite, execute the following command from the project root:

```bash
pytest
```

This will discover and run all tests in the `tests/` directory, including unit tests, integration tests for the database and API, and system-level workflow tests.

### Frontend Tests

The frontend tests use Jest and React Testing Library. To run them, navigate to the `frontend` directory and run:

```bash
cd frontend
npm test
```

## Usage Example

To anonymize a PDF, send a POST request to the `/api/v1/pdf/process` endpoint with the file attached:

```bash
curl -X POST "http://127.0.0.1:8000/api/v1/pdf/process" \
-H "accept: application/json" \
-H "Content-Type: multipart/form-data" \
-F "file=@/path/to/your/document.pdf;type=application/pdf"
``` 