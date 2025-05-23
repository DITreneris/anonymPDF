# AnonymPDF API Documentation

## Base URL
```
http://localhost:8000/api/v1
```

## Endpoints

### Upload PDF
Upload a PDF file for anonymization.

**Endpoint:** `POST /upload`

**Request:**
- Content-Type: `multipart/form-data`
- Body: PDF file

**Response:**
```json
{
    "id": 1,
    "original_filename": "example.pdf",
    "anonymized_filename": "anonymized_example.pdf",
    "file_size": 12345,
    "status": "completed",
    "created_at": "2024-03-14T12:00:00",
    "updated_at": "2024-03-14T12:00:05",
    "error_message": "Redaction Report..."
}
```

### List Documents
Get a list of all processed documents.

**Endpoint:** `GET /documents`

**Query Parameters:**
- `skip` (optional): Number of records to skip (default: 0)
- `limit` (optional): Maximum number of records to return (default: 100)

**Response:**
```json
[
    {
        "id": 1,
        "original_filename": "example.pdf",
        "anonymized_filename": "anonymized_example.pdf",
        "file_size": 12345,
        "status": "completed",
        "created_at": "2024-03-14T12:00:00",
        "updated_at": "2024-03-14T12:00:05",
        "error_message": "Redaction Report..."
    }
]
```

### Get Document
Get details of a specific document.

**Endpoint:** `GET /documents/{document_id}`

**Response:**
```json
{
    "id": 1,
    "original_filename": "example.pdf",
    "anonymized_filename": "anonymized_example.pdf",
    "file_size": 12345,
    "status": "completed",
    "created_at": "2024-03-14T12:00:00",
    "updated_at": "2024-03-14T12:00:05",
    "error_message": "Redaction Report..."
}
```

### Get Document Report
Get the redaction report for a specific document.

**Endpoint:** `GET /documents/{document_id}/report`

**Response:**
```json
{
    "report": "Redaction Report - 2024-03-14 12:00:05\n--------------------------------------------------\nDetected Language: en\n--------------------------------------------------\n\nNAMES:\n- John Doe (PERSON)\n\nLOCATIONS:\n- New York City (GPE)\n\nEMAILS:\n- john.doe@example.com (EMAIL)\n\nPHONES:\n- (555) 123-4567 (PHONE)\n\nSSNS:\n- 123-45-6789 (SSN)\n\nCREDIT_CARDS:\n- 1234 5678 9012 3456 (CREDIT_CARD)"
}
```

### Download Document
Download an anonymized PDF document.

**Endpoint:** `GET /documents/{document_id}/download`

**Response:**
- Content-Type: `application/pdf`
- Body: PDF file

## Error Responses

### 400 Bad Request
```json
{
    "detail": "Only PDF files are allowed"
}
```

### 404 Not Found
```json
{
    "detail": "Document not found"
}
```

### 500 Internal Server Error
```json
{
    "detail": "Failed to process PDF: [error message]"
}
```

## Status Codes

- `200 OK`: Request successful
- `400 Bad Request`: Invalid request
- `404 Not Found`: Resource not found
- `500 Internal Server Error`: Server error

## Notes

- All timestamps are in ISO 8601 format
- File sizes are in bytes
- The redaction report includes detected language and all redacted information
- Documents are stored in the `processed` directory after anonymization 