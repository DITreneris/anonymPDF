from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.api.endpoints import pdf
from app.core.dependencies import validate_dependencies_on_startup
from app.db.migrations import initialize_database_on_startup
from app.core.logging import api_logger
from app.version import __version__
import sys

# Validate dependencies and initialize database on startup
try:
    api_logger.info("Starting AnonymPDF application")

    # Validate dependencies first
    dependency_validator = validate_dependencies_on_startup()

    # Initialize database
    if not initialize_database_on_startup():
        api_logger.error("Database initialization failed. Exiting.")
        sys.exit(1)

    api_logger.info("Application startup validation completed successfully")

except Exception as e:
    api_logger.error(f"Application startup failed: {str(e)}")
    sys.exit(1)

app = FastAPI(
    title="AnonymPDF API", description="API for anonymizing PDF documents", version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(pdf.router, prefix="/api/v1", tags=["pdf"])


@app.get("/")
async def root():
    return {"message": "Welcome to AnonymPDF API"}


@app.get("/health")
async def health_check():
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/version")
def get_version():
    return {"version": __version__}


@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": exc.detail},
    )
