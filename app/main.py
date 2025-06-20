from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from app.api.endpoints import pdf, analytics, feedback, monitoring
from app.core.dependencies import validate_dependencies_on_startup
from app.db.migrations import initialize_database_on_startup
from app.core.logging import api_logger
from app.version import __version__
import sys
import os
import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

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

# --- Global UTF-8 Logging Configuration ---
# This is a critical fix for Windows environments to prevent UnicodeEncodeError.
# It ensures all log outputs, including from dependencies, use UTF-8.
def setup_utf8_logging():
    if sys.platform == "win32":
        try:
            # For console output
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
        except TypeError:
            # In some environments (like non-interactive), reconfigure might not be available
            # or needed. We can fallback to a basic configuration.
            pass

    # Configure the root logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout) # Ensure handler uses the reconfigured stdout
        ]
    )
# --- End of Global Logging Configuration ---

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
app.include_router(pdf.router, prefix="/api/v1", tags=["PDF Processing"])
app.include_router(analytics.router, prefix="/api/v1/analytics", tags=["Analytics"])
app.include_router(feedback.router, prefix="/api/v1", tags=["Feedback"])
app.include_router(monitoring.router, prefix="/api/v1/monitoring", tags=["Monitoring"])

# Mount static files for frontend (PyInstaller and dev compatible)
if hasattr(sys, "_MEIPASS"):
    # Running as PyInstaller bundle
    frontend_path = os.path.join(sys._MEIPASS, "frontend", "dist")
    if not os.path.exists(frontend_path):
        # fallback: sometimes just "dist" is present
        frontend_path = os.path.join(sys._MEIPASS, "dist")
else:
    # Running from source
    frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend", "dist")

if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")
    api_logger.info(f"Mounted frontend static files from: {frontend_path}")
else:
    api_logger.warning(f"Frontend directory not found at: {frontend_path}")

@app.get("/api")
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


# Add main entry point for PyInstaller executable
if __name__ == "__main__":
    import uvicorn
    import webbrowser
    import time
    from threading import Timer
    
    api_logger.info("Starting AnonymPDF web server...")
    
    # Function to open browser after server starts
    def open_browser():
        time.sleep(2)  # Wait for server to start
        try:
            webbrowser.open("http://localhost:8000")
            api_logger.info("Opening browser to http://localhost:8000")
        except Exception as e:
            api_logger.warning(f"Could not open browser: {e}")
    
    # Start browser opener in background
    Timer(0.1, open_browser).start()
    
    # Start the server
    try:
        api_logger.info("Starting uvicorn server on http://localhost:8000")
        uvicorn.run(
            app,
            host="127.0.0.1",
            port=8000,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        api_logger.info("Server stopped by user")
    except Exception as e:
        api_logger.error(f"Server error: {e}")
        sys.exit(1)
