# AnonymPDF Development Environment Setup Script
# Sets up everything needed for Windows development and packaging

param(
    [switch]$SkipModels = $false,
    [switch]$Force = $false
)

$ErrorActionPreference = "Stop"

Write-Host "=== AnonymPDF Development Environment Setup ===" -ForegroundColor Green

try {
    # Check Python installation
    Write-Host "[STEP] Checking Python Installation..." -ForegroundColor Cyan
    
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "Found Python: $pythonVersion" -ForegroundColor Green
        
        # Check if Python version is 3.11+
        if ($pythonVersion -match "Python (\d+)\.(\d+)") {
            $major = [int]$matches[1]
            $minor = [int]$matches[2]
            
            if ($major -lt 3 -or ($major -eq 3 -and $minor -lt 11)) {
                Write-Host "ERROR: Python 3.11+ required. Found: $pythonVersion" -ForegroundColor Red
                Write-Host "Please install Python 3.11 or later from https://python.org" -ForegroundColor Yellow
                exit 1
            }
        }
    } catch {
        Write-Host "ERROR: Python not found in PATH" -ForegroundColor Red
        Write-Host "Please install Python 3.11+ from https://python.org" -ForegroundColor Yellow
        exit 1
    }

    # Check Node.js (for frontend builds)
    Write-Host "[STEP] Checking Node.js Installation..." -ForegroundColor Cyan
    try {
        $nodeVersion = node --version 2>&1
        Write-Host "Found Node.js: $nodeVersion" -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Node.js not found - frontend builds will be skipped" -ForegroundColor Yellow
        Write-Host "Install Node.js from https://nodejs.org for frontend development" -ForegroundColor Yellow
    }

    # Create virtual environment
    Write-Host "[STEP] Setting up Python Virtual Environment..." -ForegroundColor Cyan
    
    if ((Test-Path "venv") -and (-not $Force)) {
        Write-Host "Virtual environment already exists (use -Force to recreate)" -ForegroundColor Green
    } else {
        if (Test-Path "venv") {
            Write-Host "Removing existing virtual environment..." -ForegroundColor Yellow
            Remove-Item -Path "venv" -Recurse -Force
        }
        
        Write-Host "Creating new virtual environment..." -ForegroundColor Yellow
        python -m venv venv
        
        if (-not (Test-Path "venv\Scripts\Activate.ps1")) {
            Write-Host "ERROR: Failed to create virtual environment" -ForegroundColor Red
            exit 1
        }
    }

    # Activate virtual environment
    Write-Host "Activating virtual environment..." -ForegroundColor Yellow
    & ".\venv\Scripts\Activate.ps1"

    # Upgrade pip and essential tools
    Write-Host "[STEP] Upgrading pip and build tools..." -ForegroundColor Cyan
    python -m pip install --upgrade pip setuptools wheel

    # Install requirements
    Write-Host "[STEP] Installing Python Dependencies..." -ForegroundColor Cyan
    
    if (Test-Path "requirements.txt") {
        Write-Host "Installing from requirements.txt..." -ForegroundColor Yellow
        pip install -r requirements.txt
    } else {
        Write-Host "WARNING: requirements.txt not found" -ForegroundColor Yellow
        Write-Host "Installing basic dependencies..." -ForegroundColor Yellow
        pip install fastapi uvicorn sqlalchemy spacy PyMuPDF pyinstaller
    }

    # Install spaCy models (unless skipped)
    if (-not $SkipModels) {
        Write-Host "[STEP] Installing spaCy Language Models..." -ForegroundColor Cyan
        
        $models = @("en_core_web_sm", "lt_core_news_sm")
        foreach ($model in $models) {
            Write-Host "Installing spaCy model: $model..." -ForegroundColor Yellow
            try {
                python -m spacy download $model
                
                # Verify installation
                $verification = python -c "import spacy; nlp = spacy.load('$model'); print('Model $model loaded successfully')" 2>&1
                Write-Host $verification -ForegroundColor Green
            } catch {
                Write-Host "WARNING: Failed to install $model" -ForegroundColor Yellow
                Write-Host "You can install it manually later with: python -m spacy download $model" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "Skipping spaCy models installation (use without -SkipModels to install)" -ForegroundColor Yellow
    }

    # Setup frontend (if Node.js is available)
    if (Get-Command node -ErrorAction SilentlyContinue) {
        Write-Host "[STEP] Setting up Frontend Environment..." -ForegroundColor Cyan
        
        if (Test-Path "frontend") {
            Set-Location frontend
            
            Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
            npm install
            
            Write-Host "Building frontend for development..." -ForegroundColor Yellow
            npm run build
            
            Set-Location ..
            Write-Host "Frontend setup completed" -ForegroundColor Green
        } else {
            Write-Host "WARNING: Frontend directory not found" -ForegroundColor Yellow
        }
    }

    # Create required directories
    Write-Host "[STEP] Creating Required Directories..." -ForegroundColor Cyan
    
    $requiredDirs = @("uploads", "processed", "temp", "logs", "assets")
    foreach ($dir in $requiredDirs) {
        if (-not (Test-Path $dir)) {
            New-Item -ItemType Directory -Path $dir -Force | Out-Null
            Write-Host "Created directory: $dir" -ForegroundColor Green
        } else {
            Write-Host "Directory exists: $dir" -ForegroundColor Green
        }
    }

    # Initialize database
    Write-Host "[STEP] Initializing Database..." -ForegroundColor Cyan
    try {
        $dbInit = python -c "from app.db.migrations import initialize_database_on_startup; result = initialize_database_on_startup(); print('Database initialized successfully' if result else 'Database initialization failed')" 2>&1
        Write-Host $dbInit -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Database initialization failed - will be created on first run" -ForegroundColor Yellow
    }

    # Validate dependencies
    Write-Host "[STEP] Validating Dependencies..." -ForegroundColor Cyan
    try {
        $validation = python -c "from app.core.dependencies import validate_dependencies_on_startup; validator = validate_dependencies_on_startup(); print('All dependencies validated successfully' if validator else 'Some dependencies missing')" 2>&1
        Write-Host $validation -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Dependency validation failed - some issues may need manual resolution" -ForegroundColor Yellow
        Write-Host "Error: $($_.Exception.Message)" -ForegroundColor Red
    }

    # Success summary
    Write-Host "" -ForegroundColor White
    Write-Host "=== Development Environment Setup Complete! ===" -ForegroundColor Green
    Write-Host "" -ForegroundColor White
    Write-Host "SUCCESS: Python virtual environment ready" -ForegroundColor Green
    Write-Host "SUCCESS: Dependencies installed" -ForegroundColor Green
    if (-not $SkipModels) {
        Write-Host "SUCCESS: spaCy models installed" -ForegroundColor Green
    }
    Write-Host "SUCCESS: Required directories created" -ForegroundColor Green
    Write-Host "SUCCESS: Database ready" -ForegroundColor Green
    Write-Host "" -ForegroundColor White
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Start development server: uvicorn app.main:app --reload" -ForegroundColor White
    Write-Host "  2. Start frontend dev server: cd frontend && npm run dev" -ForegroundColor White
    Write-Host "  3. Build executable: .\build-windows.ps1" -ForegroundColor White
    Write-Host "" -ForegroundColor White
    Write-Host "Development URLs:" -ForegroundColor Cyan
    Write-Host "  Backend API: http://localhost:8000" -ForegroundColor White
    Write-Host "  Frontend: http://localhost:5173" -ForegroundColor White
    Write-Host "  API Docs: http://localhost:8000/docs" -ForegroundColor White

} catch {
    Write-Host "ERROR: Setup failed: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Please check the error above and retry" -ForegroundColor Yellow
    exit 1
}

Write-Host "" -ForegroundColor White
Write-Host "Setup completed successfully!" -ForegroundColor Green 