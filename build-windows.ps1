# AnonymPDF Windows Build Script
# PowerShell script for building Windows executable with PyInstaller

param(
    [string]$Environment = "production",
    [switch]$Clean = $false,
    [switch]$Test = $false,
    [switch]$Verbose = $false
)

# Set error action preference
$ErrorActionPreference = "Stop"

Write-Host "=== AnonymPDF Windows Build Process ===" -ForegroundColor Green

try {
    # Step 1: Clean previous builds
    if ($Clean) {
        Write-Host "[STEP] Cleaning previous builds..." -ForegroundColor Cyan
        $CleanDirs = @("build", "dist", "__pycache__")
        foreach ($dir in $CleanDirs) {
            if (Test-Path $dir) {
                Write-Host "Removing $dir" -ForegroundColor Yellow
                Remove-Item -Path $dir -Recurse -Force -ErrorAction SilentlyContinue
            }
        }
    }

    # Step 2: Verify Python environment
    Write-Host "[STEP] Verifying Python Environment..." -ForegroundColor Cyan
    
    # Check if virtual environment is active
    if (-not $env:VIRTUAL_ENV) {
        Write-Host "WARNING: Virtual environment not detected. Activating venv..." -ForegroundColor Yellow
        if (Test-Path "venv\Scripts\Activate.ps1") {
            & ".\venv\Scripts\Activate.ps1"
            Write-Host "Virtual environment activated" -ForegroundColor Green
        } else {
            Write-Host "ERROR: Virtual environment not found. Please run setup-dev.ps1 first." -ForegroundColor Red
            exit 1
        }
    }

    # Verify Python version
    $pythonVersion = python --version 2>&1
    Write-Host "Python version: $pythonVersion" -ForegroundColor Green

    # Step 3: Install/Update dependencies
    Write-Host "[STEP] Installing/Updating Dependencies..." -ForegroundColor Cyan
    Write-Host "Upgrading pip, setuptools, and wheel..." -ForegroundColor Yellow
    python -m pip install --upgrade pip setuptools wheel

    Write-Host "Installing requirements..." -ForegroundColor Yellow
    pip install -r requirements.txt

    Write-Host "Installing PyInstaller if not present..." -ForegroundColor Yellow
    pip install pyinstaller==6.3.0

    # Step 4: Download spaCy models
    Write-Host "[STEP] Downloading spaCy Language Models..." -ForegroundColor Cyan
    
    $spacyModels = @("en_core_web_sm", "lt_core_news_sm")
    foreach ($model in $spacyModels) {
        Write-Host "Downloading spaCy model: $model..." -ForegroundColor Yellow
        python -m spacy download $model
        
        # Verify model installation
        $modelCheck = python -c "import spacy; spacy.load('$model'); print('Model $model loaded successfully')" 2>&1
        if ($LASTEXITCODE -eq 0) {
            Write-Host $modelCheck -ForegroundColor Green
        } else {
            Write-Host "WARNING: Failed to load $model - continuing anyway" -ForegroundColor Yellow
        }
    }

    # Step 5: Build Frontend
    Write-Host "[STEP] Building Frontend Assets..." -ForegroundColor Cyan
    if (Test-Path "frontend") {
        Set-Location frontend
        
        Write-Host "Installing frontend dependencies..." -ForegroundColor Yellow
        npm install
        
        Write-Host "Building production frontend..." -ForegroundColor Yellow
        npm run build
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "ERROR: Frontend build failed" -ForegroundColor Red
            exit 1
        }
        
        Set-Location ..
        Write-Host "Frontend build completed successfully" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Frontend directory not found - skipping frontend build" -ForegroundColor Yellow
    }

    # Step 6: Validate dependencies
    Write-Host "[STEP] Validating Dependencies..." -ForegroundColor Cyan
    try {
        $validation = python -c "from app.core.dependencies import validate_dependencies_on_startup; validator = validate_dependencies_on_startup(); print('SUCCESS: All dependencies validated' if validator else 'WARNING: Some dependencies missing')" 2>$null
        Write-Host $validation -ForegroundColor Green
    } catch {
        Write-Host "WARNING: Dependency validation step skipped - continuing with build" -ForegroundColor Yellow
    }

    # Step 7: Run PyInstaller
    Write-Host "[STEP] Building Executable with PyInstaller..." -ForegroundColor Cyan
    
    $pyinstallerArgs = @(
        "--clean",
        "--noconfirm"
    )
    
    if ($Verbose) {
        $pyinstallerArgs += "--log-level=DEBUG"
    }
    
    if (Test-Path "AnonymPDF.spec") {
        $pyinstallerArgs += "AnonymPDF.spec"
        Write-Host "Using existing spec file: AnonymPDF.spec" -ForegroundColor Green
    } else {
        Write-Host "WARNING: Spec file not found, using basic PyInstaller command" -ForegroundColor Yellow
        $pyinstallerArgs += @(
            "--onefile",
            "--name=AnonymPDF",
            "--add-data=config;config",
            "--add-data=frontend/dist;frontend/dist",
            "--hidden-import=spacy",
            "--hidden-import=en_core_web_sm",
            "--hidden-import=lt_core_news_sm",
            "app/main.py"
        )
    }
    
    Write-Host "PyInstaller command: pyinstaller $($pyinstallerArgs -join ' ')" -ForegroundColor Cyan
    & pyinstaller @pyinstallerArgs
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: PyInstaller build failed" -ForegroundColor Red
        exit 1
    }

    # Step 8: Verify build output
    Write-Host "[STEP] Verifying Build Output..." -ForegroundColor Cyan
    
    $exePath = "dist\AnonymPDF\AnonymPDF.exe"
    if (-not (Test-Path $exePath)) {
        $exePath = "dist\AnonymPDF.exe"
    }
    
    if (Test-Path $exePath) {
        $fileSize = [math]::Round((Get-Item $exePath).Length / 1MB, 2)
        Write-Host "SUCCESS: Executable created: $exePath ($fileSize MB)" -ForegroundColor Green
        
        # Test executable if requested
        if ($Test) {
            Write-Host "[STEP] Testing Executable..." -ForegroundColor Cyan
            Write-Host "Running basic executable test..." -ForegroundColor Yellow
            $testOutput = & $exePath --help 2>&1
            if ($LASTEXITCODE -eq 0) {
                Write-Host "SUCCESS: Executable test passed" -ForegroundColor Green
            } else {
                Write-Host "WARNING: Executable test failed but file exists" -ForegroundColor Yellow
            }
        }
    } else {
        Write-Host "ERROR: Executable not found in expected location" -ForegroundColor Red
        Write-Host "Contents of dist directory:" -ForegroundColor Yellow
        Get-ChildItem -Path dist -Recurse | ForEach-Object { Write-Host "  $($_.FullName)" -ForegroundColor White }
        exit 1
    }

    # Step 9: Create distribution info
    Write-Host "[STEP] Creating Distribution Information..." -ForegroundColor Cyan
    
    $buildInfo = @{
        "build_date" = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
        "environment" = $Environment
        "python_version" = $pythonVersion
        "executable_path" = $exePath
        "executable_size_mb" = $fileSize
        "build_machine" = $env:COMPUTERNAME
        "build_user" = $env:USERNAME
    }
    
    $buildInfo | ConvertTo-Json -Depth 2 | Out-File -FilePath "dist\build_info.json" -Encoding UTF8
    Write-Host "Build information saved to dist\build_info.json" -ForegroundColor Green

    # Success message
    Write-Host "" -ForegroundColor White
    Write-Host "=== Build Completed Successfully! ===" -ForegroundColor Green
    Write-Host "" -ForegroundColor White
    Write-Host "SUCCESS: Executable location: $exePath" -ForegroundColor Green
    Write-Host "SUCCESS: File size: $fileSize MB" -ForegroundColor Green
    Write-Host "" -ForegroundColor White
    Write-Host "Next steps:" -ForegroundColor Cyan
    Write-Host "  1. Test the executable: $exePath" -ForegroundColor White
    Write-Host "  2. Create installer with Inno Setup (next phase)" -ForegroundColor White
    Write-Host "  3. Test on clean Windows system" -ForegroundColor White

} catch {
    Write-Host "ERROR: Build failed with error: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host "Error details: $($_.Exception.ToString())" -ForegroundColor Red
    exit 1
}

Write-Host "" -ForegroundColor White
Write-Host "Build script completed!" -ForegroundColor Green 