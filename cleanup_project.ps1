# AnonymPDF Project Cleanup Script
# ================================
# This script organizes working files and cleans up the project structure

Write-Host "🧹 Starting AnonymPDF Project Cleanup..." -ForegroundColor Green

# Create directories if they don't exist
Write-Host "📁 Creating directory structure..." -ForegroundColor Yellow
$directories = @(
    "docs\working",
    "docs\archive\sessions", 
    "logs\archive",
    "temp\reports"
)

foreach ($dir in $directories) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
        Write-Host "  ✅ Created: $dir" -ForegroundColor Green
    } else {
        Write-Host "  📁 Exists: $dir" -ForegroundColor Cyan
    }
}

# Move working documents
Write-Host "📝 Moving working documents..." -ForegroundColor Yellow
$workingFiles = @(
    @{Source="DAY_PLAN.md"; Dest="docs\working\"},
    @{Source="Enhanced_analytics.txt"; Dest="docs\working\"}
)

foreach ($file in $workingFiles) {
    if (Test-Path $file.Source) {
        Move-Item $file.Source $file.Dest -Force
        Write-Host "  ✅ Moved: $($file.Source) → $($file.Dest)" -ForegroundColor Green
    } else {
        Write-Host "  ❌ Not found: $($file.Source)" -ForegroundColor Red
    }
}

# Archive completed session notes
Write-Host "📚 Archiving session notes..." -ForegroundColor Yellow
$sessionFiles = Get-ChildItem -Name "morning_ses*.md" -ErrorAction SilentlyContinue
foreach ($file in $sessionFiles) {
    Move-Item $file "docs\archive\sessions\" -Force
    Write-Host "  ✅ Archived: $file → docs\archive\sessions\" -ForegroundColor Green
}

# Archive old planning document
if (Test-Path "TEST_COV_PLAN.md") {
    Move-Item "TEST_COV_PLAN.md" "docs\archive\" -Force
    Write-Host "  ✅ Archived: TEST_COV_PLAN.md → docs\archive\" -ForegroundColor Green
}

# Archive log files
Write-Host "📋 Archiving log files..." -ForegroundColor Yellow
$logFiles = Get-ChildItem -Name "*_log.txt" -ErrorAction SilentlyContinue
foreach ($file in $logFiles) {
    Move-Item $file "logs\archive\" -Force
    Write-Host "  ✅ Archived: $file → logs\archive\" -ForegroundColor Green
}

if (Test-Path "cov_report.txt") {
    Move-Item "cov_report.txt" "logs\archive\" -Force
    Write-Host "  ✅ Archived: cov_report.txt → logs\archive\" -ForegroundColor Green
}

# Move temporary reports
Write-Host "📊 Moving reports..." -ForegroundColor Yellow
$reportFiles = Get-ChildItem -Name "doc*_report.json" -ErrorAction SilentlyContinue
foreach ($file in $reportFiles) {
    Move-Item $file "temp\reports\" -Force
    Write-Host "  ✅ Moved: $file → temp\reports\" -ForegroundColor Green
}

# Remove temporary test files
Write-Host "🗑️ Removing temporary files..." -ForegroundColor Yellow
if (Test-Path "01_test.py") {
    Remove-Item "01_test.py" -Force
    Write-Host "  ✅ Removed: 01_test.py" -ForegroundColor Green
}

# Check git status for newly ignored files
Write-Host "🔍 Checking Git status..." -ForegroundColor Yellow
try {
    $gitStatus = git status --porcelain 2>$null
    if ($gitStatus) {
        Write-Host "📋 Files that will be ignored by new .gitignore:" -ForegroundColor Cyan
        $gitStatus | Where-Object { $_ -match '\?\?' } | ForEach-Object {
            Write-Host "  📄 $($_.Substring(3))" -ForegroundColor Gray
        }
    }
} catch {
    Write-Host "  ⚠️ Git not available or not a git repository" -ForegroundColor Yellow
}

# Summary
Write-Host ""
Write-Host "✅ Cleanup completed successfully!" -ForegroundColor Green
Write-Host ""
Write-Host "📋 Summary of changes:" -ForegroundColor Cyan
Write-Host "  • Working documents moved to docs/working/" -ForegroundColor White
Write-Host "  • Session notes archived to docs/archive/sessions/" -ForegroundColor White  
Write-Host "  • Log files archived to logs/archive/" -ForegroundColor White
Write-Host "  • Reports moved to temp/reports/" -ForegroundColor White
Write-Host "  • Temporary files removed" -ForegroundColor White
Write-Host ""
Write-Host "🎯 Next steps:" -ForegroundColor Yellow
Write-Host "  1. Review archived files and delete if no longer needed" -ForegroundColor White
Write-Host "  2. Commit the updated .gitignore file" -ForegroundColor White
Write-Host "  3. Consider removing CLEANUP_RECOMMENDATIONS.md after review" -ForegroundColor White
Write-Host ""
Write-Host "🚀 Your project is now much cleaner and better organized!" -ForegroundColor Green 