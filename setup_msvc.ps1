# setup_msvc.ps1
# Sets up MSVC environment for PyTorch CUDA extensions

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "Setting up MSVC + Python Environment" -ForegroundColor Cyan
Write-Host "========================================`n" -ForegroundColor Cyan

# Find Visual Studio
$vsPaths = @(
    "C:\Program Files\Microsoft Visual Studio\2022\Community",
    "C:\Program Files\Microsoft Visual Studio\2022\Professional",
    "C:\Program Files\Microsoft Visual Studio\2022\Enterprise",
    "C:\Program Files\Microsoft Visual Studio\2022\BuildTools",
    "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools"
)

$vcvarsall = $null
foreach ($path in $vsPaths) {
    $testPath = Join-Path $path "VC\Auxiliary\Build\vcvarsall.bat"
    if (Test-Path $testPath) {
        $vcvarsall = $testPath
        Write-Host "✅ Found Visual Studio at:" -ForegroundColor Green
        Write-Host "   $path`n" -ForegroundColor Gray
        break
    }
}

if (-not $vcvarsall) {
    Write-Host "❌ Visual Studio 2022 not found!`n" -ForegroundColor Red
    Write-Host "Please install Visual Studio 2022 Build Tools:" -ForegroundColor Yellow
    Write-Host "https://visualstudio.microsoft.com/downloads/`n" -ForegroundColor Yellow
    exit 1
}

# Initialize MSVC
Write-Host "Initializing MSVC x64 environment..." -ForegroundColor Cyan
cmd /c "`"$vcvarsall`" x64 >nul 2>&1 & set" | ForEach-Object {
    if ($_ -match "^(.*?)=(.*)$") {
        Set-Item -Path "env:$($matches[1])" -Value $matches[2] -Force
    }
}

# Verify
$clPath = Get-Command cl -ErrorAction SilentlyContinue
if ($clPath) {
    Write-Host "✅ MSVC compiler available`n" -ForegroundColor Green
} else {
    Write-Host "⚠️  cl.exe not found - extension compilation may fail`n" -ForegroundColor Yellow
}

# Activate venv
$venvPath = "D:\GAN\venv\Scripts\Activate.ps1"
if (Test-Path $venvPath) {
    Write-Host "Activating virtual environment..." -ForegroundColor Cyan
    & $venvPath
    Write-Host "✅ Virtual environment activated`n" -ForegroundColor Green
    
    # Show Python info
    Write-Host "Python:" (python --version) -ForegroundColor Gray
    Write-Host ""
} else {
    Write-Host "⚠️  venv not found at: $venvPath`n" -ForegroundColor Yellow
}

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "✅ Environment Ready!" -ForegroundColor Green
Write-Host "========================================`n" -ForegroundColor Cyan