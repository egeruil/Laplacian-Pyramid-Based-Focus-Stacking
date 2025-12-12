<#
PowerShell setup script for Windows.
Creates a virtual environment, installs dependencies and runs initialize.py
Run with: PowerShell -ExecutionPolicy Bypass -File .\setup.ps1
#>

Write-Host "Starting environment setup..."

# find python
$pyCmd = $null
if (Get-Command python -ErrorAction SilentlyContinue) { $pyCmd = "python" }
elseif (Get-Command python3 -ErrorAction SilentlyContinue) { $pyCmd = "python3" }
else {
    Write-Host "Error: Python is not found on PATH. Install Python 3.8+ and retry." -ForegroundColor Red
    exit 1
}

Write-Host "Using Python command: $pyCmd"

# Create virtual environment if missing
$venvPath = Join-Path $PWD ".venv"
if (-Not (Test-Path $venvPath)) {
    Write-Host "Creating virtual environment (.venv)..."
    & $pyCmd -m venv .venv
} else {
    Write-Host "Virtual environment (.venv) already exists."
}

# Path to venv python
$venvPython = Join-Path $venvPath "Scripts\python.exe"
if (-Not (Test-Path $venvPython)) {
    Write-Host "Error: Could not find virtual environment Python at $venvPython" -ForegroundColor Red
    exit 1
}

# Upgrade pip
Write-Host "Upgrading pip..."
& $venvPython -m pip install --upgrade pip

# Install requirements
$req = Join-Path $PWD "requirements.txt"
if (Test-Path $req) {
    Write-Host "Installing dependencies from requirements.txt..."
    & $venvPython -m pip install -r $req
} else {
    Write-Host "Warning: requirements.txt not found. Skipping dependency installation." -ForegroundColor Yellow
}

# Run initialization script
$init = Join-Path $PWD "initialize.py"
if (Test-Path $init) {
    Write-Host "Running initialization script (Download data & Precompute cache)..."
    & $venvPython $init
} else {
    Write-Host "Error: initialize.py not found." -ForegroundColor Red
    exit 1
}

Write-Host "----------------------------------------------------------------"
Write-Host "Setup complete!"
Write-Host "To activate the environment in your current PowerShell session, run:"
Write-Host "  . .\ .venv\Scripts\Activate.ps1" -ForegroundColor Cyan
Write-Host "Or start a new PowerShell with the environment's Python directly:"
Write-Host "  .\ .venv\Scripts\python.exe script.py" -ForegroundColor Cyan
Write-Host "----------------------------------------------------------------"
