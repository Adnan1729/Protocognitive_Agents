# One-time environment setup. Run from the repository root:
#     .\setup.ps1

$ErrorActionPreference = "Stop"

$pythonCmd = "py -3.11"

try {
    & py -3.11 --version | Out-Null
} catch {
    Write-Host "Error: Python 3.11 not found." -ForegroundColor Red
    Write-Host "TensorFlow requires Python 3.9-3.12. Install one from python.org,"
    Write-Host "or edit this script to use a different version (e.g. py -3.10)."
    exit 1
}

Write-Host "Creating virtual environment in .venv\ using Python 3.11..."
py -3.11 -m venv .venv

Write-Host "Activating environment..."
& .\.venv\Scripts\Activate.ps1

Write-Host "Upgrading pip..."
python -m pip install --upgrade pip

Write-Host "Installing proteinoid_spikes (editable) and dependencies..."
pip install -e .

Write-Host ""
Write-Host "Setup complete. Activate the environment with:" -ForegroundColor Green
Write-Host "    .\.venv\Scripts\Activate.ps1"