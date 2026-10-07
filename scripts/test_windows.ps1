$ErrorActionPreference = "Stop"
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    throw "Virtual environment not found. Run scripts/install_windows.ps1 first."
}
$env:PYTHONPATH = "packages/agent-core/src;packages/creation-agent/src;apps/backend/src"
& .\.venv\Scripts\python.exe -m pytest -q
