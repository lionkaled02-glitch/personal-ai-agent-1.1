$ErrorActionPreference = "Stop"
if (-not (Test-Path .\.venv\Scripts\python.exe)) {
    throw "Virtual environment not found. Run scripts/install_windows.ps1 first."
}
& .\.venv\Scripts\python.exe -m uvicorn apps.backend.src.api:app --host 127.0.0.1 --port 8000
