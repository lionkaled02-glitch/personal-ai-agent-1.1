$ErrorActionPreference = "Stop"

Write-Host "Personal AI Agent - Windows setup"
if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "Python launcher 'py' was not found. Install Python 3.11+ first."
}

py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt

Write-Host "Core installation complete."
Write-Host "Default mode is offline/mock and needs no API key."
Write-Host "For OpenAI: install packages/agent-core[openai] and set OPENAI_API_KEY."
Write-Host "For browser automation: install [browser-playwright], then run 'playwright install chromium'."
Write-Host "For Windows desktop control: install packages/agent-core[computer-windows]."
Write-Host "FFmpeg is optional and must be installed separately if video rendering is enabled."
