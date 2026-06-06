Set-Location -LiteralPath $PSScriptRoot
Write-Host "Starting Turkish Legal RAG demo..."
Write-Host "Open http://127.0.0.1:7860 in your browser."
Write-Host "Keep this PowerShell window open while using the demo."
python -u scripts\demo_app.py --data-dir data --port 7860 --no-browser
