@echo off
cd /d "%~dp0"
echo Starting Turkish Legal RAG demo...
echo Open http://127.0.0.1:7860 in your browser.
echo Keep this window open while using the demo.
python -u scripts\demo_app.py --data-dir data --port 7860 --no-browser
pause

