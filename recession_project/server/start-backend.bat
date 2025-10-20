@echo off
wsl -e bash -c "cd /mnt/d/Downloads/QRM/recession_project/server && .venv/bin/python -m uvicorn frame_api:app --host 0.0.0.0 --port 8001 --reload"
