#!/usr/bin/env bash
# start-wsl.sh — run from WSL/bash (repo root)
# Starts backend (in the project's .venv) and then starts the frontend dev server
set -e

# Adjust path if your checkout is in a different mount
cd /mnt/d/Downloads/QRM/recession_project/server
source .venv/bin/activate
nohup python -m uvicorn frame_api:app --host 127.0.0.1 --port 8001 --log-level info > uvicorn.log 2>&1 &
echo $! > /tmp/uvicorn.pid

cd /mnt/d/Downloads/QRM/recession_project/frontend/heatmap-client
npm install
npm run dev
