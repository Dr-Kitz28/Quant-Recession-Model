# start-dev.ps1 — run from Windows PowerShell (repo root)
# Launches backend in WSL (no interactive attach) then starts the frontend in PowerShell

# Adjust the path below if your checkout is not at D:\Downloads\QRM
wsl -e bash -lc "cd /mnt/d/Downloads/QRM/recession_project/server && . .venv/bin/activate && nohup python -m uvicorn frame_api:app --host 127.0.0.1 --port 8001 --log-level info > uvicorn.log 2>&1 & echo $! > /tmp/uvicorn.pid"

Push-Location D:\Downloads\QRM\recession_project\frontend\heatmap-client
npm install
npm run dev
Pop-Location
