# Frame API (FastAPI)

This service exposes correlation heatmap frames on demand so the frontend can
stream data instead of embedding gigabytes of HTML.

## Setup (Ubuntu / WSL)

```bash
cd /mnt/d/Downloads/QRM
python3 -m venv .venv_wsl
source .venv_wsl/bin/activate
pip install -r recession_project/server/requirements.txt
```

## Run the API

```bash
cd /mnt/d/Downloads/QRM/recession_project/server
uvicorn frame_api:app --host 0.0.0.0 --port 8001 --reload
```

The service exposes:

- `GET /` — health & dataset summary
- `GET /meta` — dates, spreads, counts, dtype
- `GET /frame/{idx}` — single frame as `float32` binary (row-major)
- `GET /frames?start=<s>&end=<e>` — batch frames (`start <= idx < end`)

To stop the server press `Ctrl+C`.
