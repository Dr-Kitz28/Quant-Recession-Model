# Correlation Heatmap Web Application - Setup Summary

## Issues Fixed

### 1. **Circular Dependency in React Hooks**
- **Problem**: `loadAllFrames` was defined before `ensureRange`, causing a reference error
- **Fix**: Reordered the useCallback hooks to define `ensureRange` before `loadAllFrames`
- **File**: `src/App.jsx`

### 2. **API Base URL Configuration**
- **Problem**: Frontend was trying to connect to `http://localhost:8001` directly, but Vite proxy was configured
- **Fix**: Set `VITE_API_BASE` to empty string to use Vite's proxy configuration
- **File**: `.env`

### 3. **Vite Proxy Configuration**
- **Status**: Already properly configured to proxy API calls to `http://localhost:8001`
- **File**: `vite.config.js`

### 4. **CORS Configuration**
- **Status**: Backend already has proper CORS middleware configured with `allow_origins=["*"]`
- **File**: `server/frame_api.py`

### 5. **Development Server Startup**
- **Problem**: PowerShell command simplification was causing directory navigation issues
- **Fix**: Created `start-dev.bat` batch file for reliable server startup
- **File**: `start-dev.bat`

## Current Architecture

```
┌─────────────────────────────────────────────────┐
│           Browser (localhost:5173)              │
│                                                 │
│  ┌─────────────────────────────────────────┐   │
│  │  React Application (App.jsx)            │   │
│  │  - Fetches metadata from API            │   │
│  │  - Streams correlation frames           │   │
│  │  - Renders Plotly heatmap               │   │
│  │  - Caches frames in IndexedDB           │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
                      ↓
              Vite Proxy (Dev Server)
                      ↓
┌─────────────────────────────────────────────────┐
│         FastAPI Backend (localhost:8001)        │
│              (Running in WSL)                   │
│                                                 │
│  Endpoints:                                     │
│  - GET /meta      → Metadata & dates            │
│  - GET /frames    → Batch frame data            │
│  - GET /tiles     → Progressive load tiles      │
│                                                 │
│  Data: correlation_tensor_usa.npz               │
│  Size: 10,190 dates × 55 spreads                │
└─────────────────────────────────────────────────┘
```

## Running the Application

### Start Backend (FastAPI)
Already running via WSL:
```bash
wsl -e bash -lc "cd /mnt/d/Downloads/QRM/recession_project/server && .venv/bin/uvicorn frame_api:app --host 0.0.0.0 --port 8001 --reload"
```

### Start Frontend (Vite + React)
```cmd
d:\Downloads\QRM\recession_project\frontend\heatmap-client\start-dev.bat
```

Or manually:
```cmd
cd d:\Downloads\QRM\recession_project\frontend\heatmap-client
npm run dev
```

### Access Application
Open browser: **http://localhost:5173/**

## Key Files Modified

1. **`.env`** - Set API base to empty for proxy usage
2. **`src/App.jsx`** - Fixed hook dependency order, added debug logging
3. **`src/main.jsx`** - Added environment variable logging
4. **`start-dev.bat`** - Created for reliable dev server startup

## Features

- **Interactive heatmap** showing correlation between bond spreads over time
- **Timeline scrubbing** with 10,190 frames
- **Play/pause animation** with adjustable speed
- **Bulk frame loading** with progress indicator
- **Progressive loading** for gradual refinement
- **IndexedDB caching** for offline persistence
- **Lazy loading** of Plotly library for faster initial load

## Troubleshooting

If the page is blank:

1. **Check browser console** (F12 → Console)
   - Should see: "Using API Base: " (empty string)
   - Should see: "Loading metadata from: "
   - Should see metadata loaded successfully

2. **Verify backend is running**:
   ```cmd
   wsl -e bash -lc 'curl -I http://localhost:8001/meta'
   ```
   Should return 200 OK

3. **Verify frontend is running**:
   - Check terminal shows: "ROLLDOWN-VITE v7.1.14  ready"
   - Open http://localhost:5173/

4. **Check network tab** (F12 → Network)
   - Should see successful requests to `/meta` and `/frames`

## Environment Variables

- `VITE_API_BASE` - API base URL (empty = use proxy, or specify full URL with CORS)

## Dependencies

Frontend:
- React 19.1.1
- Plotly.js-dist-min 3.1.1
- Vite 7.1.14 (rolldown-vite)

Backend:
- FastAPI
- NumPy
- Uvicorn

## Data Source

The application visualizes correlation data from:
- **File**: `outputs/correlation_tensor_usa.npz`
- **Dates**: 1985-01-02 to 2025-10-01 (10,190 days)
- **Spreads**: 55 different US Treasury yield spreads
- **Matrix**: 55×55 correlation matrix per date
