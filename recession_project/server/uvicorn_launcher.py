import uvicorn
import frame_api

if __name__ == '__main__':
    uvicorn.run(frame_api.app, host='127.0.0.1', port=8001, log_level='info')
