#!/bin/bash
# start FastAPI server on the port provided by the platform
uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
