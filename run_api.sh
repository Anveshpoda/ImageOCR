#!/bin/bash

# Activate the virtual environment
source venv/bin/activate

# Run the API
cd api
uvicorn api:app --reload --host 0.0.0.0 --port 8000
