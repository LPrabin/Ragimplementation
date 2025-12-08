#!/bin/bash

python -m streamlit run frontend/app.py --server.address=0.0.0.0 & python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000