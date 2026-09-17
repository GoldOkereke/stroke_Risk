"""Uvicorn entrypoint.

This file re-exports the FastAPI app defined in `app/main.py` so you can run:
`uvicorn main:app --reload --port 8000`
"""

from app.main import app  # noqa: F401
