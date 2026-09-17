from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .core.config import settings
from .core.logging import logger
from .routers import assessment, auth
from .api.routes import neurological
from .api.routes import signal      # FR1 — signal processing
from .api.routes import fusion      # FR5 — fusion engine
from .api.routes import neuro_tests # FR4 — active neurological tests

# ── 1. Create app first ────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
    description="Multimodal Pre-TIA Screening System",
)

# ── 2. Middleware ──────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def log_requests(request, call_next):
    logger.info(f"Incoming request: {request.method} {request.url}")
    response = await call_next(request)
    logger.info(f"Response status: {response.status_code}")
    return response

# ── 3. Routers (only after app is created) ────────────────────────────────
app.include_router(assessment.router,       prefix="/api/assessment",    tags=["assessment"])
app.include_router(auth.router,             prefix="/api/auth",          tags=["auth"])
app.include_router(neurological.router,     prefix="/api/neurological",  tags=["neurological"])
app.include_router(signal.router,           prefix="/api/signal",        tags=["signal"])       # FR1
app.include_router(fusion.router,           prefix="/api/fusion",        tags=["fusion"])       # FR5
app.include_router(neuro_tests.router,      prefix="/api/neuro-tests",   tags=["neuro-tests"])  # FR4

# ── 4. Base endpoints ─────────────────────────────────────────────────────
@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}

@app.get("/", tags=["root"])
async def root():
    return {
        "message": f"Welcome to the {settings.APP_NAME} API!",
        "version": settings.APP_VERSION,
    }