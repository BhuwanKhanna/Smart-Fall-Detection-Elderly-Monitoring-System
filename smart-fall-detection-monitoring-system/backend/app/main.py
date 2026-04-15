from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.routes import router
from backend.app.core.config import settings
from backend.app.services.monitor import MonitoringService
from backend.app.services.storage import EventStore


app = FastAPI(
    title="Smart Fall Detection API",
    description="Real-time fall detection and elderly monitoring backend.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

store = EventStore(settings.database_path)
store.initialize()
app.state.monitoring = MonitoringService(settings=settings, store=store)
app.include_router(router, prefix=settings.api_prefix)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "name": settings.app_name,
        "docs": "/docs",
        "health": f"{settings.api_prefix}/health",
    }
