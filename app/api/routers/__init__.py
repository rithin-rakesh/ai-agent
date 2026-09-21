"""API Routers package."""

from app.api.routers.automation import router as automation_router
from app.api.routers.jobs import router as jobs_router
from app.api.routers.llm import router as llm_router
from app.api.routers.matches import router as matches_router
from app.api.routers.profile import router as profile_router

__all__ = ["jobs_router", "profile_router", "matches_router", "llm_router", "automation_router"]
