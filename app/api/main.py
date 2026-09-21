"""FastAPI Application Entrypoint for AI Job Application Agent.

Provides base health checks, root status endpoints, and application lifecycle management.
"""

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict

import uvicorn
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse

from app.api.routers.agents import router as agents_router
from app.api.routers.automation import router as automation_router
from app.api.routers.candidate_answers import router as candidate_answers_router
from app.api.routers.glassdoor_automation import router as glassdoor_automation_router
from app.api.routers.jobs import router as jobs_router
from app.api.routers.linkedin import router as linkedin_router
from app.api.routers.llm import router as llm_router
from app.api.routers.matches import router as matches_router
from app.api.routers.profile import router as profile_router
from app.config.settings import Settings, get_settings

# Configure application logger
settings: Settings = get_settings()
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("ai_job_agent")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and shutdown events."""
    logger.info(
        "Starting AI Job Application Agent API (Environment: %s, Port: %s)",
        settings.APP_ENV,
        settings.APP_PORT,
    )
    yield
    logger.info("Shutting down AI Job Application Agent API")


app = FastAPI(
    title="AI Job Application Agent API",
    description="Backend API for AI Job Application Agent - Phase 6.2 LinkedIn Outreach Agent",
    version="0.6.2",
    lifespan=lifespan,
)

# Include Routers
app.include_router(jobs_router)
app.include_router(profile_router)
app.include_router(matches_router)
app.include_router(llm_router)
app.include_router(automation_router)
app.include_router(glassdoor_automation_router)
app.include_router(candidate_answers_router)
app.include_router(agents_router)
app.include_router(linkedin_router)



@app.get(
    "/",
    tags=["Root"],
    summary="Root service identifier",
    response_description="Basic application metadata",
)
async def root() -> Dict[str, Any]:
    """Root endpoint to identify the service and current release phase."""
    return {
        "message": "AI Job Application Agent API is running",
        "phase": "Phase 5.1 - Indeed Playwright Browser Foundation",
        "environment": settings.APP_ENV,
        "docs_url": "/docs",
    }


@app.get(
    "/health",
    tags=["Health"],
    summary="API Health Check",
    response_description="Service status and runtime environment",
    status_code=status.HTTP_200_OK,
)
async def health_check() -> Dict[str, str]:
    """Health check endpoint to verify service availability.

    Returns:
        JSON response with status and active environment name.
    """
    return {
        "status": "ok",
        "environment": settings.APP_ENV,
    }


if __name__ == "__main__":
    uvicorn.run(
        "app.api.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=(settings.APP_ENV == "development"),
    )
