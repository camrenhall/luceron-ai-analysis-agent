"""
FastAPI application creation and configuration.
"""

from fastapi import FastAPI

from core import lifespan
from api.routes import health, workflows, chat


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="Document Analysis Agent",
        description="AI-powered document intelligence for Family Law financial discovery",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Include health check route at root
    app.include_router(health.router, tags=["health"])
    
    # Include workflow routes
    app.include_router(workflows.router, prefix="/workflows", tags=["workflows"])
    
    # Include chat routes
    app.include_router(chat.router, tags=["chat"])
    
    return app