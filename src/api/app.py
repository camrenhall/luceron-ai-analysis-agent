"""
FastAPI application creation and configuration.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from core import lifespan
from api.routes import health, chat


def create_app() -> FastAPI:
    """Create and configure the FastAPI application"""
    
    app = FastAPI(
        title="Document Analysis Agent",
        description="AI-powered document intelligence for Family Law financial discovery",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://simple-s3-upload.onrender.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Include health check route at root
    app.include_router(health.router, tags=["health"])
    
    # Include chat routes
    app.include_router(chat.router, tags=["chat"])
    
    return app