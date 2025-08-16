"""
Application lifecycle management.
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI

from services import http_client_service


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle"""
    # Startup
    await http_client_service.init_client()
    yield
    # Shutdown
    await http_client_service.close_client()