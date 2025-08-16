"""
Document Analysis Agent - Entry Point
AI-powered document intelligence orchestrator for Family Law financial discovery
"""

import logging
import uvicorn

from src import create_app, settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = create_app()

if __name__ == "__main__":
    logger.info(f"Starting Document Analysis Agent on port {settings.PORT}")
    uvicorn.run(app, host="0.0.0.0", port=settings.PORT, log_level="info")