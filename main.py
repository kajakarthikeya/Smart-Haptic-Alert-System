"""
Main Application Bootstrap & FastAPI Server Entrypoint.

Serves:
1. REST API endpoints under /api/v1
2. Interactive Frontend Dashboard under / and /dashboard
3. Static frontend assets under /static
"""

import argparse
from pathlib import Path
import sys
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.context.context_manager import ContextManager, EnvironmentMode
from app.services.integration_service import integration_service
from app.utils.logger import get_logger, setup_logger
from config import Config, settings

# Setup application logger
logger = setup_logger("SmartHapticAlertSystem")

# Initialize FastAPI application
app = FastAPI(
    title=settings.system.app_name,
    description="Software-Only Integration and Verification Prototype for Phases 1–8",
    version="0.8.0",
)

# Enable CORS for local development flexibility
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API routes
app.include_router(api_router)

# Mount static web frontend files
WEB_DIR = Config.paths.base_dir / "app" / "web"
if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")


@app.get("/", include_in_schema=False)
@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    """Serves the main single-page frontend prototype."""
    index_file = WEB_DIR / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {"message": "Smart Haptic Alert System API is running. Web assets not found."}


def print_banner() -> None:
    """Prints application startup banner."""
    banner = f"""
    ======================================================================
       {settings.system.app_name} v0.8.0
       Architecture: Clean Architecture (Phases 1–8 Software Integration)
       AI Classifier: 2D CNN (sound_classifier_best.keras)
       Target Sounds: Ambulance, Car Horn, Fire Alarm, Doorbell, Dog Bark
       Operating Modes: Home, Road, Office
       Hardware: Not Required (Software-Only Prototype)
       Dashboard URL: http://{settings.api.host}:{settings.api.port}/
       Swagger Docs : http://{settings.api.host}:{settings.api.port}/docs
    ======================================================================
    """
    print(banner)


def main() -> None:
    """Entrypoint function starting uvicorn server."""
    parser = argparse.ArgumentParser(description="Smart Haptic Alert System Server")
    parser.add_argument("--host", type=str, default=settings.api.host, help="Bind host")
    parser.add_argument("--port", type=int, default=settings.api.port, help="Bind port")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    print_banner()
    logger.info("Starting software integration server on http://%s:%d", args.host, args.port)
    try:
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            log_level="info",
        )
    except KeyboardInterrupt:
        logger.info("Shutdown signal received. Exiting gracefully...")
        sys.exit(0)
    except Exception as e:
        logger.critical("Unhandled exception during server run: %s", e, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
