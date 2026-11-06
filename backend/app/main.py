"""The web app: API under /api, the page from /frontend.

    uvicorn app.main:app --app-dir backend --port 8000
"""

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import Settings, get_settings
from app.core.logging import setup_logging
from app.services.llm import get_llm
from app.services.web import DemoError, DemoService

FRONTEND = Path(__file__).resolve().parents[2] / "frontend"
SECURITY_HEADERS = {
    "Content-Security-Policy": "default-src 'self'; img-src 'self' data:; style-src 'self'; script-src 'self'; "
                               "frame-ancestors 'none'",
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}


def create_app(settings: Settings | None = None, service: DemoService | None = None) -> FastAPI:
    settings = settings or get_settings()
    setup_logging(settings.log_level)
    app = FastAPI(title="SkillBridge", docs_url="/docs" if settings.enable_docs else None,
                  redoc_url=None, openapi_url="/openapi.json" if settings.enable_docs else None)
    app.state.settings = settings
    app.state.service = service or DemoService(settings, get_llm)

    @app.exception_handler(DemoError)
    def demo_error(_: Request, error: DemoError) -> JSONResponse:
        return JSONResponse({"detail": error.message}, status_code=error.status)

    @app.middleware("http")
    async def security_headers(request: Request, call_next):
        response = await call_next(request)
        response.headers.update(SECURITY_HEADERS)
        return response

    app.include_router(router)
    if FRONTEND.exists():
        app.mount("/", StaticFiles(directory=FRONTEND, html=True), name="frontend")
    return app


app = create_app()
