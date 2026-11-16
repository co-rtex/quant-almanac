"""FastAPI application factory."""

from collections.abc import Awaitable, Callable

from fastapi import FastAPI, Request, Response
from fastapi.staticfiles import StaticFiles

from quant_signal.api.routes.console import STATIC_DIR
from quant_signal.api.routes.console import router as console_router
from quant_signal.api.routes.governance import router as governance_router
from quant_signal.api.routes.health import router as health_router
from quant_signal.api.routes.models import router as model_router
from quant_signal.api.routes.signals import router as signal_router
from quant_signal.core.config import get_settings
from quant_signal.core.logging import configure_logging

CONSOLE_SECURITY_HEADERS = {
    # The console is read-only, script-free, and loads nothing from third parties.
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'self'; img-src 'self' data:; "
        "base-uri 'none'; form-action 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
}


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    settings = get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Quant Almanac",
        version="0.2.0",
        description=(
            "Read-only API for a governed equity forecasting pipeline: signals, model "
            "metadata, data-quality evidence, lineage, and artifact verification."
        ),
    )

    @app.middleware("http")
    async def console_security_headers(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        response = await call_next(request)
        if request.url.path.startswith("/console"):
            for header, value in CONSOLE_SECURITY_HEADERS.items():
                response.headers.setdefault(header, value)
        return response

    app.include_router(health_router)
    app.include_router(signal_router)
    app.include_router(model_router)
    app.include_router(governance_router)
    app.include_router(console_router)
    app.mount("/console/static", StaticFiles(directory=str(STATIC_DIR)), name="console-static")
    return app


app = create_app()


def run() -> None:
    """Console entry point placeholder."""

    return None
