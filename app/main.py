"""Application entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes_assets import router as assets_router
from app.api.routes_edit import router as edit_router
from app.api.routes_feedback import router as feedback_router
from app.api.routes_jobs import router as jobs_router
from app.api.routes_meta import router as meta_router
from app.api.routes_review import router as review_router
from app.api.schemas import HealthResponse


def create_app() -> FastAPI:
    """Create the FastAPI application."""

    app = FastAPI(
        title="PsAgent API",
        version="0.1.0",
        description="API layer for the photo-editing agent.",
    )

    # 先放开本地前端开发常用来源，方便 Vue/Vite 联调上传和流式请求。
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://127.0.0.1:5173",
            "http://localhost:5173",
            "http://127.0.0.1:4173",
            "http://localhost:4173",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["system"])
    async def health() -> HealthResponse:
        """Simple health check."""

        return HealthResponse()

    app.include_router(assets_router)
    app.include_router(edit_router)
    app.include_router(jobs_router)
    app.include_router(feedback_router)
    app.include_router(review_router)
    app.include_router(meta_router)
    return app


app = create_app()
