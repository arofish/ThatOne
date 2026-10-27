
from __future__ import annotations

import os
import secrets

from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

from fastapi import (
    Depends,
    FastAPI,
    Header,
    Request,
)
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response

from pydantic import BaseModel, ConfigDict, Field

from starlette.exceptions import (
    HTTPException as StarletteHTTPException,
)

from photoindex.search_service import (
    PreviewChangedError,
    SearchService,
    SearchValidationError,
    ThumbnailNotFoundError,
    ThumbnailReadError,
)


# ---------- 请求与响应数据结构 ----------

class SearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(
        strict=True,
        min_length=1,
        max_length=1000,
    )


class SearchResultOut(BaseModel):
    asset_id: int
    score: float = Field(allow_inf_nan=False)
    relative_path: str
    original_path: str

    original_status: Literal[
        "available",
        "missing",
        "unreadable",
    ]

    thumbnail_url: str


class SearchResponseOut(BaseModel):
    query: str
    run_id: str

    results: list[SearchResultOut] = Field(
        max_length=5,
    )


# ---------- API 业务错误 ----------

class APIError(Exception):
    def __init__(
        self,
        status_code: int,
        code: str,
        message: str,
    ):
        self.status_code = status_code
        self.code = code
        self.message = message


def error_response(
    status: int,
    code: str,
    message: str,
) -> JSONResponse:

    headers = {
        "Cache-Control": "no-store",
    }

    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"

    return JSONResponse(
        status_code=status,
        content={
            "error": {
                "code": code,
                "message": message,
            }
        },
        headers=headers,
    )


# ---------- 应用工厂 ----------

def create_app(
    *,
    service: SearchService | None = None,
    token: str | None = None,
    project_root: Path | None = None,
) -> FastAPI:

    if token is None:
        token = os.environ.get("THATONE_API_TOKEN")

    if (
        not isinstance(token, str)
        or len(token) < 32
        or any(c.isspace() for c in token)
    ):
        raise RuntimeError(
            "THATONE_API_TOKEN must be a fresh random token "
            "of at least 32 non-whitespace characters"
        )

    root = (
        Path(project_root).resolve()
        if project_root is not None
        else Path(__file__).resolve().parents[1]
    )

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = service

        if active is None:
            active = await run_in_threadpool(
                SearchService.from_project,
                root,
            )

        app.state.search_service = active

        try:
            yield
        finally:
            app.state.search_service = None

    app = FastAPI(
        title="ThatOne local search",
        docs_url=None,
        redoc_url=None,
        openapi_url=None,
        lifespan=lifespan,
        debug=False,
    )

    # ---------- Bearer Token 认证 ----------

    def authorize(
        authorization: Annotated[
            str | None,
            Header(),
        ] = None,
    ) -> None:

        raw = authorization or ""
        scheme, separator, supplied = raw.partition(" ")

        if (
            not separator
            or scheme.lower() != "bearer"
            or not secrets.compare_digest(supplied, token)
        ):
            raise APIError(
                401,
                "UNAUTHORIZED",
                "Missing or invalid bearer token",
            )

    def get_service(request: Request) -> SearchService:

        active = getattr(
            request.app.state,
            "search_service",
            None,
        )

        if active is None:
            raise APIError(
                503,
                "SERVICE_UNAVAILABLE",
                "Search service not ready",
            )

        return active

    # ---------- 统一异常处理 ----------

    @app.exception_handler(APIError)
    async def api_error_handler(
        request: Request,
        exc: APIError,
    ):
        return error_response(
            exc.status_code,
            exc.code,
            exc.message,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(
        request: Request,
        exc: RequestValidationError,
    ):
        # 不把请求体或异常中的私人文本返回给客户端。
        return error_response(
            422,
            "INVALID_REQUEST",
            "Invalid request structure",
        )

    @app.exception_handler(StarletteHTTPException)
    async def http_error_handler(
        request: Request,
        exc: StarletteHTTPException,
    ):
        if exc.status_code == 404:
            return error_response(
                404,
                "NOT_FOUND",
                "Endpoint not found",
            )

        if exc.status_code == 405:
            return error_response(
                405,
                "METHOD_NOT_ALLOWED",
                "Method not allowed",
            )

        return error_response(
            exc.status_code,
            "HTTP_ERROR",
            "HTTP request failed",
        )

    @app.exception_handler(Exception)
    async def unexpected_error_handler(
        request: Request,
        exc: Exception,
    ):
        return error_response(
            500,
            "INTERNAL_ERROR",
            "Unexpected server error",
        )

    # ---------- GET /health ----------

    @app.get(
        "/health",
        dependencies=[Depends(authorize)],
    )
    def health(request: Request):

        active = get_service(request)

        return {
            "status": "ready",
            "run_id": active.run_id,
            "asset_count": len(active.asset_ids),
        }

    # ---------- POST /search ----------

    @app.post(
        "/search",
        response_model=SearchResponseOut,
        dependencies=[Depends(authorize)],
    )
    def search(
        payload: SearchRequest,
        request: Request,
    ):

        active = get_service(request)

        try:
            result = active.search(payload.query)

        except SearchValidationError as exc:
            raise APIError(
                422,
                "INVALID_QUERY",
                "Query is empty or exceeds model limits",
            ) from exc

        return asdict(result)

    # ---------- GET /assets/{id}/thumbnail ----------

    @app.get(
        "/assets/{asset_id}/thumbnail",
        dependencies=[Depends(authorize)],
    )
    def thumbnail(
        asset_id: int,
        request: Request,
    ):

        active = get_service(request)

        try:
            png_bytes = active.thumbnail_bytes(asset_id)

        except ThumbnailNotFoundError as exc:
            raise APIError(
                404,
                "THUMBNAIL_NOT_FOUND",
                "Thumbnail not found",
            ) from exc

        except PreviewChangedError as exc:
            raise APIError(
                409,
                "PREVIEW_CHANGED",
                "Thumbnail no longer matches the verified cache",
            ) from exc

        except ThumbnailReadError as exc:
            raise APIError(
                500,
                "THUMBNAIL_READ_FAILED",
                "Thumbnail could not be read",
            ) from exc

        return Response(
            content=png_bytes,
            media_type="image/png",
            headers={
                "Cache-Control": "no-store",
                "X-Content-Type-Options": "nosniff",
            },
        )

    return app


# ---------- 本地开发入口 ----------

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        create_app(),
        host="127.0.0.1",
        port=8765,
        workers=1,
        reload=False,
        access_log=False,
        proxy_headers=False,
        server_header=False,
    )
