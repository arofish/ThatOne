
"""v1.0.0 local API; use instead of frozen M3 API after release tests pass."""
from __future__ import annotations

import os
import secrets
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from photoindex.user_library import UserLibraryService, LibraryBusyError
from photoindex.search_service import (
    SearchValidationError, ThumbnailNotFoundError,
    PreviewChangedError, ThumbnailReadError,
)

class SearchPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(strict=True, min_length=1, max_length=1000)

class ImportPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")
    directory: str = Field(strict=True, min_length=1, max_length=4096)

class SearchResultOut(BaseModel):
    asset_id: int
    score: float = Field(allow_inf_nan=False)
    relative_path: str
    original_path: str
    original_status: Literal["available", "missing", "unreadable"]
    thumbnail_url: str

class SearchResponseOut(BaseModel):
    query: str
    run_id: str
    results: list[SearchResultOut] = Field(max_length=5)

class APIError(Exception):
    def __init__(self, status: int, code: str, message: str):
        self.status, self.code, self.message = status, code, message

def error(status: int, code: str, message: str):
    headers = {"Cache-Control": "no-store"}
    if status == 401:
        headers["WWW-Authenticate"] = "Bearer"
    return JSONResponse(status_code=status,
        content={"error": {"code": code, "message": message}}, headers=headers)

def create_app(*, service: UserLibraryService | None = None,
               token: str | None = None,
               model_dir: Path | None = None,
               data_root: Path | None = None) -> FastAPI:
    token = token or os.environ.get("THATONE_API_TOKEN")
    if not isinstance(token, str) or len(token) < 32 or any(ch.isspace() for ch in token):
        raise RuntimeError("Missing secure local token")
    model_dir = Path(model_dir or os.environ.get("THATONE_MODEL_DIR", ""))
    data_root = Path(data_root or os.environ.get("THATONE_DATA_ROOT", ""))
    if service is None and (str(model_dir)=="." or str(data_root)=="."):
        raise RuntimeError("Required release paths missing")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = service or await run_in_threadpool(
            UserLibraryService.from_release,
            model_dir=model_dir, data_root=data_root,
        )
        app.state.library_service = active
        try:
            yield
        finally:
            app.state.library_service = None
            if service is None:
                await run_in_threadpool(active.shutdown)

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None,
                  openapi_url=None, debug=False)

    def authorize(authorization: Annotated[str | None, Header()] = None):
        scheme, sep, supplied = (authorization or "").partition(" ")
        if not sep or scheme.lower() != "bearer" or not secrets.compare_digest(supplied, token):
            raise APIError(401, "UNAUTHORIZED", "Missing or invalid bearer token")

    def get_service(request: Request) -> UserLibraryService:
        active = getattr(request.app.state, "library_service", None)
        if active is None:
            raise APIError(503,"NOT_READY","Service not ready")
        return active

    @app.exception_handler(APIError)
    async def handle_api(request: Request, exc: APIError):
        return error(exc.status, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def handle_validation(request: Request, exc: RequestValidationError):
        return error(422,"INVALID_REQUEST","Invalid request structure")

    @app.exception_handler(StarletteHTTPException)
    async def handle_http(request: Request, exc: StarletteHTTPException):
        return error(exc.status_code,"HTTP_ERROR","Request failed")

    @app.exception_handler(Exception)
    async def handle_other(request: Request, exc: Exception):
        return error(500,"INTERNAL_ERROR","Unexpected server error")

    @app.get("/health", dependencies=[Depends(authorize)])
    def health(request: Request):
        active = get_service(request)
        return {"status":"ready", "run_id":active.run_id,
                "asset_count":len(active.asset_ids)}

    @app.get("/library/status", dependencies=[Depends(authorize)])
    def status(request: Request):
        return get_service(request).library_status()

    @app.post("/library/import", status_code=202,
              dependencies=[Depends(authorize)])
    def import_directory(payload: ImportPayload, request: Request):
        try:
            job_id = get_service(request).start_import(Path(payload.directory))
        except LibraryBusyError as exc:
            raise APIError(409,"INDEX_BUSY","Another scan is already running") from exc
        except (OSError, ValueError) as exc:
            raise APIError(422,"INVALID_DIRECTORY","Directory cannot be imported") from exc
        return {"job_id": job_id, "status": "queued"}

    @app.get("/library/jobs/{job_id}", dependencies=[Depends(authorize)])
    def job(job_id: str, request: Request):
        result = get_service(request).job_status(job_id)
        if result is None:
            raise APIError(404,"JOB_NOT_FOUND","Unknown import job")
        return result

    @app.post("/search", response_model=SearchResponseOut,
              dependencies=[Depends(authorize)])
    def search(payload: SearchPayload, request: Request):
        try:
            return asdict(get_service(request).search(payload.query))
        except SearchValidationError as exc:
            raise APIError(422,"INVALID_QUERY","Query is blank or exceeds model limits") from exc

    @app.get("/assets/{asset_id}/thumbnail", dependencies=[Depends(authorize)])
    def thumbnail(asset_id: int, request: Request):
        try:
            data = get_service(request).thumbnail_bytes(asset_id)
        except ThumbnailNotFoundError as exc:
            raise APIError(404,"THUMBNAIL_NOT_FOUND","Thumbnail not found") from exc
        except PreviewChangedError as exc:
            raise APIError(409,"PREVIEW_CHANGED","Thumbnail changed after indexing") from exc
        except ThumbnailReadError as exc:
            raise APIError(500,"THUMBNAIL_READ_FAILED","Cannot read thumbnail") from exc
        return Response(content=data,media_type="image/png",headers={
            "Cache-Control":"no-store", "X-Content-Type-Options":"nosniff"
        })
    return app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(create_app(),host="127.0.0.1",port=8765,workers=1,
                reload=False,access_log=False,proxy_headers=False)
