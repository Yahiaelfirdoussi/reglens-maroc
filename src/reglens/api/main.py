"""HTTP API: question answering (plain and streamed), ingestion, health and metrics.

Run:  reglens serve   (or: uvicorn reglens.api.main:app)
"""

import json
import re
import time
import uuid
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

import structlog
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

from reglens import metrics
from reglens.api.security import KeyRing, RateLimiter, Role, key_id
from reglens.cache import cache_key
from reglens.config import Settings, get_settings
from reglens.generation.citations import Citation
from reglens.guardrails import MAX_QUESTION_CHARS
from reglens.ingestion.metadata import write_sidecar
from reglens.ingestion.ocr import OcrUnavailableError, ocr_document
from reglens.log import configure_logging
from reglens.models import DocumentMetadata, Issuer, Language
from reglens.rag import Answer
from reglens.service import Service, build_service

log = structlog.get_logger(__name__)

Status = Literal["answered", "not_found", "out_of_scope", "sources_only"]


# --- Schemas --------------------------------------------------------------------------------


class QueryIn(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS * 2)
    k: int | None = Field(default=None, ge=1, le=20)


class SourceOut(BaseModel):
    number: int
    cited: bool
    issuer: str
    reference: str | None
    title: str
    section: str | None
    page_start: int
    page_end: int
    url: str
    snippet: str
    score: float


class QueryOut(BaseModel):
    request_id: str
    status: Status
    answer: str | None
    citations: list[Citation]
    sources: list[SourceOut]
    guardrail: str
    abstention: str
    cached: bool
    retrieval_ms: float
    generation_ms: float | None


class IngestOut(BaseModel):
    file: str
    status: Literal["added", "updated", "unchanged"]
    chunks: int
    ocr: bool


def status_of(answer: Answer) -> Status:
    if answer.guardrail != "ok":
        return "out_of_scope"
    if answer.abstention != "none":
        return "not_found"
    return "answered" if answer.text is not None else "sources_only"


def to_output(answer: Answer, request_id: str, cached: bool) -> QueryOut:
    cited = {c.number for c in answer.citations}
    return QueryOut(
        request_id=request_id,
        status=status_of(answer),
        answer=answer.text,
        citations=answer.citations,
        sources=[
            SourceOut(
                number=n,
                cited=n in cited,
                issuer=s.chunk.issuer,
                reference=s.chunk.reference,
                title=s.chunk.title,
                section=s.chunk.section,
                page_start=s.chunk.page_start,
                page_end=s.chunk.page_end,
                url=s.chunk.url,
                snippet=s.chunk.text[:300],
                score=s.score,
            )
            for n, s in enumerate(answer.sources, start=1)
        ],
        guardrail=answer.guardrail,
        abstention=answer.abstention,
        cached=cached,
        retrieval_ms=answer.retrieval_ms,
        generation_ms=answer.generation_ms,
    )


def observe(answer: Answer, total_s: float) -> None:
    metrics.STAGE_SECONDS.labels("total").observe(total_s)
    if answer.guardrail != "ok":
        metrics.GUARDRAIL.labels(answer.guardrail).inc()
        return
    metrics.STAGE_SECONDS.labels("retrieval").observe(answer.retrieval_ms / 1000)
    if answer.generation_ms is not None:
        metrics.STAGE_SECONDS.labels("generation").observe(answer.generation_ms / 1000)
    if answer.abstention != "none":
        metrics.ABSTENTION.labels(answer.abstention).inc()
    metrics.LLM_TOKENS.labels("prompt").inc(answer.prompt_tokens)
    metrics.LLM_TOKENS.labels("completion").inc(answer.completion_tokens)


# --- Middleware -----------------------------------------------------------------------------


class RequestContext(BaseHTTPMiddleware):
    """Request ID (from X-Request-ID or new), bound to every log line and echoed back."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        request_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        structlog.contextvars.bind_contextvars(request_id=request_id, path=request.url.path)
        start = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["X-Request-ID"] = request_id
        route = request.scope.get("route")
        endpoint = getattr(route, "path", "unmatched")
        metrics.REQUESTS.labels(endpoint, str(response.status_code)).inc()
        log.info(
            "request",
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            ms=round((time.perf_counter() - start) * 1000),
        )
        return response


# --- Application ----------------------------------------------------------------------------


def create_app(settings: Settings | None = None, service: Service | None = None) -> FastAPI:
    settings = settings or get_settings()
    keys = KeyRing(settings.keys(), settings.keys(admin=True))
    limiter = RateLimiter(settings.rate_limit_per_minute)
    state: dict[str, Service] = {}  # "service" once built; "error" if building failed

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        configure_logging(settings.log_level, settings.log_json)
        if not keys.enabled:
            log.warning("api_open", reason="no REGLENS_API_KEYS configured: development only")
        try:
            state["service"] = service or build_service(settings, warm=settings.warm_on_start)
        except Exception as error:  # start anyway: /health stays up, /ready reports why
            log.error("service_unavailable", error=f"{type(error).__name__}: {error}"[:300])
            state["error"] = f"{type(error).__name__}: {error}"[:300]  # type: ignore[assignment]
        yield

    app = FastAPI(title="RegLens Maroc API", version="1.0.0", lifespan=lifespan)
    app.add_middleware(RequestContext)
    origins = [o.strip() for o in settings.cors_origins.split(",") if o.strip()]
    if origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=origins,
            allow_methods=["GET", "POST"],
            allow_headers=["X-API-Key", "Content-Type", "X-Request-ID"],
        )

    def caller(request: Request, minimum: Role) -> str:
        key = request.headers.get("X-API-Key")
        role = keys.role(key)
        if role is None:
            raise HTTPException(401, "Missing or invalid X-API-Key.")
        if minimum == "admin" and role != "admin":
            raise HTTPException(403, "This endpoint needs an admin key.")
        who = key_id(key) if key else (request.client.host if request.client else "unknown")
        allowed, retry_after = limiter.allow(who)
        if not allowed:
            metrics.RATE_LIMITED.inc()
            raise HTTPException(
                429,
                f"Rate limit of {settings.rate_limit_per_minute} requests per minute exceeded.",
                headers={"Retry-After": str(max(1, round(retry_after)))},
            )
        return who

    def user(request: Request) -> str:
        return caller(request, "user")

    def admin(request: Request) -> str:
        return caller(request, "admin")

    def svc() -> Service:
        if "service" not in state:
            raise HTTPException(503, "The service is not ready (index or providers unavailable).")
        return state["service"]

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness: the process is up."""
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> JSONResponse:
        """Readiness: the index is loaded and not empty."""
        service_ = state.get("service")
        chunks = service_.retriever.store.count() if service_ else 0
        ok = chunks > 0
        body: dict[str, object] = {"ready": ok, "chunks": chunks}
        if "error" in state:
            body["error"] = state["error"]
        return JSONResponse(body, status_code=200 if ok else 503)

    @app.get("/metrics")
    def prometheus() -> PlainTextResponse:
        return PlainTextResponse(
            generate_latest(metrics.REGISTRY).decode(), media_type=CONTENT_TYPE_LATEST
        )

    @app.post("/v1/query", response_model=QueryOut)
    def query(body: QueryIn, request: Request, _: Annotated[str, Depends(user)]) -> QueryOut:
        service_ = svc()
        k = body.k or service_.pipeline.top_k
        key = cache_key(body.question, k)
        cached = service_.cache.get(key)
        metrics.CACHE.labels("hit" if cached else "miss").inc()
        if cached is not None:
            return to_output(cached, request.state.request_id, cached=True)
        start = time.perf_counter()
        answer = service_.pipeline.answer(body.question)
        observe(answer, time.perf_counter() - start)
        if answer.guardrail == "ok":
            service_.cache.set(key, answer)
        return to_output(answer, request.state.request_id, cached=False)

    @app.post("/v1/query/stream")
    def query_stream(
        body: QueryIn, request: Request, _: Annotated[str, Depends(user)]
    ) -> StreamingResponse:
        """Server-Sent Events: `token` events with text, then one `answer` event (QueryOut)."""
        service_ = svc()
        request_id = request.state.request_id
        k = body.k or service_.pipeline.top_k
        key = cache_key(body.question, k)

        def sse(event: str, data: str) -> str:
            payload = "\n".join(f"data: {line}" for line in data.split("\n"))
            return f"event: {event}\n{payload}\n\n"

        def events() -> Iterator[str]:
            cached = service_.cache.get(key)
            metrics.CACHE.labels("hit" if cached else "miss").inc()
            if cached is not None:
                yield sse("token", json.dumps(cached.text or ""))
                yield sse("answer", to_output(cached, request_id, True).model_dump_json())
                return
            start = time.perf_counter()
            for piece in service_.pipeline.stream(body.question):
                if isinstance(piece, Answer):
                    observe(piece, time.perf_counter() - start)
                    if piece.guardrail == "ok":
                        service_.cache.set(key, piece)
                    yield sse("answer", to_output(piece, request_id, False).model_dump_json())
                else:
                    yield sse("token", json.dumps(piece))

        return StreamingResponse(
            events(),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.post("/v1/ingest", response_model=IngestOut)
    async def ingest(
        _: Annotated[str, Depends(admin)],
        file: Annotated[UploadFile, File(description="PDF of an official text")],
        title: Annotated[str, Form(min_length=3, max_length=500)],
        issuer: Annotated[Issuer, Form()],
        url: Annotated[str, Form(pattern=r"^https?://")],
        reference: Annotated[str | None, Form(max_length=100)] = None,
        language: Annotated[Language, Form()] = "fr",
    ) -> IngestOut:
        """Add or update one document (admin). PDF only, size-capped, OCR'd when scanned."""
        limit = settings.upload_max_mb * 1024 * 1024
        content = await file.read(limit + 1)
        if len(content) > limit:
            raise HTTPException(413, f"File larger than {settings.upload_max_mb} MB.")
        if not (file.filename or "").lower().endswith(".pdf") or not content.startswith(b"%PDF"):
            raise HTTPException(415, "Only PDF files are accepted.")
        stem = re.sub(r"[^a-z0-9]+", "-", Path(file.filename or "upload").stem.lower()).strip("-")
        raw_dir = settings.data_dir
        target = raw_dir / issuer.lower() / f"{stem or 'document'}.pdf"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        write_sidecar(
            target,
            DocumentMetadata(
                title=title, issuer=issuer, reference=reference, url=url, language=language
            ),
        )
        try:
            ocr = ocr_document(target).status == "ocr"
        except OcrUnavailableError as error:
            target.unlink(missing_ok=True)
            target.with_suffix(".yaml").unlink(missing_ok=True)
            raise HTTPException(
                503, "This PDF is scanned and needs OCR, which is not available on this server."
            ) from error
        status, chunks = svc().ingest_file(target, raw_dir)
        assert status in ("added", "updated", "unchanged")
        log.info("api_ingest", file=target.name, status=status, chunks=chunks, ocr=ocr)
        return IngestOut(
            file=target.relative_to(raw_dir).as_posix(),
            status=status,
            chunks=chunks,
            ocr=ocr,
        )

    return app


app = create_app()
