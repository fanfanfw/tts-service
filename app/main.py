from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Optional

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import ValidationError

from .audio import stream_chunks
from .config import Settings, get_settings
from .metrics import AUDIO_BYTES, session_tracker, track_latency
from .schemas import ErrorResponse, SynthesisRequest
from .tts_engine import ChatterboxEngine

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Chatterbox TTS Service", version="0.2.0")
_engine: Optional[ChatterboxEngine] = None


def get_engine(settings: Settings = Depends(get_settings)) -> ChatterboxEngine:
    global _engine
    if _engine is None:
        _engine = ChatterboxEngine(settings)
    return _engine


async def verify_auth(
    settings: Settings = Depends(get_settings),
    authorization: str | None = Header(default=None, alias="Authorization"),
) -> None:
    token = settings.auth_token
    if not token:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="missing or invalid authorization header")
    value = authorization.split(" ", 1)[1]
    if value != token:
        raise HTTPException(status_code=403, detail="invalid auth token")


@app.get("/healthz")
async def health(settings: Settings = Depends(get_settings)) -> dict[str, Any]:
    return {
        "status": "ok",
        "model": settings.model_id,
        "variant": settings.model_variant,
        "default_language": settings.default_language_id,
        "device": settings.device,
        "chunk_ms": settings.chunk_duration_ms,
    }


@app.get("/metrics")
async def metrics(settings: Settings = Depends(get_settings)):
    if not settings.metrics_endpoint_enabled:
        return JSONResponse(status_code=404, content={"detail": "metrics disabled"})
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/tts/stream")
async def tts_stream(
    payload: SynthesisRequest,
    settings: Settings = Depends(get_settings),
    engine: ChatterboxEngine = Depends(get_engine),
    _: None = Depends(verify_auth),
):
    if len(payload.text) > settings.max_text_chars:
        raise HTTPException(status_code=400, detail="text is too long")

    tenant = payload.tenant or "default"
    chunk_ms = payload.chunk_ms or settings.chunk_duration_ms
    language_id = payload.language_id or (settings.default_language_id if settings.model_variant == "multilingual" else None)
    cfg_scale = payload.cfg_scale if payload.cfg_scale is not None else settings.default_cfg_scale
    exaggeration = payload.exaggeration if payload.exaggeration is not None else settings.default_exaggeration
    temperature = payload.temperature if payload.temperature is not None else settings.default_temperature
    top_p = payload.top_p if payload.top_p is not None else settings.default_top_p
    min_p = payload.min_p if payload.min_p is not None else settings.default_min_p
    repetition_penalty = (
        payload.repetition_penalty if payload.repetition_penalty is not None else settings.default_repetition_penalty
    )
    audio_prompt_path = payload.audio_prompt_path or settings.default_audio_prompt_path
    top_k = payload.top_k

    loop = asyncio.get_running_loop()

    def _run_model():
        with track_latency():
            return engine.synthesize(
                payload.text,
                language_id=language_id,
                audio_prompt_path=audio_prompt_path,
                cfg_scale=cfg_scale,
                exaggeration=exaggeration,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
            )

    try:
        audio, source_rate = await loop.run_in_executor(None, _run_model)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("failed to synthesize audio")
        raise HTTPException(status_code=500, detail=f"synthesis failed: {exc}") from exc

    async def audio_generator() -> AsyncIterator[bytes]:
        with session_tracker(tenant):
            async for chunk in stream_chunks(
                audio,
                source_rate,
                settings.output_sample_rate,
                chunk_ms,
            ):
                AUDIO_BYTES.labels(tenant).inc(len(chunk))
                yield chunk

    headers = {
        "X-Session-Id": payload.session_id,
        "X-Tenant": tenant,
        "X-Sample-Rate": str(settings.output_sample_rate),
        "Content-Type": "application/octet-stream",
    }
    return StreamingResponse(audio_generator(), media_type="application/octet-stream", headers=headers)


@app.exception_handler(ValidationError)
async def validation_exception_handler(_: Request, exc: ValidationError):
    return JSONResponse(status_code=422, content=ErrorResponse(detail=str(exc)).model_dump())


@app.on_event("startup")
async def warmup_models() -> None:
    settings = get_settings()
    if not settings.warmup_on_start:
        return
    loop = asyncio.get_running_loop()
    engine = get_engine(settings)
    await loop.run_in_executor(None, engine.ensure_loaded)
