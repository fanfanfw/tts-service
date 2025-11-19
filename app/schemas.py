from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SynthesisRequest(BaseModel):
    session_id: str = Field(..., min_length=1)
    tenant: Optional[str] = None
    text: str = Field(..., min_length=1, max_length=2000)
    seed: Optional[int] = None
    chunk_ms: Optional[int] = Field(default=None, ge=50, le=1000)
    language_id: Optional[str] = Field(default=None, min_length=1)
    audio_prompt_path: Optional[str] = Field(default=None, min_length=1)
    cfg_scale: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    exaggeration: Optional[float] = Field(default=None, ge=0.0, le=2.0)
    temperature: Optional[float] = Field(default=None, gt=0.0)
    top_p: Optional[float] = Field(default=None, gt=0.0, le=1.0)
    min_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    repetition_penalty: Optional[float] = Field(default=None, gt=0.0)
    top_k: Optional[int] = Field(default=None, ge=1, le=256)


class ErrorResponse(BaseModel):
    detail: str
