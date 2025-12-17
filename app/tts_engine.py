from __future__ import annotations

import logging
import threading
from typing import Optional

import numpy as np
import torch

from chatterbox.mtl_tts import ChatterboxMultilingualTTS
from chatterbox.tts import ChatterboxTTS

from .config import Settings

logger = logging.getLogger(__name__)


def _ensure_watermarker_available() -> None:
    """
    Chatterbox tries to instantiate perth.PerthImplicitWatermarker at import time.
    On some platforms the compiled extension is missing and the symbol is None, which
    causes a startup crash. Fall back to the dummy implementation so synthesis still works.
    """
    try:
        import perth
    except Exception:  # pragma: no cover - defensive
        logger.warning("perth package not available; watermarking disabled")
        return

    if getattr(perth, "PerthImplicitWatermarker", None) is None:
        logger.warning("PerthImplicitWatermarker unavailable; using DummyWatermarker fallback")
        perth.PerthImplicitWatermarker = perth.DummyWatermarker


class ChatterboxEngine:
    """Wrapper around ResembleAI Chatterbox models."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model = None
        self._lock = threading.Lock()

    def ensure_loaded(self):
        if self._model is not None:
            return self._model
        with self._lock:
            if self._model is not None:
                return self._model
            logger.info("Loading Chatterbox model (%s, %s)", self.settings.model_id, self.settings.model_variant)
            model_cls = ChatterboxMultilingualTTS if self.settings.model_variant == "multilingual" else ChatterboxTTS
            device = torch.device(self.settings.device)
            _ensure_watermarker_available()
            model = model_cls.from_pretrained(device=device)
            self._model = model
            logger.info("Chatterbox model ready")
            return model

    def synthesize(
        self,
        text: str,
        *,
        language_id: Optional[str] = None,
        audio_prompt_path: Optional[str] = None,
        cfg_scale: Optional[float] = None,
        exaggeration: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        min_p: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> tuple[np.ndarray, int]:
        model = self.ensure_loaded()
        kwargs = {}
        if isinstance(model, ChatterboxMultilingualTTS):
            if not language_id:
                raise ValueError("language_id is required for multilingual model")
            kwargs["language_id"] = language_id.lower()
            if audio_prompt_path:
                kwargs["audio_prompt_path"] = audio_prompt_path
            else:
                raise ValueError("audio_prompt_path must be provided for multilingual usage")
            if cfg_scale is not None:
                kwargs["cfg_weight"] = cfg_scale
            if exaggeration is not None:
                kwargs["exaggeration"] = exaggeration
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if min_p is not None:
                kwargs["min_p"] = min_p
            if repetition_penalty is not None:
                kwargs["repetition_penalty"] = repetition_penalty
        else:
            if language_id:
                logger.debug("language_id ignored for base model")
            if audio_prompt_path:
                kwargs["audio_prompt_path"] = audio_prompt_path
            if cfg_scale is not None:
                kwargs["cfg_weight"] = cfg_scale
            if exaggeration is not None:
                kwargs["exaggeration"] = exaggeration
            if temperature is not None:
                kwargs["temperature"] = temperature
            if top_p is not None:
                kwargs["top_p"] = top_p
            if min_p is not None:
                kwargs["min_p"] = min_p
            if repetition_penalty is not None:
                kwargs["repetition_penalty"] = repetition_penalty
            if top_k:
                logger.debug("top_k parameter is not used by Chatterbox base model; ignoring.")

        with torch.inference_mode():
            audio = model.generate(text, **kwargs)

        if isinstance(audio, torch.Tensor):
            waveform = audio.squeeze(0).detach().cpu().numpy()
        else:
            waveform = np.asarray(audio)

        sample_rate = getattr(model, "sr", 44100)
        return waveform, int(sample_rate)
