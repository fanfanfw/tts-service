from __future__ import annotations

from typing import AsyncIterator, Iterator

import numpy as np
import torch
import torchaudio

PCM_SAMPLE_WIDTH = 2


def ensure_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    if audio.ndim == 2:
        return audio.mean(axis=0)
    raise ValueError("audio array must be 1D or 2D")


def resample_audio(audio: np.ndarray, input_rate: int, output_rate: int) -> np.ndarray:
    if input_rate == output_rate:
        return audio
    tensor = torch.from_numpy(audio).float().unsqueeze(0)
    resampled = torchaudio.functional.resample(tensor, input_rate, output_rate)
    return resampled.squeeze(0).cpu().numpy()


def float_to_pcm16(audio: np.ndarray) -> bytes:
    audio = np.clip(audio, -1.0, 1.0)
    return (audio * 32767.0).astype(np.int16).tobytes()


def chunk_pcm16(audio: np.ndarray, sample_rate: int, chunk_ms: int) -> Iterator[bytes]:
    samples_per_chunk = int(sample_rate * (chunk_ms / 1000.0))
    samples_per_chunk = max(samples_per_chunk, sample_rate // 50)
    total_samples = audio.shape[-1]
    for start in range(0, total_samples, samples_per_chunk):
        chunk = audio[start : start + samples_per_chunk]
        if not len(chunk):
            continue
        yield float_to_pcm16(chunk)


async def stream_chunks(
    audio: np.ndarray,
    input_rate: int,
    target_rate: int,
    chunk_ms: int,
) -> AsyncIterator[bytes]:
    mono = ensure_mono(audio)
    resampled = resample_audio(mono, input_rate, target_rate)
    for chunk in chunk_pcm16(resampled, target_rate, chunk_ms):
        yield chunk
