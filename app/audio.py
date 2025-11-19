from __future__ import annotations

import asyncio
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


def _chunk_indices(length: int, samples_per_chunk: int) -> Iterator[tuple[int, int]]:
    for start in range(0, length, samples_per_chunk):
        end = min(start + samples_per_chunk, length)
        if end > start:
            yield start, end


async def stream_chunks(
    audio: np.ndarray,
    input_rate: int,
    target_rate: int,
    chunk_ms: int,
) -> AsyncIterator[bytes]:
    mono = ensure_mono(audio)
    samples_per_chunk = int(input_rate * (chunk_ms / 1000.0))
    samples_per_chunk = max(samples_per_chunk, input_rate // 50)

    async def _resample_chunk(chunk: np.ndarray) -> bytes:
        return await asyncio.to_thread(_process_chunk, chunk, input_rate, target_rate)

    for start, end in _chunk_indices(mono.shape[-1], samples_per_chunk):
        chunk = mono[start:end]
        if not len(chunk):
            continue
        yield await _resample_chunk(chunk)


def _process_chunk(chunk: np.ndarray, input_rate: int, target_rate: int) -> bytes:
    resampled = resample_audio(chunk, input_rate, target_rate)
    return float_to_pcm16(resampled)
