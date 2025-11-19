from __future__ import annotations

from contextlib import contextmanager
from time import perf_counter

from prometheus_client import Counter, Gauge, Histogram

SYNTH_REQUESTS = Counter(
    "tts_requests_total",
    "Number of synthesis requests",
    labelnames=("tenant",),
)

SYNTH_ACTIVE = Gauge(
    "tts_stream_active",
    "Active TTS streams",
)

AUDIO_BYTES = Counter(
    "tts_audio_bytes_total",
    "Total PCM bytes streamed",
    labelnames=("tenant",),
)

SYNTH_LATENCY = Histogram(
    "tts_generation_latency_seconds",
    "Model generation latency",
    buckets=(0.5, 1, 2, 4, 8, 16, 32, 64),
)


@contextmanager
def track_latency():
    start = perf_counter()
    try:
        yield
    finally:
        SYNTH_LATENCY.observe(perf_counter() - start)


@contextmanager
def session_tracker(tenant: str | None = None):
    tenant_label = tenant or "default"
    SYNTH_REQUESTS.labels(tenant_label).inc()
    SYNTH_ACTIVE.inc()
    try:
        yield
    finally:
        SYNTH_ACTIVE.dec()
