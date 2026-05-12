from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import time
from typing import Iterator, Mapping


@dataclass(frozen=True, slots=True)
class ProfilingEvent:
    stage: str
    elapsed_ms: float
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ProfilingSummary:
    event_count: int
    total_ms: float
    stage_ms: dict[str, float]


class BenchmarkProfiler:
    def __init__(self) -> None:
        self._events: list[ProfilingEvent] = []

    @property
    def events(self) -> tuple[ProfilingEvent, ...]:
        return tuple(self._events)

    def record(self, stage: str, elapsed_ms: float, metadata: Mapping[str, object] | None = None) -> None:
        if not stage:
            raise ValueError("profiling stage is required")
        if elapsed_ms < 0.0:
            raise ValueError("profiling elapsed_ms must be non-negative")
        self._events.append(ProfilingEvent(stage=stage, elapsed_ms=float(elapsed_ms), metadata=dict(metadata or {})))

    @contextmanager
    def stage(self, stage: str, metadata: Mapping[str, object] | None = None) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(stage, (time.perf_counter() - start) * 1000.0, metadata)

    def summary(self) -> ProfilingSummary:
        totals: dict[str, float] = {}
        for event in self._events:
            totals[event.stage] = totals.get(event.stage, 0.0) + event.elapsed_ms
        return ProfilingSummary(
            event_count=len(self._events),
            total_ms=sum(totals.values()),
            stage_ms=totals,
        )

    def latency_samples_ms(self) -> tuple[float, ...]:
        return tuple(event.elapsed_ms for event in self._events)
