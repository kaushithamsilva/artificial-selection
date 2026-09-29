from typing import Protocol

from artificial_selection.core.domain.stats import GenerationStats


class IOutputWriter(Protocol):
    """Sink for per-generation results (e.g. a CSV file or a JSONL event log)."""

    def record(self, stats: GenerationStats) -> None: ...

    def close(self) -> None: ...
