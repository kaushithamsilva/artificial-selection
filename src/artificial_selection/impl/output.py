import csv
from dataclasses import asdict
from pathlib import Path
from typing import TextIO

from loguru import logger

from artificial_selection.core.domain.stats import GenerationStats

CSV_FIELDS = ["generation_number", "avg_E", "avg_B", "avg_N", "avg_fitness", "phase"]


class CsvResultWriter:
    """Writes one row per generation to a CSV file for downstream plotting."""

    def __init__(self, path: Path) -> None:
        self._file: TextIO = path.open("w", newline="")
        self._writer = csv.writer(self._file)
        self._writer.writerow(CSV_FIELDS)

    def record(self, stats: GenerationStats) -> None:
        self._writer.writerow(
            [
                stats.generation_number,
                stats.avg_e,
                stats.avg_b,
                stats.avg_n,
                stats.avg_fitness,
                stats.phase,
            ]
        )

    def close(self) -> None:
        self._file.close()


class JsonlEventWriter:
    """Publishes each generation's stats as a structured JSONL event via loguru.

    Pairs with a plain-text trace sink (see cli.configure_logging): this sink
    only ever receives records tagged event_type="generation_stats".
    """

    def __init__(self, path: Path) -> None:
        self._sink_id = logger.add(
            path,
            serialize=True,
            level="INFO",
            filter=lambda record: record["extra"].get("event_type") == "generation_stats",
        )

    def record(self, stats: GenerationStats) -> None:
        logger.bind(event_type="generation_stats", **asdict(stats)).info("generation_stats")

    def close(self) -> None:
        logger.remove(self._sink_id)
