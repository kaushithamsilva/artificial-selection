from dataclasses import dataclass


@dataclass(frozen=True)
class GenerationStats:
    """Population-level summary logged once per generation."""

    generation_number: int
    avg_e: float
    avg_b: float
    avg_n: float
    avg_fitness: float
    phase: int
