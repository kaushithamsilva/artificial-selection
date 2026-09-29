import tomllib
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SimulationConfig:
    """Hyperparameters for the avian domestication GA simulation."""

    population_size: int = 1000
    genome_length: int = 32
    mutation_rate: float = 1.0 / 32.0
    phase_1_generations: int = 200
    phase_2_generations: int = 200
    seed: int | None = 42

    # Substrate depletion model (see core/genetics.py). k_bone/k_energy are
    # shared by both phases; b_min/n_min/viability_steepness shape phase 1's
    # survival curve only -- phase 2 never references them.
    k_bone: float = 0.8
    k_energy: float = 0.6
    b_min: float = 2.0
    n_min: float = 2.0
    viability_steepness: float = 2.0
    artificial_alpha: float = 2.0

    @property
    def total_generations(self) -> int:
        return self.phase_1_generations + self.phase_2_generations

    @classmethod
    def from_toml(cls, path: Path) -> "SimulationConfig":
        with path.open("rb") as f:
            data: dict[str, Any] = tomllib.load(f)

        known_fields = {f.name for f in fields(cls)}
        unknown = set(data) - known_fields
        if unknown:
            raise ValueError(
                f"Unknown key(s) in {path}: {sorted(unknown)}. "
                f"Valid keys are: {sorted(known_fields)}"
            )
        return cls(**data)
