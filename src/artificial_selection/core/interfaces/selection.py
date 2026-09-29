from typing import Protocol

import numpy as np
from numpy.typing import NDArray


class ISelectionStrategy(Protocol):
    """Picks parent pairs for the next generation, weighted by fitness."""

    def select_parents(
        self,
        fitness: NDArray[np.float64],
        rng: np.random.Generator,
        num_pairs: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        """Return (parent_a_indices, parent_b_indices), each of length num_pairs."""
        ...
