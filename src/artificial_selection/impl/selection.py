import numpy as np
from numpy.typing import NDArray


class RouletteWheelSelection:
    """Fitness-proportionate selection: P(pick i) = fitness_i / sum(fitness).

    Unlike tournament selection (which only compares rank and so treats any
    nonzero fitness difference as decisive), this respects the actual
    magnitude of fitness, so a fitness function's own parameters act as real,
    continuous dials on selection strength rather than being flattened to an
    all-or-nothing signal. Structurally satisfies ISelectionStrategy.
    """

    def select_parents(
        self,
        fitness: NDArray[np.float64],
        rng: np.random.Generator,
        num_pairs: int,
    ) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
        probabilities = self._selection_probabilities(fitness)
        population_size = fitness.shape[0]
        parent_a = rng.choice(population_size, size=num_pairs, p=probabilities)
        parent_b = rng.choice(population_size, size=num_pairs, p=probabilities)
        return parent_a.astype(np.int64), parent_b.astype(np.int64)

    @staticmethod
    def _selection_probabilities(fitness: NDArray[np.float64]) -> NDArray[np.float64]:
        total = fitness.sum()
        if total <= 0:
            population_size = fitness.shape[0]
            uniform: NDArray[np.float64] = np.full(population_size, 1.0 / population_size)
            return uniform
        probabilities: NDArray[np.float64] = fitness / total
        return probabilities
