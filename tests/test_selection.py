import numpy as np

from artificial_selection.impl.selection import RouletteWheelSelection


def test_roulette_selection_returns_valid_indices() -> None:
    rng = np.random.default_rng(0)
    fitness = np.array([1.0, 5.0, 0.0, 3.0, 9.0])
    strategy = RouletteWheelSelection()

    parent_a, parent_b = strategy.select_parents(fitness, rng, num_pairs=1000)

    assert parent_a.shape == (1000,)
    assert parent_b.shape == (1000,)
    assert parent_a.min() >= 0
    assert parent_a.max() < len(fitness)


def test_roulette_selection_probability_matches_fitness_share() -> None:
    rng = np.random.default_rng(1)
    fitness = np.array([1.0, 1.0, 1.0, 1.0, 6.0])  # agent 4 holds 60% of total fitness
    strategy = RouletteWheelSelection()

    parent_a, _ = strategy.select_parents(fitness, rng, num_pairs=20_000)

    observed_share = (parent_a == 4).mean()
    assert abs(observed_share - 0.6) < 0.02


def test_roulette_selection_never_picks_zero_fitness_agent_when_others_are_positive() -> None:
    rng = np.random.default_rng(2)
    fitness = np.array([0.0, 10.0, 0.0, 5.0])
    strategy = RouletteWheelSelection()

    parent_a, parent_b = strategy.select_parents(fitness, rng, num_pairs=2000)

    assert 0 not in parent_a
    assert 2 not in parent_a
    assert 0 not in parent_b
    assert 2 not in parent_b


def test_roulette_selection_falls_back_to_uniform_when_all_fitness_is_zero() -> None:
    rng = np.random.default_rng(3)
    fitness = np.zeros(4)
    strategy = RouletteWheelSelection()

    parent_a, _ = strategy.select_parents(fitness, rng, num_pairs=4000)

    counts = np.bincount(parent_a, minlength=4) / 4000
    assert np.allclose(counts, 0.25, atol=0.03)
