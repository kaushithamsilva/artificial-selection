import inspect

import numpy as np

from artificial_selection.core import genetics


def test_decode_phenotypes_reads_correct_bit_blocks() -> None:
    genome = np.zeros((1, 32), dtype=np.uint8)
    genome[0, 0:3] = [1, 1, 1]  # E = 7
    genome[0, 10:13] = [0, 1, 0]  # B = 2
    genome[0, 20:23] = [1, 0, 0]  # N = 4

    e, b, n = genetics.decode_phenotypes(genome)

    assert e[0] == 7
    assert b[0] == 2
    assert n[0] == 4


def test_compute_expressed_phenotypes_applies_depletion() -> None:
    e_potential = np.array([0, 3, 7])
    b_capacity = np.array([5, 5, 5])
    n_capacity = np.array([5, 5, 5])

    e, b_exp, n_exp = genetics.compute_expressed_phenotypes(
        e_potential, b_capacity, n_capacity, k_bone=0.8, k_energy=0.6
    )

    assert np.allclose(e, [0, 3, 7])
    assert np.allclose(b_exp, [5.0, 5 - 0.8 * 3, 0.0])  # E=7: 5-0.8*7=-0.6 -> floored to 0
    assert np.allclose(n_exp, [5.0, 5 - 0.6 * 3, 5 - 0.6 * 7])


def test_compute_expressed_phenotypes_floors_at_zero() -> None:
    e_potential = np.array([7])
    b_capacity = np.array([1])  # 1 - 0.8*7 = -4.6, must floor to 0, not go negative
    n_capacity = np.array([1])

    _, b_exp, n_exp = genetics.compute_expressed_phenotypes(
        e_potential, b_capacity, n_capacity, k_bone=0.8, k_energy=0.6
    )

    assert b_exp[0] == 0.0
    assert n_exp[0] == 0.0


def test_fitness_natural_selection_matches_formula() -> None:
    e = np.array([3.0])
    b_exp = np.array([4.0])
    n_exp = np.array([1.0])

    fitness = genetics.fitness_natural_selection(
        e, b_exp, n_exp, b_min=2.0, n_min=2.0, viability_steepness=2.0
    )

    viability_b = 1 / (1 + np.exp(-2.0 * (4.0 - 2.0)))
    viability_n = 1 / (1 + np.exp(-2.0 * (1.0 - 2.0)))
    assert np.allclose(fitness, [3.0 * viability_b * viability_n])


def test_fitness_natural_selection_collapses_when_expressed_traits_are_hollowed_out() -> None:
    # Same E, but one bird's bones/energy are depleted well below b_min/n_min.
    e = np.array([5.0, 5.0])
    b_exp = np.array([5.0, 0.0])
    n_exp = np.array([5.0, 0.0])

    fitness = genetics.fitness_natural_selection(e, b_exp, n_exp)

    assert fitness[0] > fitness[1]
    assert fitness[1] < 1.0  # near-total viability collapse, not just a mild penalty


def test_fitness_artificial_selection_matches_formula() -> None:
    e = np.array([3.0, 7.0])

    fitness = genetics.fitness_artificial_selection(e, alpha=2.0)

    assert np.allclose(fitness, [9.0, 49.0])


def test_fitness_artificial_selection_has_no_expressed_trait_parameters() -> None:
    # B and N aren't merely unrewarded here -- they're not inputs at all.
    params = inspect.signature(genetics.fitness_artificial_selection).parameters
    assert "b_expressed" not in params
    assert "n_expressed" not in params


def test_high_egg_potential_depletes_bone_and_collapses_natural_fitness() -> None:
    # Same B/N capacity, different E: the causal link runs through
    # compute_expressed_phenotypes, not a formula combining independent traits.
    e_potential = np.array([1, 7])
    b_capacity = np.array([5, 5])
    n_capacity = np.array([5, 5])

    e, b_exp, n_exp = genetics.compute_expressed_phenotypes(
        e_potential, b_capacity, n_capacity, k_bone=0.8, k_energy=0.6
    )
    fitness = genetics.fitness_natural_selection(e, b_exp, n_exp)

    assert b_exp[1] < b_exp[0]  # the high-E bird's bones are more depleted
    assert fitness[1] < fitness[0]  # ...and that costs it fitness, not just yield


def test_fitness_is_never_negative() -> None:
    e_potential = np.array([0, 7])
    b_capacity = np.array([0, 7])
    n_capacity = np.array([0, 7])
    e, b_exp, n_exp = genetics.compute_expressed_phenotypes(
        e_potential, b_capacity, n_capacity, k_bone=0.8, k_energy=0.6
    )

    natural = genetics.fitness_natural_selection(e, b_exp, n_exp)
    artificial = genetics.fitness_artificial_selection(e)

    assert np.all(natural >= 0.0)
    assert np.all(artificial >= 0.0)


def test_single_point_crossover_respects_pivot() -> None:
    rng = np.random.default_rng(0)
    parents_a = np.zeros((5, 32), dtype=np.uint8)
    parents_b = np.ones((5, 32), dtype=np.uint8)

    children = genetics.single_point_crossover(parents_a, parents_b, rng)

    for child in children:
        # child = A (all zeros) up to and including the pivot, then B (all ones).
        assert child[0] == 0  # pivot is always >= 1, so bit 0 is always from A
        transitions = np.flatnonzero(np.diff(child.astype(np.int64)))
        assert len(transitions) <= 1  # at most one 0->1 switch, or none at all


def test_mutate_flips_all_bits_at_rate_one() -> None:
    rng = np.random.default_rng(0)
    genomes = np.zeros((4, 32), dtype=np.uint8)
    mutated = genetics.mutate(genomes, mutation_rate=1.0, rng=rng)
    assert np.all(mutated == 1)


def test_mutate_flips_no_bits_at_rate_zero() -> None:
    rng = np.random.default_rng(0)
    genomes = np.zeros((4, 32), dtype=np.uint8)
    mutated = genetics.mutate(genomes, mutation_rate=0.0, rng=rng)
    assert np.all(mutated == 0)
