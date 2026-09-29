"""Orchestrator: runs the GA generation loop.

Per generation G:
  1. Decode genotype (E, B, N capacities) from every genome, then apply
     substrate depletion to get the expressed phenotype (B, N only).
  2. Evaluate fitness: natural selection (viability-gated) while
     G < phase_1_generations, artificial selection (yield-only) afterwards.
  3. Log population-average stats and the full phenotype distribution for G.
  4. Select parent pairs (ISelectionStrategy) and breed the next population via
     single-point crossover and per-bit mutation.
"""

import numpy as np
from loguru import logger

from artificial_selection.core import genetics
from artificial_selection.core.domain.config import SimulationConfig
from artificial_selection.core.domain.distribution import PHENOTYPE_BIN_EDGES, PopulationSnapshot
from artificial_selection.core.domain.stats import GenerationStats
from artificial_selection.core.genetics import FloatArray, Genomes
from artificial_selection.core.interfaces.output import IOutputWriter
from artificial_selection.core.interfaces.selection import ISelectionStrategy

_BIN_EDGES = np.asarray(PHENOTYPE_BIN_EDGES)


def _initialize_population(config: SimulationConfig, rng: np.random.Generator) -> Genomes:
    genomes: Genomes = rng.integers(
        0, 2, size=(config.population_size, config.genome_length), dtype=np.uint8
    )
    return genomes


def _histogram_phenotype(values: FloatArray) -> tuple[int, ...]:
    counts, _ = np.histogram(values, bins=_BIN_EDGES)
    return tuple(int(c) for c in counts)


def run(
    config: SimulationConfig,
    selection_strategy: ISelectionStrategy,
    output_writers: list[IOutputWriter],
) -> tuple[list[GenerationStats], list[PopulationSnapshot]]:
    rng = np.random.default_rng(config.seed)
    genomes = _initialize_population(config, rng)
    history: list[GenerationStats] = []
    snapshots: list[PopulationSnapshot] = []

    for generation in range(config.total_generations):
        e_potential, b_capacity, n_capacity = genetics.decode_phenotypes(genomes)
        e, b_expressed, n_expressed = genetics.compute_expressed_phenotypes(
            e_potential, b_capacity, n_capacity, config.k_bone, config.k_energy
        )

        if generation < config.phase_1_generations:
            fitness = genetics.fitness_natural_selection(
                e,
                b_expressed,
                n_expressed,
                config.b_min,
                config.n_min,
                config.viability_steepness,
            )
            phase = 1
        else:
            if generation == config.phase_1_generations:
                logger.info(
                    "Phase transition at generation {}: switching to artificial selection",
                    generation,
                )
            fitness = genetics.fitness_artificial_selection(e, config.artificial_alpha)
            phase = 2

        stats = GenerationStats(
            generation_number=generation,
            avg_e=float(np.mean(e)),
            avg_b=float(np.mean(b_expressed)),
            avg_n=float(np.mean(n_expressed)),
            avg_fitness=float(np.mean(fitness)),
            phase=phase,
        )
        history.append(stats)
        for writer in output_writers:
            writer.record(stats)

        snapshots.append(
            PopulationSnapshot(
                generation_number=generation,
                phase=phase,
                e_counts=_histogram_phenotype(e),
                b_counts=_histogram_phenotype(b_expressed),
                n_counts=_histogram_phenotype(n_expressed),
            )
        )

        parent_a_idx, parent_b_idx = selection_strategy.select_parents(
            fitness, rng, config.population_size
        )
        children = genetics.single_point_crossover(
            genomes[parent_a_idx], genomes[parent_b_idx], rng
        )
        genomes = genetics.mutate(children, config.mutation_rate, rng)

    return history, snapshots
