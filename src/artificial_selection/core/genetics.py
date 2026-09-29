"""Vectorized genome decoding, fitness, crossover and mutation.

Population genomes are stored as a (population_size, genome_length) uint8
array of bits, so a 3-bit exon block decodes as bits[..., 0]*4 + bits[..., 1]*2
+ bits[..., 2], matching bit position 0 as the most significant bit of the block.
"""

import numpy as np
from numpy.typing import NDArray

Genomes = NDArray[np.uint8]
IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

E_GENE_START = 0
B_GENE_START = 10
N_GENE_START = 20
GENE_WIDTH = 3


def _decode_gene(genomes: Genomes, start: int) -> IntArray:
    block = genomes[:, start : start + GENE_WIDTH].astype(np.int64)
    return block[:, 0] * 4 + block[:, 1] * 2 + block[:, 2]


def decode_phenotypes(genomes: Genomes) -> tuple[IntArray, IntArray, IntArray]:
    """Extract (E, B, N) exon values, each in [0, 7], from every genome.

    These are genetic *capacities* (the ceiling the genotype allows), not yet
    the expressed phenotype -- see compute_expressed_phenotypes.
    """
    e = _decode_gene(genomes, E_GENE_START)
    b = _decode_gene(genomes, B_GENE_START)
    n = _decode_gene(genomes, N_GENE_START)
    return e, b, n


def compute_expressed_phenotypes(
    e_potential: IntArray,
    b_capacity: IntArray,
    n_capacity: IntArray,
    k_bone: float,
    k_energy: float,
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """Substrate depletion: egg production draws directly on bone and energy.

    Producing E eggs obligately extracts k_bone*E calcium and k_energy*E
    energy from the bird's genetic capacity, floored at 0 (a skeleton can't
    have negative bone density). This is the actual observable phenotype --
    B and N are only ever selected on, logged and plotted in their expressed
    form, never as the raw underlying capacity.
    """
    e = e_potential.astype(np.float64)
    b_expressed: FloatArray = np.maximum(0.0, b_capacity.astype(np.float64) - k_bone * e)
    n_expressed: FloatArray = np.maximum(0.0, n_capacity.astype(np.float64) - k_energy * e)
    return e, b_expressed, n_expressed


def fitness_natural_selection(
    e: FloatArray,
    b_expressed: FloatArray,
    n_expressed: FloatArray,
    b_min: float = 2.0,
    n_min: float = 2.0,
    viability_steepness: float = 2.0,
) -> FloatArray:
    """Reproduction only counts if the bird survives: intact bones, adequate energy.

    Viability is a logistic curve of each expressed trait relative to its
    survival threshold -- smooth rather than a hard cutoff, but steep enough
    that dropping much below the threshold collapses survival probability.
    A bird whose skeleton has been hollowed out by heavy egg production
    fails on viability here, whatever its egg count -- the coupling between
    E and B/N is physiological (via compute_expressed_phenotypes), not a
    separate fitness penalty bolted on afterward.
    """
    viability_b = 1.0 / (1.0 + np.exp(-viability_steepness * (b_expressed - b_min)))
    viability_n = 1.0 / (1.0 + np.exp(-viability_steepness * (n_expressed - n_min)))
    fitness: FloatArray = (e * viability_b * viability_n).astype(np.float64)
    return fitness


def fitness_artificial_selection(e: FloatArray, alpha: float = 2.0) -> FloatArray:
    """The farmer subsidizes viability (cages, feed, no predators) and selects on yield alone.

    B and N never enter this formula at all: the farmer is not weakly
    penalizing them, they are simply invisible to selection. Whatever happens
    to expressed bone density and energy is now a pure side effect of how
    much egg production the genotype drives, via compute_expressed_phenotypes.
    """
    fitness: FloatArray = (e**alpha).astype(np.float64)
    return fitness


def single_point_crossover(
    parents_a: Genomes, parents_b: Genomes, rng: np.random.Generator
) -> Genomes:
    """Child bits [0, pivot] come from parent A, the rest from parent B."""
    num_children, genome_length = parents_a.shape
    pivots: IntArray = rng.integers(1, genome_length, size=(num_children,))
    positions = np.arange(genome_length)
    take_from_a = positions[np.newaxis, :] <= pivots[:, np.newaxis]
    return np.where(take_from_a, parents_a, parents_b).astype(np.uint8)


def mutate(genomes: Genomes, mutation_rate: float, rng: np.random.Generator) -> Genomes:
    """Flip each bit independently with probability mutation_rate."""
    flip_mask = rng.random(genomes.shape) < mutation_rate
    return np.bitwise_xor(genomes, flip_mask.astype(np.uint8))
