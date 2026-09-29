from dataclasses import dataclass

#: Bin edges for histogramming expressed phenotypes over [0, 7]. Expressed B/N
#: are continuous (see compute_expressed_phenotypes), so this bins rather than
#: counting exact values; used uniformly for E, B and N.
PHENOTYPE_MIN = 0.0
PHENOTYPE_MAX = 7.0
PHENOTYPE_BINS = 28
PHENOTYPE_BIN_EDGES: tuple[float, ...] = tuple(
    PHENOTYPE_MIN + (PHENOTYPE_MAX - PHENOTYPE_MIN) * i / PHENOTYPE_BINS
    for i in range(PHENOTYPE_BINS + 1)
)


@dataclass(frozen=True)
class PopulationSnapshot:
    """Full per-generation distribution of each expressed phenotype.

    Counts are histogram bin counts over PHENOTYPE_BIN_EDGES, not exact
    per-integer counts: expressed B and N are continuous real values (see
    compute_expressed_phenotypes), not raw 3-bit exon values.
    """

    generation_number: int
    phase: int
    e_counts: tuple[int, ...]
    b_counts: tuple[int, ...]
    n_counts: tuple[int, ...]
