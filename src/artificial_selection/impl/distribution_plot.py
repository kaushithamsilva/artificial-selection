"""Renders per-generation phenotype-distribution heatmaps.

Expressed B and N are continuous (substrate depletion: capacity minus a
real-valued extraction), so this histograms over PHENOTYPE_BIN_EDGES rather
than counting exact integer values. This shows what the line chart's
population-average can hide: whether the population is fixed on a single
value or still spread across several.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

from artificial_selection.core.domain.distribution import (
    PHENOTYPE_MAX,
    PHENOTYPE_MIN,
    PopulationSnapshot,
)

SURFACE = "#fcfcfb"
MUTED_INK = "#898781"
PRIMARY_INK = "#0b0b0b"
BASELINE = "#c3c2b7"

# Same fixed categorical hues as the line chart, reused here as light->dark
# sequential ramps so the two charts share one visual language.
COLOR_E = "#2a78d6"
COLOR_B = "#eb6834"
COLOR_N = "#1baf7a"


def _sequential_cmap(hue: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(f"seq_{hue}", [SURFACE, hue])


def plot_distribution_heatmap(
    snapshots: list[PopulationSnapshot],
    phase_1_generations: int,
    output_path: Path,
) -> None:
    generations = [s.generation_number for s in snapshots]
    last_gen = max(generations)
    population_size = sum(snapshots[0].e_counts)

    traits = [
        ("Egg yield (E)", [s.e_counts for s in snapshots], COLOR_E),
        ("Bone density (B)", [s.b_counts for s in snapshots], COLOR_B),
        ("Energy (N)", [s.n_counts for s in snapshots], COLOR_N),
    ]

    fig, axes = plt.subplots(3, 1, figsize=(10.5, 8.5), dpi=150, sharex=True)
    fig.patch.set_facecolor(SURFACE)

    for ax, (label, counts_over_time, color) in zip(axes, traits, strict=True):
        matrix = np.array(counts_over_time, dtype=np.float64).T / population_size
        image = ax.imshow(
            matrix,
            aspect="auto",
            origin="lower",
            extent=(min(generations) - 0.5, last_gen + 0.5, PHENOTYPE_MIN, PHENOTYPE_MAX),
            cmap=_sequential_cmap(color),
            vmin=0.0,
            vmax=1.0,
        )
        ax.set_facecolor(SURFACE)
        ax.axvline(phase_1_generations, color=MUTED_INK, linestyle="--", linewidth=1)
        ax.set_yticks(range(int(PHENOTYPE_MIN), int(PHENOTYPE_MAX) + 1))
        ax.set_ylabel(label, color=color, fontsize=10, fontweight="bold")
        ax.tick_params(colors=MUTED_INK, labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(BASELINE)

        colorbar = fig.colorbar(image, ax=ax, pad=0.012, fraction=0.035)
        colorbar.set_label("Fraction of\npopulation", color=MUTED_INK, fontsize=7.5)
        colorbar.ax.tick_params(colors=MUTED_INK, labelsize=7)

    axes[-1].set_xlabel("Generation", color=MUTED_INK, fontsize=10)
    fig.suptitle(
        "Population Distribution Over Time (expressed phenotype, 0-7)",
        color=PRIMARY_INK,
        fontsize=14,
        fontweight="bold",
        x=0.44,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
