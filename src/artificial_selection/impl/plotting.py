"""Renders the generation time-series as a static PNG chart.

Follows the project's data-viz conventions: fixed categorical hue order
(never cycled), a legend plus direct end-of-line labels since this is <=4
series, recessive gridlines/axes, and one shared axis (all three series share
the same 0-7 phenotype scale, so no dual-axis trickery is needed).
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

from artificial_selection.core.domain.config import SimulationConfig
from artificial_selection.core.domain.stats import GenerationStats

SURFACE = "#fcfcfb"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"
PRIMARY_INK = "#0b0b0b"
MUTED_INK = "#898781"

# Fixed categorical order (slots 1-3 of the validated default palette).
COLOR_E = "#2a78d6"  # blue
COLOR_B = "#eb6834"  # orange
COLOR_N = "#1baf7a"  # aqua

# Very light tints of the phase-1/phase-2 marker hue, for the background bands.
PHASE_1_BAND = "#f2f6fb"
PHASE_2_BAND = "#fdf2ec"

def _depletion_formula(k_bone: float, k_energy: float) -> str:
    return (
        rf"$B_{{exp}}=\max(0,\,B_{{cap}}-{k_bone:g}E)$"
        rf"    $N_{{exp}}=\max(0,\,N_{{cap}}-{k_energy:g}E)$"
    )


def _natural_formula(b_min: float, n_min: float) -> str:
    return rf"$F_{{natural}} = E\cdot\sigma(B_{{exp}}-{b_min:g})\cdot\sigma(N_{{exp}}-{n_min:g})$"


def _artificial_formula(alpha: float) -> str:
    return rf"$F_{{artificial}} = E^{{{alpha:g}}}$"


def _declutter_label_positions(
    values_by_label: dict[str, float], min_gap: float
) -> dict[str, float]:
    """Push direct end-of-line labels apart vertically so they don't overlap."""
    ordered = sorted(values_by_label.items(), key=lambda item: item[1])
    adjusted = [ordered[0][1]]
    for _, value in ordered[1:]:
        adjusted.append(max(value, adjusted[-1] + min_gap))
    return {label: y for (label, _), y in zip(ordered, adjusted, strict=True)}


def plot_results(
    history: list[GenerationStats],
    config: SimulationConfig,
    output_path: Path,
) -> None:
    generations = [row.generation_number for row in history]
    avg_e = [row.avg_e for row in history]
    avg_b = [row.avg_b for row in history]
    avg_n = [row.avg_n for row in history]
    last_gen = max(generations)
    phase_1_generations = config.phase_1_generations

    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    data_ceiling = 7.0
    y_top = 7.6
    ax.axvspan(0, phase_1_generations, color=PHASE_1_BAND, zorder=0, lw=0)
    ax.axvspan(phase_1_generations, last_gen, color=PHASE_2_BAND, zorder=0, lw=0)
    ax.axvline(phase_1_generations, color=MUTED_INK, linestyle="--", linewidth=1, zorder=1)

    series = [
        ("Egg yield (E)", avg_e, COLOR_E),
        ("Bone density (B)", avg_b, COLOR_B),
        ("Energy (N)", avg_n, COLOR_N),
    ]
    label_y = _declutter_label_positions(
        {label: values[-1] for label, values, _ in series}, min_gap=0.55
    )
    for label, values, color in series:
        ax.plot(
            generations,
            values,
            color=color,
            linewidth=2.25,
            label=label,
            zorder=3,
            solid_capstyle="round",
            solid_joinstyle="round",
        )
        ax.text(
            generations[-1] + last_gen * 0.015,
            label_y[label],
            label,
            color=color,
            fontsize=9.5,
            va="center",
            ha="left",
            fontweight="bold",
        )

    ax.set_xlim(0, last_gen * 1.2)
    ax.set_ylim(0, y_top)
    ax.set_yticks(range(0, int(data_ceiling) + 1))
    ax.set_xlabel("Generation", color=MUTED_INK, fontsize=10)
    ax.set_ylabel("Population-average phenotype value (0-7)", color=MUTED_INK, fontsize=10)

    depletion_formula = _depletion_formula(config.k_bone, config.k_energy)
    natural_formula = _natural_formula(config.b_min, config.n_min)
    artificial_formula = _artificial_formula(config.artificial_alpha)

    fig.suptitle(
        "Artificial Selection Decouples Egg Yield from Bone Density and Energy",
        color=PRIMARY_INK,
        fontsize=15,
        fontweight="bold",
        x=0.5,
        y=0.985,
    )
    # Formulas live in this fixed header row (not floating inside the plot at
    # data coordinates) so they can never collide with each other or with the
    # data lines, regardless of how narrow either phase's generation span is.
    ax.set_title(
        f"population={config.population_size}  ·  mutation rate={config.mutation_rate:.4f}"
        f"  ·  seed={config.seed}\n"
        f"Depletion: {depletion_formula}\n"
        f"Natural selection: {natural_formula}      Artificial selection: {artificial_formula}",
        color=MUTED_INK,
        fontsize=9.5,
        pad=14,
        loc="center",
    )

    ax.grid(axis="y", color=GRIDLINE, linewidth=0.8, zorder=0.5)
    ax.set_axisbelow(True)
    for spine_name in ("top", "right"):
        ax.spines[spine_name].set_visible(False)
    for spine_name in ("left", "bottom"):
        ax.spines[spine_name].set_color(BASELINE)
    ax.tick_params(colors=MUTED_INK, labelsize=9)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        frameon=False,
        fontsize=9.5,
        labelcolor=MUTED_INK,
        handlelength=1.8,
        columnspacing=1.8,
    )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(output_path, facecolor=SURFACE)
    plt.close(fig)
