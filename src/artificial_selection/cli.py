import sys
from pathlib import Path
from typing import TYPE_CHECKING

import typer
from loguru import logger
from rich.console import Console
from rich.table import Table

from artificial_selection.core.domain.config import SimulationConfig
from artificial_selection.core.domain.stats import GenerationStats
from artificial_selection.core.engine import run
from artificial_selection.core.interfaces.output import IOutputWriter
from artificial_selection.core.interfaces.selection import ISelectionStrategy
from artificial_selection.impl.distribution_plot import plot_distribution_heatmap
from artificial_selection.impl.output import CsvResultWriter, JsonlEventWriter
from artificial_selection.impl.plotting import plot_results
from artificial_selection.impl.selection import RouletteWheelSelection

if TYPE_CHECKING:
    from loguru import Record

app = typer.Typer(add_completion=False)
console = Console()


def _is_event_record(record: "Record") -> bool:
    return record["extra"].get("event_type") == "generation_stats"


def configure_logging(log_dir: Path) -> None:
    """Dual-log setup: a human-readable trace sink plus a JSONL event sink.

    The event sink is added separately by JsonlEventWriter; this only wires
    the trace sinks and makes sure they never see per-generation event noise.
    """
    logger.remove()
    logger.add(sys.stderr, level="INFO", filter=lambda r: not _is_event_record(r))
    logger.add(
        log_dir / "simulation.log",
        level="DEBUG",
        filter=lambda r: not _is_event_record(r),
        rotation="5 MB",
    )


def _summary_table(history: list[GenerationStats]) -> Table:
    table = Table(title="Avian Domestication Simulation — Sample Generations")
    table.add_column("Gen", justify="right")
    table.add_column("Phase", justify="right")
    table.add_column("avg E", justify="right")
    table.add_column("avg B", justify="right")
    table.add_column("avg N", justify="right")

    last_idx = len(history) - 1
    transition_idx = next((i for i, row in enumerate(history) if row.phase == 2), None)
    candidates = {0, last_idx}
    if transition_idx is not None:
        candidates |= {
            max(transition_idx - 1, 0),
            transition_idx,
            min(transition_idx + 1, last_idx),
        }
    sample_indices = sorted(candidates)
    for idx in sample_indices:
        row = history[idx]
        table.add_row(
            str(row.generation_number),
            str(row.phase),
            f"{row.avg_e:.3f}",
            f"{row.avg_b:.3f}",
            f"{row.avg_n:.3f}",
        )
    return table


@app.command()
def main(
    config_file: Path | None = typer.Option(
        None,
        "--config",
        help="TOML file with SimulationConfig fields. Overrides all hyperparameter flags below.",
    ),
    population_size: int = typer.Option(1000, help="Agents per generation."),
    genome_length: int = typer.Option(32, help="Bits per genome."),
    mutation_rate: float = typer.Option(1.0 / 32.0, help="Per-bit mutation probability."),
    phase_1_generations: int = typer.Option(200, help="Natural selection generations."),
    phase_2_generations: int = typer.Option(200, help="Artificial selection generations."),
    k_bone: float = typer.Option(0.8, help="Calcium drawn per unit of egg output."),
    k_energy: float = typer.Option(0.6, help="Energy drawn per unit of egg output."),
    b_min: float = typer.Option(2.0, help="Bone density viability threshold (natural selection)."),
    n_min: float = typer.Option(2.0, help="Energy viability threshold (natural selection)."),
    viability_steepness: float = typer.Option(2.0, help="Steepness of the viability curves."),
    artificial_alpha: float = typer.Option(2.0, help="Artificial selection's exponent on E."),
    seed: int = typer.Option(42, help="RNG seed, ignored if --random is passed."),
    random_seed: bool = typer.Option(
        False, "--random", help="Ignore --seed and draw entropy from the OS."
    ),
    output_dir: Path = typer.Option(Path("data"), help="Directory for results.csv."),
    log_dir: Path = typer.Option(Path("logs"), help="Directory for trace/event logs."),
    plot: bool = typer.Option(True, help="Also render results.png after the run."),
) -> None:
    """Run the avian domestication GA simulation and write generation-level results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    configure_logging(log_dir)

    if config_file is not None:
        config = SimulationConfig.from_toml(config_file)
        logger.info("Loaded config from {} (other hyperparameter flags ignored)", config_file)
    else:
        config = SimulationConfig(
            population_size=population_size,
            genome_length=genome_length,
            mutation_rate=mutation_rate,
            phase_1_generations=phase_1_generations,
            phase_2_generations=phase_2_generations,
            k_bone=k_bone,
            k_energy=k_energy,
            b_min=b_min,
            n_min=n_min,
            viability_steepness=viability_steepness,
            artificial_alpha=artificial_alpha,
            seed=None if random_seed else seed,
        )
    selection_strategy: ISelectionStrategy = RouletteWheelSelection()

    csv_writer = CsvResultWriter(output_dir / "results.csv")
    jsonl_writer = JsonlEventWriter(log_dir / "events.jsonl")
    output_writers: list[IOutputWriter] = [csv_writer, jsonl_writer]

    logger.info(
        "Starting simulation: population={} generations={} seed={}",
        config.population_size,
        config.total_generations,
        config.seed,
    )
    try:
        history, snapshots = run(config, selection_strategy, output_writers)
    finally:
        csv_writer.close()
        jsonl_writer.close()

    console.print(_summary_table(history))
    console.print(f"[green]Wrote[/green] {output_dir / 'results.csv'}")
    console.print(f"[green]Wrote[/green] {log_dir / 'events.jsonl'}")

    if plot:
        plot_path = output_dir / "results.png"
        plot_results(history, config, plot_path)
        console.print(f"[green]Wrote[/green] {plot_path}")

        distribution_path = output_dir / "results_distribution.png"
        plot_distribution_heatmap(snapshots, config.phase_1_generations, distribution_path)
        console.print(f"[green]Wrote[/green] {distribution_path}")


if __name__ == "__main__":
    app()
