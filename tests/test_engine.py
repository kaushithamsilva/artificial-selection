from artificial_selection.core.domain.config import SimulationConfig
from artificial_selection.core.domain.distribution import PHENOTYPE_BIN_EDGES
from artificial_selection.core.engine import run
from artificial_selection.core.interfaces.selection import ISelectionStrategy
from artificial_selection.impl.selection import RouletteWheelSelection


def _small_config(**overrides: object) -> SimulationConfig:
    defaults: dict[str, object] = dict(
        population_size=50,
        genome_length=32,
        mutation_rate=1.0 / 32.0,
        phase_1_generations=5,
        phase_2_generations=5,
        seed=7,
    )
    defaults.update(overrides)
    return SimulationConfig(**defaults)  # type: ignore[arg-type]


def test_run_produces_one_stats_row_per_generation() -> None:
    config = _small_config()
    strategy: ISelectionStrategy = RouletteWheelSelection()

    history, snapshots = run(config, strategy, output_writers=[])

    assert len(history) == config.total_generations
    assert len(snapshots) == config.total_generations
    assert [row.generation_number for row in history] == list(range(config.total_generations))


def test_run_is_deterministic_given_a_seed() -> None:
    strategy_a: ISelectionStrategy = RouletteWheelSelection()
    strategy_b: ISelectionStrategy = RouletteWheelSelection()

    history_a, _ = run(_small_config(seed=123), strategy_a, output_writers=[])
    history_b, _ = run(_small_config(seed=123), strategy_b, output_writers=[])

    assert [row.avg_e for row in history_a] == [row.avg_e for row in history_b]


def test_phase_switches_from_natural_to_artificial_selection() -> None:
    config = _small_config(phase_1_generations=5, phase_2_generations=5)
    strategy: ISelectionStrategy = RouletteWheelSelection()

    history, snapshots = run(config, strategy, output_writers=[])

    assert all(row.phase == 1 for row in history[:5])
    assert all(row.phase == 2 for row in history[5:])
    assert all(s.phase == 1 for s in snapshots[:5])
    assert all(s.phase == 2 for s in snapshots[5:])


def test_population_snapshot_counts_sum_to_population_size() -> None:
    config = _small_config(population_size=64)
    strategy: ISelectionStrategy = RouletteWheelSelection()

    _, snapshots = run(config, strategy, output_writers=[])

    for snapshot in snapshots:
        assert sum(snapshot.e_counts) == 64
        assert sum(snapshot.b_counts) == 64
        assert sum(snapshot.n_counts) == 64
        assert len(snapshot.e_counts) == len(PHENOTYPE_BIN_EDGES) - 1
