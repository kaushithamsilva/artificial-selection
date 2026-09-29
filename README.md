# Artificial Selection

Accompanying note: [kaushitha.xyz/notes/artificial-selection](https://kaushitha.xyz/notes/artificial-selection)

A genetic-algorithm simulation of avian domestication: what happens to a bird
population when selection pressure switches from natural selection (survival
in the wild) to artificial selection (a farmer breeding purely for egg
yield).

Each simulated bird is a 32-bit genome encoding three traits, each a 3-bit
exon (0-7): egg yield potential (`E`), bone density capacity (`B`), and
energy capacity (`N`); the remaining 23 bits are non-coding. Laying eggs
physiologically depletes bone density and energy (substrate depletion:
`B_expressed = max(0, B - k_bone * E)`, and similarly for `N`) -- the
expressed phenotype, not the raw genetic capacity, is what fitness,
logging, and plotting all use.

- **Natural selection**: `F = E * sigmoid(B_expressed - b_min) * sigmoid(N_expressed - n_min)`
  -- egg yield only counts if the bird stays viable.
- **Artificial selection**: `F = E ^ artificial_alpha` -- the farmer selects
  on yield alone; bone density and energy aren't inputs to this fitness
  function at all.

Reproduction uses single-point crossover, per-bit mutation, and
fitness-proportionate (roulette-wheel) selection.

## Running it

```bash
uv run run-simulation --config configs/default.toml
```

This writes `data/results.csv`, `data/results.png` (trajectory chart),
`data/results_distribution.png` (per-generation phenotype distribution
heatmap), and `logs/events.jsonl` (structured per-generation event log).

Run without `--config` to use CLI flags directly (`--population-size`,
`--mutation-rate`, `--k-bone`, `--artificial-alpha`, etc.) -- see
`uv run run-simulation --help` for the full list. Example configs in
`configs/` demonstrate different regimes (`quick_test.toml` for a fast
smoke test, `high_mutation.toml` for exploring drift under stronger
mutation).

## Development

```bash
uv run ruff check .      # lint
uv run mypy               # strict type checking
uv run pytest -q          # tests
```

## Project layout

```
src/artificial_selection/
├── core/
│   ├── domain/       # Pure dataclasses: SimulationConfig, GenerationStats, PopulationSnapshot
│   ├── interfaces/   # Protocols: ISelectionStrategy, IOutputWriter
│   ├── genetics.py   # Genome decoding, substrate depletion, fitness, crossover, mutation
│   └── engine.py      # The generation-loop orchestrator
├── impl/              # Concrete implementations: roulette-wheel selection, CSV/JSONL writers, plots
└── cli.py             # Typer entry point (`run-simulation`)
```
