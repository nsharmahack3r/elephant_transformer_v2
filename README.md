# Elephant Trajectory Forecasting

Transformer-based trajectory forecaster for African elephant (Etosha) GPS
movement, conditioned on environmental covariates.

Predicts **continuous relative displacement** with a **Mixture Density Network
(MDN)** head — no H3 grid tokenization.

## Setup

```bash
uv sync
uv run python -c "import torch; print(torch.cuda.is_available())"  # must print True
```

## Data Cleaning

```bash
uv run python clean.py
```

Reads `ACTUAL_PATH` from `.env`, writes `data/cleaned.csv`, `data/sample.csv`,
and `data/dataset.json`.

## Quick Smoke Test

```bash
uv run pytest tests/ -v
uv run python scripts/run_train.py --data $env:SAMPLE_PATH --config configs/smoke.yaml
```

## Full Training

```bash
uv run python scripts/run_train.py --config configs/default.yaml
```

## Evaluation

```bash
uv run python scripts/run_eval.py --checkpoint checkpoints/best.pt --config configs/default.yaml
```

## Project Structure

```
├── configs/          # YAML configs (smoke.yaml, default.yaml)
├── scripts/          # Entry-point scripts (run_train.py, run_eval.py)
├── src/elephant_forecast/
│   ├── config.py     # Dataclass config + YAML loader
│   ├── data/         # Sessionization, feature engineering, dataset
│   ├── models/       # Transformer backbone, MDN head, forecaster
│   ├── train.py      # Training loop + checkpointing
│   ├── evaluate.py   # Metrics + rollout visualization
│   └── utils/        # Geo helpers, metrics, seed
└── tests/            # Smoke tests
```
