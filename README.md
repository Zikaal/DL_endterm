# fin-ts-forecast-uncertainty

Time-series forecasting for financial data (Track 2): GRU/LSTM (+ temporal attention) vs Transformer, with uncertainty estimation via deep ensembles (and optional MC Dropout).

## 1) Setup

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -r requirements.txt
```

## 2) Quick start

Train a single model:

```bash
python scripts/run_train.py --config configs/gru.yaml
python scripts/run_train.py --config configs/transformer.yaml
```

Train an ensemble (K models) and evaluate uncertainty:

```bash
python scripts/run_ensemble.py --config configs/gru.yaml --k 5
```

Evaluate a saved checkpoint:

```bash
python scripts/run_eval.py --checkpoint results/checkpoints/<run_name>.pt
```

Generate plots after training:

```bash
python scripts/make_plots.py --run results/logs/<run_name>
```

## 3) Outputs

- `data/raw/` cached CSV from Yahoo Finance
- `data/processed/` cached numpy arrays + split metadata
- `results/checkpoints/` model checkpoints (`.pt`)
- `results/logs/` training logs (`.csv`) + metrics (`.json`) + predictions (`.npz`)
- `results/figures/` plots (`.png`)

## Notes

- This project uses **time-based splitting** (no leakage).
- Targets are **log returns** for multiple horizons (default: 1, 5, 20 days).
- Uncertainty:
  - **Deep ensembles**: variability across multiple independently trained models.
  - Optional **MC Dropout**: multiple stochastic forward passes with dropout enabled.

This repository is intended for course projects; it is **not investment advice**.


## 4) Ablations (required for Track 2)

GRU with vs without attention:

```bash
python -m scripts.run_train --config configs/ablations/gru_attention.yaml
python -m scripts.run_train --config configs/ablations/gru_no_attention.yaml
```

Dropout ablation:

```bash
python -m scripts.run_train --config configs/ablations/gru_dropout0.yaml
python -m scripts.run_train --config configs/ablations/gru_dropout02.yaml
```

Weight decay ablation:

```bash
python -m scripts.run_train --config configs/ablations/gru_wd0.yaml
python -m scripts.run_train --config configs/ablations/gru_wd1e4.yaml
```

Ensemble size (uncertainty) ablation:

```bash
python -m scripts.run_ensemble --config configs/gru.yaml --k 1
python -m scripts.run_ensemble --config configs/gru.yaml --k 5
```

Naive baselines:

```bash
python -m scripts.run_baselines --config configs/gru.yaml
```

Optional: run everything:

```bash
python -m scripts.run_ablation_suite
```
