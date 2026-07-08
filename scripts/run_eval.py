import argparse
import pandas as pd
from pathlib import Path
import torch
import json

from elephant_forecast.config import load_config
from elephant_forecast.data.features import FeatureBuilder
from elephant_forecast.data.dataset import build_dataloaders
from elephant_forecast.models.forecaster import ElephantForecaster
from elephant_forecast.evaluate import evaluate
from elephant_forecast.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="eval_output")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"

    config = load_config(args.config)
    config.device = args.device

    data_path = args.data or config.data_path
    if not data_path or not Path(data_path).exists():
        raise SystemExit(f"Data file not found: {data_path}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(config.seed)

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df)} rows")

    fb = FeatureBuilder(
        lon_col=config.data.lon_col,
        lat_col=config.data.lat_col,
        time_col=config.data.time_col,
        continuous_covars=config.data.continuous_covars,
        categorical_covars=config.data.categorical_covars,
    )

    train_loader, val_loader, test_loader, fb = build_dataloaders(
        df, fb,
        context_len=config.data.context_len,
        horizon=config.data.horizon,
        batch_size=config.train.batch_size,
        num_workers=config.train.num_workers,
        augment=False,
        val_elephants=config.train.fixed_val_elephants,
        test_elephants=config.train.fixed_test_elephants,
        id_col=config.data.id_col,
        time_col=config.data.time_col,
        gap_threshold_hours=config.data.gap_threshold_hours,
    )

    ckpt = torch.load(args.checkpoint, map_location=config.device)
    n_lulc_ckpt = ckpt["model_state_dict"]["lulc_embed.weight"].shape[0]
    n_continuous_ckpt = ckpt["model_state_dict"]["covar_mlp.0.weight"].shape[1]

    model = ElephantForecaster(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        ffn_mult=config.model.ffn_mult,
        dropout=config.model.dropout,
        n_mixtures=config.model.n_mixtures,
        n_continuous_covars=n_continuous_ckpt,
        n_lulc_classes=n_lulc_ckpt,
        lulc_embed_dim=config.model.lulc_embed_dim,
        time2vec_dim=config.model.time2vec_dim,
        covariate_hidden=config.model.covariate_hidden,
        fusion=config.model.fusion,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(config.device)
    model.eval()

    lat_scale = fb.displacement_scaler.scale_[0]
    lon_scale = fb.displacement_scaler.scale_[1]

    results = evaluate(
        model, test_loader, fb, config, output_dir,
        lat_scale=float(1.0 / lat_scale) if lat_scale != 0 else 1.0,
        lon_scale=float(1.0 / lon_scale) if lon_scale != 0 else 1.0,
    )

    print("\n=== Evaluation Results ===")
    for k, v in results.items():
        print(f"  {k}: {v:.4f}")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(json.dumps(results, indent=2))
    print(f"\nMetrics saved to {metrics_path}")
    print(f"Plots saved to {output_dir}")


if __name__ == "__main__":
    main()
