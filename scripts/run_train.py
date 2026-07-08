import argparse
import pandas as pd
from pathlib import Path
import torch

from elephant_forecast.config import load_config, Config
from elephant_forecast.data.features import FeatureBuilder
from elephant_forecast.data.dataset import build_dataloaders
from elephant_forecast.models.forecaster import ElephantForecaster
from elephant_forecast.train import Trainer
from elephant_forecast.utils.seed import set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default=None, help="Path to CSV data file")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--checkpoint", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("WARNING: CUDA not available, using CPU")
        args.device = "cpu"

    config = load_config(args.config)
    config.device = args.device

    data_path = args.data or config.data_path
    if not data_path or not Path(data_path).exists():
        raise SystemExit(f"Data file not found: {data_path}")

    set_seed(config.seed)

    print(f"Loading data from {data_path}")
    df = pd.read_csv(data_path)

    print(f"Loaded {len(df)} rows, {df[config.data.id_col].nunique()} elephants")

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
        augment=True,
        augment_rotate=config.train.augment_rotate,
        augment_jitter=config.train.augment_jitter,
        augment_subsample=config.train.augment_subsample,
        val_elephants=config.train.fixed_val_elephants,
        test_elephants=config.train.fixed_test_elephants,
        id_col=config.data.id_col,
        time_col=config.data.time_col,
        gap_threshold_hours=config.data.gap_threshold_hours,
    )

    print(f"Train: {len(train_loader.dataset)} windows, "
          f"Val: {len(val_loader.dataset)}, Test: {len(test_loader.dataset)}")

    lulc_le = fb.label_encoders.get("LULC_class")
    n_lulc = len(lulc_le.classes_) if lulc_le is not None else 2
    model = ElephantForecaster(
        d_model=config.model.d_model,
        n_layers=config.model.n_layers,
        n_heads=config.model.n_heads,
        ffn_mult=config.model.ffn_mult,
        max_seq_len=config.model.max_seq_len,
        dropout=config.model.dropout,
        n_mixtures=config.model.n_mixtures,
        n_continuous_covars=len(config.data.continuous_covars),
        n_lulc_classes=max(n_lulc, 2),
        lulc_embed_dim=config.model.lulc_embed_dim,
        time2vec_dim=config.model.time2vec_dim,
        covariate_hidden=config.model.covariate_hidden,
        fusion=config.model.fusion,
        aux_covariate_head=config.model.aux_covariate_head,
    )

    print(f"Model: {sum(p.numel() for p in model.parameters()):,} parameters")

    trainer = Trainer(model, config)
    if args.checkpoint:
        trainer.load_checkpoint(args.checkpoint)

    trainer.fit(train_loader, val_loader)

    fb.save(trainer.checkpoint_dir / "feature_builder.joblib")
    print(f"Saved feature builder to {trainer.checkpoint_dir / 'feature_builder.joblib'}")


if __name__ == "__main__":
    main()
