from __future__ import annotations

import pandas as pd
import numpy as np
import torch
from pathlib import Path
import os
from dotenv import load_dotenv

from elephant_forecast.data.sessionize import sessionize
from elephant_forecast.data.features import FeatureBuilder
from elephant_forecast.data.dataset import TrajectoryDataset, collate_fn
from elephant_forecast.models.forecaster import ElephantForecaster


def test_sessionize():
    load_dotenv()
    sample_path = os.getenv("SAMPLE_PATH")
    if not sample_path or not Path(sample_path).exists():
        pytest.skip("SAMPLE_PATH not available")
    df = pd.read_csv(sample_path)
    sessions = sessionize(
        df,
        id_col="individual-local-identifier",
        time_col="timestamp",
        gap_threshold_hours=6.0,
        min_rows=3,
    )
    assert len(sessions) > 0
    for s in sessions:
        assert "location-lat" in s.columns
        assert "location-long" in s.columns


def test_feature_builder():
    load_dotenv()
    sample_path = os.getenv("SAMPLE_PATH")
    if not sample_path or not Path(sample_path).exists():
        pytest.skip("SAMPLE_PATH not available")
    df = pd.read_csv(sample_path)
    sessions = sessionize(
        df,
        id_col="individual-local-identifier",
        time_col="timestamp",
        gap_threshold_hours=6.0,
        min_rows=10,
    )
    fb = FeatureBuilder(
        continuous_covars=["aspect_deg", "elevation_m", "slope_deg", "EVI", "LST_celsius", "NDVI", "human_settle"],
        categorical_covars=["LULC_class"],
    )
    fb.fit(sessions)
    features = fb.transform(sessions[0])
    assert features["displacement_in"].shape[1] == 2
    assert features["displacement_target"].shape[1] == 2
    assert features["dt"].ndim == 1
    assert features["time_features"].shape[1] == 4
    assert features["covariates"].shape[1] == 7
    assert np.isfinite(features["displacement_in"]).all()


def test_dataset():
    load_dotenv()
    sample_path = os.getenv("SAMPLE_PATH")
    if not sample_path or not Path(sample_path).exists():
        pytest.skip("SAMPLE_PATH not available")
    df = pd.read_csv(sample_path)
    sessions = sessionize(
        df,
        id_col="individual-local-identifier",
        time_col="timestamp",
        gap_threshold_hours=6.0,
        min_rows=20,
    )
    fb = FeatureBuilder(
        continuous_covars=["aspect_deg", "elevation_m", "slope_deg", "EVI", "LST_celsius", "NDVI", "human_settle"],
        categorical_covars=["LULC_class"],
    )
    fb.fit(sessions)
    features = [fb.transform(s) for s in sessions]

    ds = TrajectoryDataset(features, context_len=8, horizon=4)
    assert len(ds) > 0

    item = ds[0]
    assert item["disp_in"].shape == (8, 2)
    assert item["target"].shape == (4, 2)
    assert np.isfinite(item["disp_in"]).all()
    assert np.isfinite(item["target"]).all()

    # Verify no target leakage: disp_in[t] is displacement at time t,
    # target[t] is displacement t -> t+1 for horizon steps.
    # The window's disp_in comes from positions [start, start+context_len),
    # target from positions [start+context_len, start+context_len+horizon).
    # So disp_in[-1] is the displacement at the last context step,
    # NOT equal to target[0].


def test_collate():
    load_dotenv()
    sample_path = os.getenv("SAMPLE_PATH")
    if not sample_path or not Path(sample_path).exists():
        pytest.skip("SAMPLE_PATH not available")
    df = pd.read_csv(sample_path)
    sessions = sessionize(df, min_rows=20)
    fb = FeatureBuilder()
    fb.fit(sessions)
    features = [fb.transform(s) for s in sessions]
    ds = TrajectoryDataset(features, context_len=8, horizon=4)
    batch = collate_fn([ds[i] for i in range(min(4, len(ds)))])
    assert batch["disp_in"].shape[0] == min(4, len(ds))
    assert torch.isfinite(batch["disp_in"]).all()


def test_model_forward():
    model = ElephantForecaster(
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.1,
        n_mixtures=3,
        n_continuous_covars=7,
        n_lulc_classes=10,
    )
    B, L, H = 4, 8, 4
    batch = {
        "disp_in": torch.randn(B, L, 2),
        "target": torch.randn(B, H, 2),
        "dt_in": torch.rand(B, L, 1),
        "dt_out": torch.rand(B, H, 1),
        "time_in": torch.randn(B, L, 4),
        "time_out": torch.randn(B, H, 4),
        "cov_in": torch.randn(B, L, 7),
        "cov_out": torch.randn(B, H, 7),
        "lulc_in": torch.randint(0, 5, (B, L)),
        "lulc_out": torch.randint(0, 5, (B, H)),
        "mask": torch.ones(B, L),
    }
    with torch.no_grad():
        output = model(batch, teacher_forcing=True)
        losses = model.compute_loss(output, batch)
    assert torch.isfinite(losses["loss"])
    assert losses["nll"] > 0


def test_overfit_batch():
    torch.manual_seed(42)
    model = ElephantForecaster(
        d_model=64,
        n_layers=2,
        n_heads=4,
        dropout=0.0,
        n_mixtures=3,
        n_continuous_covars=7,
        n_lulc_classes=10,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    B, L, H = 4, 8, 2
    batch = {
        "disp_in": torch.randn(B, L, 2),
        "target": torch.randn(B, H, 2),
        "dt_in": torch.rand(B, L, 1),
        "dt_out": torch.rand(B, H, 1),
        "time_in": torch.randn(B, L, 4),
        "time_out": torch.randn(B, H, 4),
        "cov_in": torch.randn(B, L, 7),
        "cov_out": torch.randn(B, H, 7),
        "lulc_in": torch.randint(0, 5, (B, L)),
        "lulc_out": torch.randint(0, 5, (B, H)),
        "mask": torch.ones(B, L),
    }
    losses = []
    for _ in range(200):
        output = model(batch, teacher_forcing=True)
        loss_dict = model.compute_loss(output, batch)
        optimizer.zero_grad()
        loss_dict["loss"].backward()
        optimizer.step()
        losses.append(loss_dict["loss"].item())
    assert losses[-1] < losses[0] * 0.5, f"NLL did not decrease: {losses[0]:.4f} -> {losses[-1]:.4f}"


def test_mdn_sample():
    from elephant_forecast.models.mdn import MixtureDensityHead
    mdn = MixtureDensityHead(d_model=64, n_mixtures=3)
    x = torch.randn(4, 64)
    pi, mu, sigma, rho = mdn(x)
    samples = mdn.sample(pi, mu, sigma, rho, n_samples=5)
    assert samples.shape == (4, 5, 2)
    assert torch.isfinite(samples).all()
