from __future__ import annotations

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Optional, Sequence
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import json

from elephant_forecast.config import Config
from elephant_forecast.utils.geo import reachability_clip
from elephant_forecast.utils import metrics
from elephant_forecast.data.features import FeatureBuilder


@torch.no_grad()
def rollout(
    model: nn.Module,
    batch: dict[str, torch.Tensor],
    n_samples: int = 50,
    max_speed_mps: float = 7.0,
    lat_scale: float = 1.0,
    lon_scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Roll out autoregressively. Returns:
      samples: [B, H, n_samples, 2]
      modes: [B, H, 2]
      sigma: [B, H, 2]
      target: [B, H, 2]
    """
    device = next(model.parameters()).device
    batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
    B, _, _ = batch["disp_in"].shape
    H = batch["target"].shape[1]

    output = model(batch, teacher_forcing=False)
    samples = model.mdn.sample(output["pi"], output["mu"], output["sigma"], output["rho"], n_samples=n_samples)
    modes = model.mdn.mode(output["pi"], output["mu"])

    for t in range(H):
        dt = batch["dt_out"][:, t, 0]
        for s in range(n_samples):
            s_dlat = samples[:, t, s, 0]
            s_dlon = samples[:, t, s, 1]
            dlat, dlon = reachability_clip(s_dlat, s_dlon, dt, max_speed_mps, lat_scale, lon_scale)
            samples[:, t, s, 0] = dlat
            samples[:, t, s, 1] = dlon

        m_dlat = modes[:, t, 0]
        m_dlon = modes[:, t, 1]
        dlat, dlon = reachability_clip(m_dlat, m_dlon, dt, max_speed_mps, lat_scale, lon_scale)
        modes[:, t, 0] = dlat
        modes[:, t, 1] = dlon

    sigma_mean = output["sigma"].mean(dim=2)

    return (
        samples.cpu().numpy(),
        modes.cpu().numpy(),
        sigma_mean.cpu().numpy(),
        batch["target"].cpu().numpy(),
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    feature_builder: FeatureBuilder,
    config: Config,
    output_dir: Path,
    lat_scale: float = 1.0,
    lon_scale: float = 1.0,
) -> dict[str, float]:
    model.eval()
    device = next(model.parameters()).device

    all_targets = []
    all_modes = []
    all_samples = []
    all_sigmas = []
    all_nlls = []

    for batch in tqdm(loader, desc="Evaluating"):
        batch = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
        output = model(batch, teacher_forcing=False)
        losses = model.compute_loss(output, batch)
        all_nlls.append(losses["nll"].item())

        samples, modes, sigmas, targets = rollout(
            model, {k: v for k, v in batch.items()},
            n_samples=config.eval.n_samples,
            max_speed_mps=config.eval.max_speed_mps,
            lat_scale=lat_scale,
            lon_scale=lon_scale,
        )

        for i in range(len(targets)):
            all_targets.append(targets[i])
            all_modes.append(modes[i])
            all_samples.append(samples[i])
            all_sigmas.append(sigmas[i])

    all_targets = np.array(all_targets)
    all_modes = np.array(all_modes)
    all_samples = np.array(all_samples)
    all_sigmas = np.array(all_sigmas)

    results = {}

    results["nll"] = float(np.mean(all_nlls))
    results["ade"] = metrics.ade(all_modes, all_targets)
    results["fde"] = metrics.fde(all_modes, all_targets)
    results["min_ade"] = metrics.min_ade(all_samples, all_targets, k=config.eval.best_of_k)
    results["min_fde"] = metrics.min_fde(all_samples, all_targets, k=config.eval.best_of_k)

    if len(all_modes) >= 2:
        results["hausdorff"] = metrics.avg_hausdorff(all_modes, all_targets)
        results["dtw"] = metrics.avg_dtw(all_modes, all_targets)

    if len(all_modes) >= config.eval.n_clusters:
        results["pearson_cluster"] = metrics.pearson_cluster_correlation(
            all_targets, all_modes, n_clusters=config.eval.n_clusters,
        )
        results["chi_squared"] = metrics.chi_squared_cluster(
            all_targets, all_modes, n_clusters=config.eval.n_clusters,
        )

    calib = metrics.calibration_coverage(all_modes, all_sigmas, all_targets)
    results.update(calib)

    _plot_examples(all_modes, all_targets, all_samples, all_sigmas, output_dir,
                   n_examples=config.eval.plot_n_examples)

    return results


def _plot_examples(
    modes: np.ndarray,
    targets: np.ndarray,
    samples: np.ndarray,
    sigmas: np.ndarray,
    output_dir: Path,
    n_examples: int = 5,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    n = min(n_examples, len(modes))

    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, 1)

    for i in range(n):
        ax1 = axes[0, i]
        true_horizon = targets[i]
        pred_horizon = modes[i]

        true_path = np.concatenate([np.zeros((1, 2)), np.cumsum(true_horizon, axis=0)], axis=0)
        pred_path = np.concatenate([np.zeros((1, 2)), np.cumsum(pred_horizon, axis=0)], axis=0)

        ax1.plot(true_path[:, 0], true_path[:, 1], "b-o", label="True", alpha=0.7)
        ax1.plot(pred_path[:, 0], pred_path[:, 1], "r-o", label="Pred", alpha=0.7)
        ax1.set_title(f"Example {i + 1}")
        ax1.legend()
        ax1.set_aspect("equal")

        ax2 = axes[1, i]
        h = np.arange(len(true_horizon))
        ax2.plot(h, true_horizon[:, 0], "b-", label="True dlat")
        ax2.plot(h, pred_horizon[:, 0], "r-", label="Pred dlat")

        if samples is not None and len(samples) > i:
            for s in range(min(10, samples.shape[1])):
                ax2.plot(h, samples[i, :, s, 0], "r-", alpha=0.1)

        ax2.fill_between(
            h,
            pred_horizon[:, 0] - 2 * sigmas[i, :, 0],
            pred_horizon[:, 0] + 2 * sigmas[i, :, 0],
            alpha=0.2, color="red",
        )
        ax2.legend()
        ax2.set_xlabel("Step")

    plt.tight_layout()
    fig.savefig(output_dir / "rollout_examples.png", dpi=100)
    plt.close(fig)
