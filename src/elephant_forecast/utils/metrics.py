from __future__ import annotations

import torch
import numpy as np
from scipy.spatial.distance import cdist


def ade(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Average Displacement Error (L2 over horizon, averaged).
    pred/target: [N, H, 2]
    """
    return float(np.mean(np.linalg.norm(pred - target, axis=-1)))


def fde(pred: np.ndarray, target: np.ndarray) -> float:
    """Final Displacement Error (L2 at horizon[-1])."""
    return float(np.mean(np.linalg.norm(pred[:, -1] - target[:, -1], axis=-1)))


def min_ade(samples: np.ndarray, targets: np.ndarray, k: int | None = None) -> float:
    """
    Best-of-N ADE.
    samples: [N, H, n_samples, 2] or [N, n_samples, H, 2]
    targets: [N, H, 2]
    For each trajectory, pick the best among n_samples samples.
    """
    N, H = targets.shape[0], targets.shape[1]
    if samples.ndim == 4:
        if samples.shape[1] == H:
            samples = samples.transpose(0, 2, 1, 3)
        n_s = min(samples.shape[1], k) if k is not None else samples.shape[1]
        samples = samples[:, :n_s]
        diff = samples - targets[:, None]
        errors = np.linalg.norm(diff, axis=-1).mean(axis=-1)
        best = errors.min(axis=1)
        return float(best.mean())
    else:
        return ade(samples, targets)


def min_fde(samples: np.ndarray, targets: np.ndarray, k: int | None = None) -> float:
    """
    Best-of-N FDE.
    samples: [N, H, n_samples, 2] or [N, n_samples, H, 2]
    targets: [N, H, 2]
    """
    N, H = targets.shape[0], targets.shape[1]
    if samples.ndim == 4:
        if samples.shape[1] == H:
            samples = samples.transpose(0, 2, 1, 3)
        n_s = min(samples.shape[1], k) if k is not None else samples.shape[1]
        samples = samples[:, :n_s]
        diff = samples[:, :, -1] - targets[:, None, -1]
        errors = np.linalg.norm(diff, axis=-1)
        best = errors.min(axis=1)
        return float(best.mean())
    else:
        return fde(samples, targets)


def hausdorff(a: np.ndarray, b: np.ndarray) -> float:
    """Hausdorff distance between two trajectories a/b: [H, 2]."""
    return float(max(np.max(np.min(cdist(a, b), axis=1)), np.max(np.min(cdist(b, a), axis=1))))


def avg_hausdorff(preds: np.ndarray, targets: np.ndarray) -> float:
    """Average Hausdorff over batch. preds/targets: [N, H, 2]."""
    vals = [hausdorff(preds[i], targets[i]) for i in range(len(preds))]
    return float(np.mean(vals))


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Dynamic Time Warping distance using scipy."""
    from scipy.spatial.distance import euclidean
    n, m = len(a), len(b)
    dtw = np.full((n + 1, m + 1), np.inf)
    dtw[0, 0] = 0
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = euclidean(a[i - 1], b[j - 1])
            dtw[i, j] = cost + min(dtw[i - 1, j], dtw[i, j - 1], dtw[i - 1, j - 1])
    return float(dtw[n, m])


def avg_dtw(preds: np.ndarray, targets: np.ndarray) -> float:
    vals = [dtw_distance(preds[i], targets[i]) for i in range(len(preds))]
    return float(np.mean(vals))


def nll_score(nll_values: torch.Tensor | np.ndarray) -> float:
    if isinstance(nll_values, torch.Tensor):
        return float(nll_values.mean().item())
    return float(np.mean(nll_values))


def pearson_cluster_correlation(
    real_trajs: np.ndarray,
    gen_trajs: np.ndarray,
    n_clusters: int = 20,
) -> float:
    """
    WildGraph-style: cluster all trajectories (real + gen) with KMeans,
    compute per-cluster count vectors, return Pearson r between them.
    """
    from sklearn.cluster import KMeans

    real_flat = real_trajs.reshape(real_trajs.shape[0], -1)
    gen_flat = gen_trajs.reshape(gen_trajs.shape[0], -1)
    all_flat = np.concatenate([real_flat, gen_flat], axis=0)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(all_flat)

    real_labels = labels[: len(real_flat)]
    gen_labels = labels[len(real_flat):]

    real_counts = np.bincount(real_labels, minlength=n_clusters) / len(real_labels)
    gen_counts = np.bincount(gen_labels, minlength=n_clusters) / len(gen_labels)

    return float(np.corrcoef(real_counts, gen_counts)[0, 1])


def chi_squared_cluster(
    real_trajs: np.ndarray,
    gen_trajs: np.ndarray,
    n_clusters: int = 20,
) -> float:
    from sklearn.cluster import KMeans

    real_flat = real_trajs.reshape(real_trajs.shape[0], -1)
    gen_flat = gen_trajs.reshape(gen_trajs.shape[0], -1)
    all_flat = np.concatenate([real_flat, gen_flat], axis=0)

    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=0)
    labels = kmeans.fit_predict(all_flat)

    real_labels = labels[: len(real_flat)]
    gen_labels = labels[len(real_flat):]

    real_counts = np.bincount(real_labels, minlength=n_clusters)
    gen_counts = np.bincount(gen_labels, minlength=n_clusters)

    expected = real_counts * len(gen_labels) / len(real_labels)
    expected = np.where(expected == 0, 1, expected)
    return float(np.sum((gen_counts - expected) ** 2 / expected))


def calibration_coverage(
    pred_mean: np.ndarray,
    pred_std: np.ndarray,
    target: np.ndarray,
    n_intervals: int = 5,
) -> dict[str, float]:
    """
    Empirical coverage at {0.5, 0.68, 0.9, 0.95, 0.99}.
    pred_mean/std: [N, H, 2], target: [N, H, 2].
    """
    from scipy.stats import norm

    z_scores = np.abs(target - pred_mean) / (pred_std + 1e-8)
    probs = 2 * norm.cdf(z_scores) - 1

    results = {}
    for level in [0.5, 0.68, 0.9, 0.95, 0.99]:
        covered = (probs <= level).all(axis=-1).mean()
        results[f"coverage_{int(level * 100)}"] = float(covered)
    return results
