from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Sequence

from elephant_forecast.data.sessionize import sessionize
from elephant_forecast.data.features import FeatureBuilder
import pandas as pd
from pathlib import Path


class TrajectoryDataset(Dataset):
    def __init__(
        self,
        sessions: list[dict[str, np.ndarray]],
        context_len: int,
        horizon: int,
        augment: bool = False,
        augment_rotate: bool = True,
        augment_jitter: bool = True,
        augment_subsample: bool = True,
    ):
        self.context_len = context_len
        self.horizon = horizon
        self.augment = augment
        self.augment_rotate = augment_rotate
        self.augment_jitter = augment_jitter
        self.augment_subsample = augment_subsample

        self.windows: list[dict[str, np.ndarray]] = []
        self._build_windows(sessions)

    def _build_windows(self, sessions: list[dict[str, np.ndarray]]) -> None:
        window_size = self.context_len + self.horizon
        for sess in sessions:
            n = len(sess["displacement_in"])
            for start in range(0, n - window_size + 1):
                end = start + window_size
                ctx_end = start + self.context_len

                self.windows.append({
                    "disp_in": sess["displacement_in"][start:ctx_end].copy(),
                    "target": sess["displacement_target"][ctx_end:end].copy(),
                    "dt_in": sess["dt"][start:ctx_end].copy(),
                    "dt_out": sess["dt"][ctx_end:end].copy(),
                    "time_in": sess["time_features"][start:ctx_end].copy(),
                    "time_out": sess["time_features"][ctx_end:end].copy(),
                    "cov_in": sess["covariates"][start:ctx_end].copy(),
                    "cov_out": sess["covariates"][ctx_end:end].copy(),
                    "lulc_in": sess["lulc"][start:ctx_end].copy(),
                    "lulc_out": sess["lulc"][ctx_end:end].copy(),
                    "lat_start": sess["lat"][ctx_end - 1].copy(),
                    "lon_start": sess["lon"][ctx_end - 1].copy(),
                })

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> dict[str, np.ndarray]:
        window = self.windows[idx]
        if self.augment:
            window = self._augment(window)
        return window

    def _augment(self, window: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        window = {k: v.copy() for k, v in window.items()}

        if self.augment_rotate and np.random.rand() > 0.5:
            angle = np.random.uniform(0, 2 * np.pi)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]], dtype=np.float32)
            window["disp_in"] = window["disp_in"] @ rot.T
            window["target"] = window["target"] @ rot.T

        if self.augment_jitter and np.random.rand() > 0.5:
            jitter = np.random.normal(0, 0.01, window["disp_in"].shape).astype(np.float32)
            window["disp_in"] = window["disp_in"] + jitter

        if self.augment_subsample and np.random.rand() > 0.3:
            n = len(window["disp_in"])
            keep_idx = np.sort(np.random.choice(n, size=max(self.context_len // 2, n * 3 // 4), replace=False))
            for key in ("disp_in", "dt_in", "time_in", "cov_in", "lulc_in"):
                arr = window[key]
                new_arr = np.zeros_like(arr)
                new_arr[:len(keep_idx)] = arr[keep_idx]
                window[key] = new_arr

        return window


def collate_fn(batch: list[dict[str, np.ndarray]]) -> dict[str, torch.Tensor]:
    disp_in = torch.from_numpy(np.stack([b["disp_in"] for b in batch]))
    target = torch.from_numpy(np.stack([b["target"] for b in batch]))
    dt_in = torch.from_numpy(np.stack([b["dt_in"] for b in batch]))
    dt_out = torch.from_numpy(np.stack([b["dt_out"] for b in batch]))
    time_in = torch.from_numpy(np.stack([b["time_in"] for b in batch]))
    time_out = torch.from_numpy(np.stack([b["time_out"] for b in batch]))
    cov_in = torch.from_numpy(np.stack([b["cov_in"] for b in batch]))
    cov_out = torch.from_numpy(np.stack([b["cov_out"] for b in batch]))
    lulc_in = torch.from_numpy(np.stack([b["lulc_in"] for b in batch]))
    lulc_out = torch.from_numpy(np.stack([b["lulc_out"] for b in batch]))
    lat_start = torch.from_numpy(np.array([b["lat_start"] for b in batch]))
    lon_start = torch.from_numpy(np.array([b["lon_start"] for b in batch]))

    mask = (disp_in.sum(dim=-1) != 0).float()

    return {
        "disp_in": disp_in,
        "target": target,
        "dt_in": dt_in[:, :, None],
        "dt_out": dt_out[:, :, None],
        "time_in": time_in,
        "time_out": time_out,
        "cov_in": cov_in,
        "cov_out": cov_out,
        "lulc_in": lulc_in,
        "lulc_out": lulc_out,
        "lat_start": lat_start,
        "lon_start": lon_start,
        "mask": mask,
    }


def leave_one_out_split(
    df: pd.DataFrame,
    val_elephant: str,
    id_col: str = "individual-local-identifier",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into train, val, test."""
    all_ids = sorted(df[id_col].unique())
    val_mask = df[id_col] == val_elephant
    test_mask = ~val_mask
    if len(all_ids) > 1:
        other_elephant = [e for e in all_ids if e != val_elephant][0]
        test_mask = df[id_col] == other_elephant

    train = df[~(val_mask | test_mask)]
    val = df[val_mask]
    test = df[test_mask]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def fixed_split(
    df: pd.DataFrame,
    val_elephants: Sequence[str],
    test_elephants: Sequence[str],
    id_col: str = "individual-local-identifier",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    val_mask = df[id_col].isin(val_elephants)
    test_mask = df[id_col].isin(test_elephants)
    train = df[~(val_mask | test_mask)]
    val = df[val_mask]
    test = df[test_mask]
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_dataloaders(
    df: pd.DataFrame,
    feature_builder: FeatureBuilder,
    context_len: int,
    horizon: int,
    batch_size: int,
    num_workers: int = 0,
    augment: bool = False,
    augment_rotate: bool = True,
    augment_jitter: bool = True,
    augment_subsample: bool = True,
    val_elephants: Sequence[str] = (),
    test_elephants: Sequence[str] = (),
    id_col: str = "individual-local-identifier",
    time_col: str = "timestamp",
    gap_threshold_hours: float = 6.0,
) -> tuple[DataLoader, DataLoader, DataLoader, FeatureBuilder]:
    train_df, val_df, test_df = fixed_split(
        df, val_elephants, test_elephants, id_col=id_col,
    )

    train_sessions = sessionize(
        train_df, id_col=id_col, time_col=time_col,
        gap_threshold_hours=gap_threshold_hours,
        min_rows=context_len + horizon + 1,
    )
    fb = FeatureBuilder(
        continuous_covars=feature_builder.continuous_covars,
        categorical_covars=feature_builder.categorical_covars,
        lon_col=feature_builder.lon_col,
        lat_col=feature_builder.lat_col,
        time_col=feature_builder.time_col,
    )

    val_sessions = sessionize(
        val_df, id_col=id_col, time_col=time_col,
        gap_threshold_hours=gap_threshold_hours,
        min_rows=context_len + horizon + 1,
    )
    test_sessions = sessionize(
        test_df, id_col=id_col, time_col=time_col,
        gap_threshold_hours=gap_threshold_hours,
        min_rows=context_len + horizon + 1,
    )
    fb.fit(train_sessions, extra_sessions=val_sessions + test_sessions)

    def _sessions_to_dataset(sessions, augment, aug_rotate, aug_jitter, aug_subsample):
        feats = [fb.transform(s) for s in sessions]
        return TrajectoryDataset(
            feats, context_len, horizon,
            augment=augment,
            augment_rotate=aug_rotate,
            augment_jitter=aug_jitter,
            augment_subsample=aug_subsample,
        )

    train_ds = _sessions_to_dataset(train_sessions, augment=augment,
                                     aug_rotate=augment_rotate,
                                     aug_jitter=augment_jitter,
                                     aug_subsample=augment_subsample)
    val_ds = _sessions_to_dataset(val_sessions, augment=False,
                                   aug_rotate=False, aug_jitter=False, aug_subsample=False)
    test_ds = _sessions_to_dataset(test_sessions, augment=False,
                                    aug_rotate=False, aug_jitter=False, aug_subsample=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                             collate_fn=collate_fn, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader, fb
