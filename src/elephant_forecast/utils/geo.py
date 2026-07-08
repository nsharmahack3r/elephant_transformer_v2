from __future__ import annotations

import math
import torch
import numpy as np


EARTH_RADIUS_M = 6_371_000.0


def haversine_m(
    lat1: torch.Tensor | np.ndarray,
    lon1: torch.Tensor | np.ndarray,
    lat2: torch.Tensor | np.ndarray,
    lon2: torch.Tensor | np.ndarray,
) -> torch.Tensor | np.ndarray:
    is_torch = isinstance(lat1, torch.Tensor)
    if is_torch:
        sin, cos, arcsin, clip = torch.sin, torch.cos, torch.arcsin, torch.clamp
        radians = lambda x: torch.deg2rad(x)
    else:
        sin, cos, arcsin, clip = np.sin, np.cos, np.arcsin, np.clip
        radians = np.radians

    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2.0) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_RADIUS_M * arcsin(clip(a.sqrt() if is_torch else np.sqrt(a), 0.0, 1.0))


def displacement_to_latlon(
    start_lat: torch.Tensor,
    start_lon: torch.Tensor,
    dlat: torch.Tensor,
    dlon: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    return start_lat + dlat, start_lon + dlon


def reachability_mask(
    dlat: torch.Tensor,
    dlon: torch.Tensor,
    dt: torch.Tensor,
    max_speed_mps: float,
    lat_scale: float = 1.0,
    lon_scale: float = 1.0,
) -> torch.Tensor:
    """
    Returns a boolean mask: True where displacement magnitude <= max_speed * dt.
    All inputs in standardized (scaled) space. dt in seconds.
    """
    deg_per_meter = 1.0 / 111_320.0
    max_displacement_deg = max_speed_mps * dt * deg_per_meter
    max_dlat = max_displacement_deg / lat_scale
    max_dlon = max_displacement_deg / lon_scale
    return (dlat.abs() <= max_dlat) & (dlon.abs() <= max_dlon)


def reachability_clip(
    dlat: torch.Tensor,
    dlon: torch.Tensor,
    dt: torch.Tensor,
    max_speed_mps: float,
    lat_scale: float = 1.0,
    lon_scale: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    deg_per_meter = 1.0 / 111_320.0
    max_displacement_deg = max_speed_mps * dt * deg_per_meter
    max_dlat = max_displacement_deg / lat_scale
    max_dlon = max_displacement_deg / lon_scale
    dlat = torch.clamp(dlat, -max_dlat, max_dlat)
    dlon = torch.clamp(dlon, -max_dlon, max_dlon)
    return dlat, dlon
