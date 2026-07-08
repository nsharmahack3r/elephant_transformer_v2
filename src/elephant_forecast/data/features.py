from __future__ import annotations

from collections.abc import Sequence
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib
from pathlib import Path


class FeatureBuilder:
    def __init__(
        self,
        lon_col: str = "location-long",
        lat_col: str = "location-lat",
        time_col: str = "timestamp",
        continuous_covars: Sequence[str] = (),
        categorical_covars: Sequence[str] = (),
    ):
        self.lon_col = lon_col
        self.lat_col = lat_col
        self.time_col = time_col
        self.continuous_covars = list(continuous_covars)
        self.categorical_covars = list(categorical_covars)
        self.displacement_scaler = StandardScaler()
        self.covariate_scaler = StandardScaler()
        self.label_encoders: dict[str, LabelEncoder] = {}
        self._fitted = False

    @staticmethod
    def _cyclical_features(hour: np.ndarray, day_of_year: np.ndarray) -> np.ndarray:
        hour_sin = np.sin(2 * np.pi * hour / 24.0)
        hour_cos = np.cos(2 * np.pi * hour / 24.0)
        doy_sin = np.sin(2 * np.pi * day_of_year / 366.0)
        doy_cos = np.cos(2 * np.pi * day_of_year / 366.0)
        return np.stack([hour_sin, hour_cos, doy_sin, doy_cos], axis=-1)

    def fit(self, sessions: list[pd.DataFrame], extra_sessions: list[pd.DataFrame] | None = None) -> None:
        all_disps = []
        all_covars = []

        for sess in sessions:
            lats = sess[self.lat_col].values.astype(np.float64)
            lons = sess[self.lon_col].values.astype(np.float64)
            if len(lats) < 2:
                continue
            dlat = np.diff(lats, prepend=lats[0:1])
            dlon = np.diff(lons, prepend=lons[0:1])
            all_disps.append(np.stack([dlat, dlon], axis=-1))

            if self.continuous_covars:
                cov = sess[self.continuous_covars].values.astype(np.float64)
                all_covars.append(cov)

        if all_disps:
            self.displacement_scaler.fit(np.concatenate(all_disps, axis=0))
        if all_covars:
            self.covariate_scaler.fit(np.concatenate(all_covars, axis=0))

        all_sessions = sessions
        if extra_sessions:
            all_sessions = sessions + extra_sessions

        for col in self.categorical_covars:
            le = LabelEncoder()
            all_vals = pd.concat([s[col].astype(str) for s in all_sessions if col in s.columns])
            le.fit(all_vals)
            self.label_encoders[col] = le

        self._fitted = True

    def transform(self, session: pd.DataFrame) -> dict[str, np.ndarray]:
        """
        Returns dict with:
          displacement_in: [T, 2]  (Δlat, Δlon) at each step (autoregressive input)
          displacement_target: [T, 2] (next-step displacement)
          dt: [T] seconds since previous fix
          time_features: [T, 4] cyclical hour/day
          covariates: [T, C] standardized continuous
          lulc: [T] int-encoded LULC
          lat: [T], lon: [T] raw positions
        """
        n = len(session)
        lats = session[self.lat_col].values.astype(np.float64)
        lons = session[self.lon_col].values.astype(np.float64)
        times = pd.to_datetime(session[self.time_col])

        dlat = np.diff(lats, prepend=[lats[0]])
        dlon = np.diff(lons, prepend=[lons[0]])
        raw_disp = np.stack([dlat, dlon], axis=-1)

        dlat_next = np.diff(lats, append=[lats[-1]])
        dlon_next = np.diff(lons, append=[lons[-1]])
        raw_target = np.stack([dlat_next, dlon_next], axis=-1)

        if self._fitted:
            disp_in = self.displacement_scaler.transform(raw_disp)
            disp_target = self.displacement_scaler.transform(raw_target)
        else:
            disp_in = raw_disp
            disp_target = raw_target

        dt_seconds = np.zeros(n, dtype=np.float64)
        dt_seconds[1:] = times.diff().dt.total_seconds().fillna(0).values[1:]

        hours = times.dt.hour.values.astype(np.float64)
        doy = times.dt.dayofyear.values.astype(np.float64)
        time_feat = self._cyclical_features(hours, doy)

        covariates = np.zeros((n, 0), dtype=np.float64)
        if self.continuous_covars:
            cov_raw = session[self.continuous_covars].values.astype(np.float64)
            covariates = self.covariate_scaler.transform(cov_raw) if self._fitted else cov_raw

        lulc_encoded = np.zeros(n, dtype=np.int64)
        for col in self.categorical_covars:
            if col in session.columns and col in self.label_encoders:
                vals = session[col].astype(str).values
                le = self.label_encoders[col]
                known_mask = np.isin(vals, le.classes_)
                encoded = np.zeros(n, dtype=np.int64)
                encoded[known_mask] = le.transform(vals[known_mask])
                lulc_encoded = encoded

        return {
            "displacement_in": disp_in.astype(np.float32),
            "displacement_target": disp_target.astype(np.float32),
            "dt": dt_seconds.astype(np.float32),
            "time_features": time_feat.astype(np.float32),
            "covariates": covariates.astype(np.float32),
            "lulc": lulc_encoded.astype(np.int64),
            "lat": lats.astype(np.float32),
            "lon": lons.astype(np.float32),
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "displacement_scaler": self.displacement_scaler,
                "covariate_scaler": self.covariate_scaler,
                "label_encoders": self.label_encoders,
            },
            path,
        )

    @classmethod
    def load(cls, path: str | Path, **kwargs) -> "FeatureBuilder":
        data = joblib.load(path)
        fb = cls(**kwargs)
        fb.displacement_scaler = data["displacement_scaler"]
        fb.covariate_scaler = data["covariate_scaler"]
        fb.label_encoders = data["label_encoders"]
        fb._fitted = True
        return fb
