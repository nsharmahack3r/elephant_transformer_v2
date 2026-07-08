from __future__ import annotations

from collections.abc import Sequence
import pandas as pd


def sessionize(
    df: pd.DataFrame,
    id_col: str = "individual-local-identifier",
    time_col: str = "timestamp",
    gap_threshold_hours: float = 6.0,
    min_rows: int = 1,
) -> list[pd.DataFrame]:
    """
    Sort by individual + time, split into sessions when gap > threshold.
    Returns list of per-session DataFrames.
    """
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], utc=True)
    df = df.sort_values([id_col, time_col]).reset_index(drop=True)
    sessions: list[pd.DataFrame] = []

    for _, group in df.groupby(id_col, sort=False):
        group = group.sort_values(time_col)
        dt = group[time_col].diff()
        breaks = (dt > pd.Timedelta(hours=gap_threshold_hours)).cumsum()
        for _, sess in group.groupby(breaks, sort=False):
            if len(sess) >= min_rows:
                sessions.append(sess.reset_index(drop=True))

    return sessions
