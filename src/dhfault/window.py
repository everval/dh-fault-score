"""Score one observation window end to end."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .beast import detect_beast
from .score import UNFAVOURABLE, decision_scores


@dataclass
class WindowResult:
    """The decision scores of one window and what they found."""

    scores: dict
    """channel -> per-day decision scores over the whole window."""
    changepoints: pd.DataFrame
    """Every scored changepoint: channel, date, position, days_before_end,
    contribution, direction."""
    strongest: dict
    """channel -> the strongest unfavourable change as a dict with date,
    days_before_end and contribution, or None if the channel has none."""


def score_window(window, channels=("delta_t", "flow"), *, detector=detect_beast,
                 time_col="timestamp", unfavourable=None, **params):
    """
    Detect changepoints in each channel of one window, score them, and find
    the strongest unfavourable change.

    Parameters
    ----------
    window : pd.DataFrame
        One observation window, one row per day in time order, with a time
        column and one column per channel (paper: six months of daily values
        of ``delta_t`` in K and ``flow`` in m³/h).
    channels : sequence of str
        Columns to score.
    detector : callable
        ``detector(values) -> (positions, probabilities or None)``. The
        default is BEAST as in the paper. Any other detector can be used,
        for example PELT from ``ruptures``.
    time_col : str
        Name of the time column.
    unfavourable : dict, optional
        channel -> "decrease" or "increase"; defaults to ``UNFAVOURABLE``.
    **params
        Passed to ``decision_scores`` (q, z_thr, baseline, ref_days, w2, snap).

    Returns
    -------
    WindowResult
    """
    window = window.reset_index(drop=True)
    times = pd.to_datetime(window[time_col])
    directions = {**UNFAVOURABLE, **(unfavourable or {})}
    n = len(window)

    scores, rows, strongest = {}, [], {}
    for col in channels:
        if col not in directions:
            raise ValueError(f"no unfavourable direction known for channel {col!r}; pass unfavourable=")
        positions, probabilities = detector(window[col].to_numpy())
        s = decision_scores(window[col].to_numpy(), positions, probabilities,
                            unfavourable=directions[col], **params)
        scores[col] = s
        for i in np.flatnonzero(s):
            rows.append({"channel": col, "date": times[i], "position": int(i), "days_before_end": n - 1 - int(i),
                         "contribution": s[i], "direction": "unfavourable" if s[i] < 0 else "favourable"})
        if s.min() < 0:
            i = int(np.argmin(s))
            strongest[col] = {"date": times[i], "days_before_end": n - 1 - i, "contribution": s[i]}
        else:
            strongest[col] = None

    columns = ["channel", "date", "position", "days_before_end", "contribution", "direction"]
    table = pd.DataFrame(rows, columns=columns).astype({"date": "datetime64[ns]", "position": int,
                                                        "days_before_end": int, "contribution": float})
    return WindowResult(scores, table, strongest)
