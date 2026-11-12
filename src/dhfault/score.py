"""
The decision score of one channel of one observation window.

Each candidate changepoint is

1. kept if its posterior probability exceeds ``q`` (if the detector gives
   probabilities) and it lies after the ``baseline`` reference period;
2. moved to the largest absolute first difference within ``+/-snap`` days;
3. kept only if, within ``+/-w2`` days, the series departs more than
   ``z_thr`` standard deviations from its mean over the ``ref_days`` days
   before the change;
4. scored  p * log(1 + Z) * log(1 + J) * log(1 + C),  with

   Z  the largest standardised deviation within +/-w2 days,
   J  the size of the jump, |first difference| at the change,
   C  |net change over +/-w2 days| / J,

   and signed negative (unfavourable) if the jump goes in the channel's
   unfavourable direction, positive otherwise.

The defaults are the paper's settings.
"""

import numpy as np

#: Direction of an unfavourable change for the channels used in the paper.
UNFAVOURABLE = {"delta_t": "decrease", "flow": "increase", "temp_return": "increase"}

_EPS = 1e-6


def decision_scores(series, changepoints, probabilities=None, *, unfavourable,
                    q=0.99, z_thr=3.0, baseline=30, ref_days=30, w2=2, snap=1):
    """
    Per-day signed decision score of one channel of one window.

    Parameters
    ----------
    series : array-like of float
        The daily values of one channel over the whole window, reference
        period included, without missing values.
    changepoints : array-like of int
        Candidate changepoints from any detector, as positions in ``series``
        (0-based) of the first day after the change, where the jump shows
        in the first difference. A position off by a day is corrected by
        the refinement step (``snap``).
    probabilities : array-like of float, optional
        Posterior probability of each changepoint. If None (detectors such
        as PELT), every changepoint is kept and p = 1.
    unfavourable : {"decrease", "increase"}
        Direction of a fault-like change in this channel, e.g. "decrease"
        for the temperature difference and "increase" for the flow (see
        ``UNFAVOURABLE``).
    q : float
        Probability threshold; changepoints with probability <= q are dropped.
    z_thr : float or None
        Keep a changepoint only if Z > z_thr; None keeps every one.
    baseline : int
        Days at the start of the window used only as reference; changepoints
        at or before this position are not scored.
    ref_days : int
        Days before the change from which the reference mean and standard
        deviation are computed.
    w2 : int
        Half-width (days) of the local window for Z and C.
    snap : int
        Half-width (days) of the refinement window; 0 = no refinement.

    Returns
    -------
    np.ndarray
        One score per day of the window, zero except at scored changepoints
        (and always zero in the reference period). Negative = unfavourable.
    """
    if unfavourable not in ("decrease", "increase"):
        raise ValueError('unfavourable must be "decrease" or "increase"')
    y = np.asarray(series, dtype=float)
    if np.isnan(y).any():
        raise ValueError("series contains missing values; interpolate them first")
    n = len(y)
    if baseline < w2 + 1:
        raise ValueError("baseline must be longer than w2")
    jumps = np.r_[np.nan, np.diff(y)]
    scores = np.zeros(n)

    cps = np.asarray(changepoints, dtype=float)
    probs = np.ones(len(cps)) if probabilities is None else np.asarray(probabilities, dtype=float)
    keep = ~np.isnan(cps) & ((probs > q) if probabilities is not None else True)
    cps, probs = cps[keep].astype(int), probs[keep]
    after = cps > baseline
    cps, probs = cps[after], probs[after]

    for t, p in zip(cps, probs):
        if t >= n:
            continue
        # Refinement: the largest jump within +/-snap days (first one on ties).
        if snap > 0:
            lo, hi = max(t - snap, baseline + 1), min(t + snap, n - 1)
            t = lo + int(np.argmax(np.abs(jumps[lo:hi + 1])))

        reference = y[max(0, t - ref_days):t]
        mu, sigma = reference.mean(), max(reference.std(ddof=1), _EPS)
        deviation = np.abs(y[max(t - w2, baseline):t + w2 + 1] - mu)
        if z_thr is not None and not (deviation > z_thr * sigma).any():
            continue

        jump = jumps[t]
        Z = deviation.max() / sigma
        J = abs(jump)
        C = abs(jumps[t - w2:t + w2 + 1].sum()) / max(J, _EPS)
        magnitude = np.log1p(Z) * np.log1p(J) * np.log1p(C)

        bad = jump < 0 if unfavourable == "decrease" else jump > 0
        scores[t] += (-magnitude if bad else magnitude) * p
    return scores
