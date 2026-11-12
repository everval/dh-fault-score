# -*- coding: utf-8 -*-
"""
The paper's decision score and ranking rule, shared by 04_beast.py,
05_evaluate.py and 06_alternatives.py.

Decision score (`decision_scores`). Each BEAST change point with posterior
probability above P_CH, after the BASELINE reference days, is snapped to the
largest jump within +/-1 day and kept only if the series moves more than
3 sigma from its 30-day reference mean within +/-W2 days. Its magnitude is

    p * log(1 + Z) * log(1 + J) * log(1 + C)

(Z standardised deviation, J = |first difference| at the change, the jump
size, C local consistency), signed by the physical direction of the change
(`_unfavourable`). Each change point is paired with its own posterior
probability. The keyword arguments of `decision_scores` vary the score's
parameters and components for 08_sensitivity.py; their defaults reproduce
the paper.

Two indexing errors of the original code, from which this module started,
were corrected in November 2026: change points inside the baseline were
dropped from the locations but not from the probabilities they were paired
with, and J also took a +/-1-day sum that, through a wrong index, only
applied to change points in the first five days after the baseline. Neither
changed the paper's conclusions (AP 0.771 -> 0.772).

Ranking rule (`case_scores`). With N_c the sum of the unfavourable
(negative) contributions of channel c,

    joint  P_h = -|N_dT| * |N_flow|,
    sum    S_h = total delta_t score + total flow score.

Cases with unfavourable evidence in both delta_t and flow (P_h < 0, tier 1)
come first, ranked by P_h; all others follow, ranked by S_h. Lower = stronger
evidence of a fault. The product is motivated by the load factor
(flow * delta_t): utilities regard a rise in flow together with a fall in
delta_t as insufficient cooling, but in the product of the measurements the
two changes cancel. Multiplying the unfavourable evidence instead makes them
reinforce each other.
"""

import numpy as np
import pandas as pd

P_CH = 0.99      # BEAST posterior threshold q
BASELINE = 30    # days at the start of each window used only as reference
W2 = 2           # half-width (days) of the snapping and consistency window
SCORE_COLUMNS = ["delta_t", "flow"]   # channels of the paper's ranking

# Load-factor sign rules, compared in 06_alternatives.py:
#   up / down    - an increase / a decrease is unfavourable,
#   both         - every significant shift is unfavourable,
#   utility      - an increase is unfavourable if flow rises while delta_t
#                  falls or stays flat (the utility's poor-cooling pattern),
#                  any other shift counts as favourable,
#   utility_only - as utility, but other shifts score zero.
LF_DIRECTIONS = ["up", "down", "both", "utility", "utility_only"]

# Unit of the jump J in log(1 + J / unit), compared in 06_alternatives.py:
#   raw      - the channel's own units (the paper),
#   std      - the 30-day reference sigma also used in Z,
#   hh_level - the standard deviation of the channel over the whole window,
#   hh_diff  - that of its day-to-day differences.
SCALES = ["raw", "std", "hh_level", "hh_diff"]


def _unfavourable(col, step, data, check, direction):
    """True if the change at `check` (first difference `step`) is in the bad direction."""
    if col == "temp_return":
        return step >= 0
    if col == "delta_t":
        return step < 0
    if col == "flow":
        return step > 0
    if col != "load_factor":
        raise NameError(f"No sign rule for {col}!")
    if direction == "both":
        return True
    if direction in ("utility", "utility_only"):
        return (step > 0 and data.flow.diff().to_numpy()[check] > 0
                and data.delta_t.diff().to_numpy()[check] <= 0)
    return step > 0 if direction == "up" else step <= 0


def decision_scores(data_full, cps, probs_all, col, direction="up", scale="raw", *,
                    q=P_CH, z_thr=3.0, baseline=BASELINE, ref_days=None, w2=W2, snap=1,
                    use_p=True, use_Z=True, use_J=True, use_C=True, cp_offset=0):
    """
    Per-day signed decision score of one channel of one window.

    Parameters
    ----------
    data_full : pd.DataFrame
        One full window (baseline days included), with its original index.
    cps, probs_all : np.ndarray
        BEAST's change points and their posterior probabilities for `col`.
    col : str
        "delta_t", "flow", "temp_return" or "load_factor".
    direction : str
        Load-factor sign rule, one of LF_DIRECTIONS (ignored for other columns).
    scale : str
        Unit of the jump term, one of SCALES.

    Keyword-only arguments, for the sensitivity analysis in 08_sensitivity.py.
    The defaults reproduce the paper exactly.
    q : float
        BEAST posterior threshold.
    z_thr : float or None
        Keep a change point only if Z > z_thr; None keeps every one.
    baseline : int
        Days at the start of the window that are only used as reference.
    ref_days : int or None
        Length of the reference before the change (mean and sigma in Z);
        None = `baseline`. Truncated at the start of the window.
    w2 : int
        Half-width of the window for Z and of the vicinity in C.
    snap : int
        Half-width k of the snapping window: the change point moves to the
        largest |first difference| within +/-k days (0 = no snapping).
        The original two-stage code snapped with W2 = k + 1, which always
        amounts to +/-(W2 - 1) = +/-1, the paper's window.
    use_p, use_Z, use_J, use_C : bool
        False replaces that factor of the score by 1 (ablation).
    cp_offset : int
        Added to BEAST's change-point times before use. Rbeast counts time
        from 1, so its change point t is the first day of the new segment
        at 0-based position t - 1, where the jump appears in the first
        difference; -1 converts to that position. The paper uses 0, and the
        +/-1-day snapping moves about 84% of its change points back by one
        day (08_sensitivity.py compares both for the snapping window).

    Returns
    -------
    np.ndarray
        One score per day after the baseline, mostly zero.
    """
    ref_days = baseline if ref_days is None else ref_days
    s = snap + 1   # the original code's snapping parameter (see `snap`)
    data = data_full.iloc[baseline:]
    series = data[col]
    jumps = data_full[col].diff().to_numpy()
    decision_rule = np.zeros(data.shape[0])

    keep = probs_all > q
    idx = cps[keep].astype(int) + cp_offset
    probs = probs_all[keep]
    after = idx > baseline
    idx, probs = idx[after] - baseline, probs[after]

    for check, prob in zip(idx, probs):
        # Snap to the largest jump nearby, in case BEAST's location is slightly off.
        if snap > 0:
            lo = check - min(check, s)
            if data.index.get_loc(series.iloc[lo:check + s + 1].diff().abs().idxmax()) != check:
                check = data.index.get_loc(series.iloc[lo:check + s].diff().abs().idxmax())
        lo = check - min(check, w2)

        jump = jumps[baseline + check]
        vicinity = jumps[baseline + check - w2:baseline + min(len(jumps), check + w2 + 1)]
        reference = data_full[col].iloc[max(0, baseline + check - ref_days):baseline + check]   # the days before the change
        mu, sigma = reference.mean(), max(reference.std(), 1e-6)

        deviation = np.abs(series.iloc[lo:check + w2 + 1] - mu)
        if z_thr is not None and not (deviation > z_thr * sigma).any():
            continue

        J = np.abs(jump)
        unit = {"raw": 1, "std": sigma,
                "hh_level": max(data_full[col].std(), 1e-6),
                "hh_diff": max(np.nanstd(jumps), 1e-6)}[scale]
        dec = ((np.log1p(deviation.max() / sigma) if use_Z else 1.0)
               * (np.log1p(J / unit) if use_J else 1.0)
               * (np.log1p(np.abs(vicinity.sum()) / max(np.abs(jump), 1e-6)) if use_C else 1.0))

        bad = _unfavourable(col, series.diff().to_numpy()[check], data, check, direction)
        if not bad and direction == "utility_only" and col == "load_factor":
            continue
        decision_rule[check] += (-dec if bad else dec) * (prob if use_p else 1.0)

    return decision_rule


def case_scores(detections, uids, columns=SCORE_COLUMNS):
    """
    Sum, joint score, tier and ranking key per case.

    Parameters
    ----------
    detections : dict
        detections[uid][col] -> per-day decision scores.
    uids : iterable
    columns : list of two str
        The channels to combine (the paper: delta_t and flow).

    Returns
    -------
    pd.DataFrame indexed by uid with columns
        sum   - S_h,
        joint - P_h (0 when a channel has no unfavourable evidence),
        tier  - 1 if P_h < 0, else 2,
        key   - one number that sorts tier 1 by P_h before tier 2 by S_h,
                for sorting and for rank-based metrics such as AP.
    """
    uids = list(uids)
    total = {c: np.array([detections[u][c].sum() for u in uids]) for c in columns}
    negative = {c: np.array([np.clip(detections[u][c], None, 0).sum() for u in uids]) for c in columns}
    s = sum(total.values())
    p = -np.abs(np.prod([negative[c] for c in columns], axis=0))
    key = np.where(p < 0, p + min(s.min(), 0) - 1, s)   # shift tier 1 below every sum
    return pd.DataFrame({"sum": s, "joint": p, "tier": np.where(p < 0, 1, 2), "key": key}, index=uids)


def faults_at_k(key, fault, k):
    """
    Expected number of faults among the `k` lowest keys, with ties in random
    order. The data list reported-fault windows first, so a stable sort would
    rank them first within a tie.
    """
    key, fault = np.asarray(key), np.asarray(fault)
    cut = np.sort(key)[k - 1]
    above, tied = key < cut, key == cut
    return fault[above].sum() + (k - above.sum()) * fault[tied].mean()
