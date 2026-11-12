"""The two-tier ranking of observation windows."""

import numpy as np
import pandas as pd


def rank_windows(scores, channels=("delta_t", "flow")):
    """
    Rank observation windows by their evidence of fault-like behaviour.

    Windows with unfavourable (negative) contributions in both channels form
    tier 1 and are ranked by the joint evidence

        P_h = |sum of unfavourable contributions in channel 1|
              * |sum of unfavourable contributions in channel 2|,

    largest first. All other windows form tier 2 and are ranked by the
    summed score S_h, the sum of all contributions of both channels, most
    negative first.

    Parameters
    ----------
    scores : dict
        scores[window][channel] -> per-day decision scores (see
        ``decision_scores``), or a mapping of window -> ``WindowResult``.
    channels : pair of str
        The two channels to combine.

    Returns
    -------
    pd.DataFrame
        One row per window, in rank order, with columns rank, window, tier,
        joint_evidence (P_h, 0 outside tier 1), summed_score (S_h) and the
        total score of each channel.
    """
    scores = {w: getattr(s, "scores", s) for w, s in scores.items()}
    windows = list(scores)
    total = {c: np.array([scores[w][c].sum() for w in windows]) for c in channels}
    unfavourable = {c: np.array([np.clip(scores[w][c], None, 0).sum() for w in windows]) for c in channels}
    joint = np.abs(np.prod([unfavourable[c] for c in channels], axis=0))
    summed = sum(total.values())
    tier = np.where(joint > 0, 1, 2)

    table = pd.DataFrame({"window": windows, "tier": tier, "joint_evidence": joint, "summed_score": summed,
                          **{f"score_{c}": total[c] for c in channels}})
    # Tier 1 by descending P_h, then tier 2 by ascending S_h; ties keep the input order.
    order = np.lexsort((np.arange(len(windows)), np.where(tier == 1, -joint, summed), tier))
    table = table.iloc[order].reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return table
