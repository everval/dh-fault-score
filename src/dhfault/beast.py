"""BEAST changepoint detection as used in the paper."""

import contextlib
import os

import numpy as np

SEED = 18


def detect_beast(series, *, seed=SEED):
    """
    Changepoints of a series from BEAST (Rbeast) with a piecewise-linear
    trend and no seasonal component, as in the paper.

    Parameters
    ----------
    series : array-like of float
        Daily values of one channel over the whole window.
    seed : int
        Seed of BEAST's MCMC sampler (the paper uses 18).

    Returns
    -------
    positions, probabilities : np.ndarray
        The changepoints and their posterior probabilities.

    Notes
    -----
    Rbeast counts time from 1, so a returned position, read as a 0-based
    index, lies one day after the jump. The paper passes these positions on
    unchanged, and the +/-1-day refinement of the decision score moves them
    back to the jump (Appendix A of the paper). Subtract 1 to obtain 0-based
    positions of the first day after the change.
    """
    import Rbeast as rb   # imported here so that the rest of the package works without it

    with open(os.devnull, "w") as fnull, contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        out = rb.beast(np.asarray(series, dtype=float), season="none", mcmc_seed=seed)
    cps, probs = np.atleast_1d(np.array(out.trend.cp)), np.atleast_1d(np.array(out.trend.cpPr))
    found = ~np.isnan(cps)
    return cps[found], probs[found]
