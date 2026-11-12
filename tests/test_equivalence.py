"""
The package must reproduce the paper's analysis code (tests/reference_scoring.py,
an unchanged copy of BEAST-Paper/analysis/scoring.py) exactly.
"""

import itertools
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import reference_scoring as ref
from dhfault import UNFAVOURABLE, decision_scores, detect_beast, rank_windows, score_window

DATA = Path(__file__).resolve().parents[1] / "data" / "synthetic_windows.csv"


def synthetic_windows():
    if not DATA.exists():
        pytest.skip("run generate_data.py first")
    data = pd.read_csv(DATA, parse_dates=["timestamp"])
    return {c: w.reset_index(drop=True) for c, w in data.groupby("case", sort=False)}


def reference(window, col, cps, probs, **kw):
    """The analysis code's scores, padded to the full window."""
    kw.setdefault("baseline", 30)
    out = ref.decision_scores(window, np.asarray(cps, float), np.asarray(probs, float), col, **kw)
    return np.r_[np.zeros(kw["baseline"]), out]


def package(window, col, cps, probs, **kw):
    return decision_scores(window[col].to_numpy(), cps, probs, unfavourable=UNFAVOURABLE[col], **kw)


@pytest.mark.parametrize("col", ["delta_t", "flow"])
def test_beast_on_synthetic_data(col):
    for window in synthetic_windows().values():
        cps, probs = detect_beast(window[col])
        assert np.array_equal(package(window, col, cps, probs), reference(window, col, cps, probs))


def random_window(rng, n=183):
    """
    A noisy series with level shifts of random size and sign, as delta_t and
    as flow, and candidate changepoints: one near each shift (within a day,
    as a detector would report it) plus some anywhere, a few inside the
    reference period or below the probability threshold.
    """
    shifts = np.sort(rng.choice(np.arange(5, n - 3), rng.integers(1, 6), replace=False))
    level = np.zeros(n)
    for t in shifts:
        level[t:] += rng.choice([-1, 1]) * rng.uniform(3, 15)
    dt = 35 + level + rng.normal(0, 1, n)
    flow = 0.1 * np.exp(-level / 30) * np.exp(rng.normal(0, 0.05, n))
    near = np.clip(shifts + rng.integers(-1, 2, len(shifts)), 1, n - 1)
    anywhere = rng.integers(1, n, rng.integers(0, 6))
    cps = np.r_[near, anywhere].astype(float)
    probs = np.r_[rng.uniform(0.95, 1.0, len(near)), rng.uniform(0.3, 1.0, len(anywhere))]
    return pd.DataFrame({"delta_t": dt, "flow": flow}), np.r_[cps, np.nan, np.nan], np.r_[probs, np.nan, np.nan]


def test_random_cases_are_not_trivial():
    """The random cases must actually produce favourable and unfavourable scores."""
    signs = []
    for seed in range(40):
        window, cps, probs = random_window(np.random.default_rng(seed))
        for col in ["delta_t", "flow"]:
            s = package(window, col, cps, probs)
            signs += list(np.sign(s[s != 0]))
    assert signs.count(-1) > 10 and signs.count(1) > 10


@pytest.mark.parametrize("seed", range(40))
def test_random_series_and_changepoints(seed):
    window, cps, probs = random_window(np.random.default_rng(seed))
    for col in ["delta_t", "flow"]:
        assert np.array_equal(package(window, col, cps, probs), reference(window, col, cps, probs))


@pytest.mark.parametrize("q,z_thr,baseline,w2,snap",
                         list(itertools.product([0.5, 0.99], [None, 2.0, 3.0], [21, 30, 45], [1, 2, 5], [0, 1, 2])))
def test_parameters(q, z_thr, baseline, w2, snap):
    rng = np.random.default_rng(int(q * 100) + baseline + 7 * w2 + snap)
    window, cps, probs = random_window(rng)
    kw = dict(q=q, z_thr=z_thr, baseline=baseline, w2=w2, snap=snap)
    for col in ["delta_t", "flow"]:
        got = package(window, col, cps, probs, ref_days=baseline, **kw)
        assert np.array_equal(got, reference(window, col, cps, probs, ref_days=baseline, **kw))


def test_no_probabilities_means_p_equals_one():
    window, cps, _ = random_window(np.random.default_rng(1))
    cps = cps[~np.isnan(cps)]
    for col in ["delta_t", "flow"]:
        assert np.array_equal(package(window, col, cps, None), reference(window, col, cps, np.ones(len(cps))))


def test_ranking_matches_reference():
    rng = np.random.default_rng(3)
    scores = {}
    for w in range(60):
        scores[f"w{w}"] = {c: np.where(rng.random(150) < 0.02, rng.normal(0, 3, 150), 0.0) for c in ["delta_t", "flow"]}
    expected = ref.case_scores(scores, scores).sort_values("key", kind="stable")
    got = rank_windows(scores)
    assert list(got.window) == list(expected.index)
    assert np.array_equal(got.tier.to_numpy(), expected.tier.to_numpy())
    assert np.allclose(got.joint_evidence, np.abs(expected.joint))
    assert np.allclose(got.summed_score, expected["sum"])


def test_score_window_on_synthetic_data():
    windows = synthetic_windows()
    truth = pd.read_csv(DATA.parent / "ground_truth.csv").set_index("case")
    results = {c: score_window(w) for c, w in windows.items()}
    ranking = rank_windows(results)
    assert list(ranking.window) == ["fault_both", "fault_delta_t", "no_fault"]
    assert list(ranking.tier) == [1, 2, 2]
    assert results["no_fault"].strongest == {"delta_t": None, "flow": None}
    for case, channels in [("fault_both", ["delta_t", "flow"]), ("fault_delta_t", ["delta_t"])]:
        onset = pd.Timestamp(truth.fault_onset[case])
        for col in channels:
            assert results[case].strongest[col]["date"] == onset
    cp = results["fault_both"].changepoints
    assert set(cp.columns) == {"channel", "date", "position", "days_before_end", "contribution", "direction"}
