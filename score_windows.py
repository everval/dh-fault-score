# -*- coding: utf-8 -*-
"""
Run BEAST on each window, compute the paper's decision score, rank the
windows, and report the date of the strongest unfavourable change.

Usage
-----
    python score_windows.py                       # the synthetic data
    python score_windows.py my_windows.csv        # your own data

The input CSV has one row per day and the columns
    case, timestamp, delta_t, flow
(other columns, such as temp_supply and temp_return, are plotted if present).
Each `case` is one observation window; the paper uses six months of daily
values, the first 30 days of which serve only as a reference period.

Steps (as in the paper; implemented in the dhfault package)
---------------------------------------------------------------
1. dhfault.detect_beast: BEAST (Rbeast) with season="none" and
   mcmc_seed=18 on delta_t and flow.
2. dhfault.decision_scores: changepoints with posterior probability
   > 0.99, after the 30-day reference period, refined to the largest jump
   within +/-1 day, kept if the series departs more than 3 sigma from its
   30-day reference mean, and scored  -/+ p log(1+Z) log(1+J) log(1+C)  by
   the direction of the change (a fall in delta_t or a rise in flow is
   unfavourable, negative).
3. dhfault.rank_windows: windows with unfavourable evidence in both
   channels form tier 1, ranked by the joint evidence P_h; the rest follow
   in tier 2, ranked by the summed score S_h.

Output (results/)
-----------------
    summary.csv        one row per window: rank, tier, P_h, S_h, channel
                       scores, and the date of the strongest unfavourable
                       change in each channel
    changepoints.csv   every scored changepoint: window, channel, date,
                       contribution, direction
    <case>.png         the measurements with the scored changepoints
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from dhfault import rank_windows, score_window

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
BASELINE = 30   # reference period (days), not scored
CHANNELS = ("delta_t", "flow")
LABELS = {"delta_t": "ΔT", "flow": "Flow"}


def plot_window(case, window, scores, path, onset=None):
    panels = [("temp", "Supply and return temperature (°C)"),
              ("delta_t", "Temperature difference ΔT (K)"),
              ("flow", "Flow (m³/h)"),
              ("score", "Decision-score contributions")]
    fig, ax = plt.subplots(len(panels), 1, figsize=(8, 7), sharex=True)
    fig.subplots_adjust(hspace=0.55)
    d = window.timestamp
    for a, (key, title) in zip(ax, panels):
        if key == "temp":
            for col, c in (("temp_supply", "tab:red"), ("temp_return", "tab:blue")):
                if col in window:
                    a.plot(d, window[col], color=c, label=col.replace("temp_", ""))
            a.legend(loc="center left", fontsize=8, frameon=False)
        elif key == "score":
            for col, c in zip(CHANNELS, ("tab:green", "tab:purple")):
                a.plot(d, scores[col], color=c, label=LABELS[col])
            a.axhline(0, color="0.6", lw=0.6)
            a.legend(loc="lower left", fontsize=8, frameon=False)
        else:
            a.plot(d, window[key], color="tab:green" if key == "delta_t" else "tab:purple")
        a.axvspan(d.iloc[0], d.iloc[BASELINE - 1], color="0.92", lw=0)
        if onset is not None:
            a.axvline(pd.Timestamp(onset), color="k", ls="--", lw=0.8)
        a.set_title(title, fontsize=10, loc="left")
        a.tick_params(labelsize=8)
        a.spines[["top", "right"]].set_visible(False)
    loc = mdates.AutoDateLocator()
    ax[-1].xaxis.set_major_locator(loc)
    ax[-1].xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    note = "grey: 30-day reference period" + ("; dashed: true fault onset" if onset is not None else "")
    fig.suptitle(f"{case}   ({note})", fontsize=10, y=0.95)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main(path):
    data = pd.read_csv(path, parse_dates=["timestamp"])
    truth_file = HERE / "data" / "ground_truth.csv"
    truth = (pd.read_csv(truth_file).set_index("case")
             if path == HERE / "data" / "synthetic_windows.csv" and truth_file.exists() else None)
    RESULTS.mkdir(exist_ok=True)

    windows = {c: w.sort_values("timestamp").reset_index(drop=True)
               for c, w in data.groupby("case", sort=False)}
    results, changepoints = {}, []
    for case, window in windows.items():
        results[case] = r = score_window(window, CHANNELS)
        changepoints.append(r.changepoints.drop(columns="position").assign(
            case=case, date=lambda d: d.date.dt.date))
        onset = truth.fault_onset.get(case) if truth is not None else None
        plot_window(case, window, r.scores, RESULTS / f"{case}.png",
                    onset if isinstance(onset, str) else None)

    rows = []
    for _, r in rank_windows(results, CHANNELS).iterrows():
        row = {"rank": r["rank"], "case": r.window, "tier": int(r.tier),
               "joint_evidence_P_h": r.joint_evidence, "summed_score_S_h": r.summed_score}
        for col in CHANNELS:
            found = results[r.window].strongest[col]
            row[f"S_{col}"] = r[f"score_{col}"]
            row[f"strongest_unfavourable_{col}_date"] = found["date"].date() if found else None
            row[f"strongest_unfavourable_{col}_days_before_end"] = found["days_before_end"] if found else None
        if truth is not None:
            row["true_fault_onset"] = truth.fault_onset.get(r.window)
        rows.append(row)
    summary = pd.DataFrame(rows)
    for col in CHANNELS:
        days = f"strongest_unfavourable_{col}_days_before_end"
        summary[days] = summary[days].astype("Int64")
    summary.to_csv(RESULTS / "summary.csv", index=False)
    cols = ["case", "channel", "date", "days_before_end", "contribution", "direction"]
    pd.concat(changepoints, ignore_index=True)[cols].to_csv(RESULTS / "changepoints.csv", index=False)

    # Report.
    print("Ranking (tier 1: unfavourable changes in both ΔT and flow, ranked by P_h;"
          " tier 2: the rest, ranked by S_h)\n")
    for _, r in summary.iterrows():
        evidence = (f"joint evidence P_h = {r.joint_evidence_P_h:.2f}" if r.tier == 1
                    else f"summed score S_h = {r.summed_score_S_h:.2f}")
        print(f"{r['rank']}. {r.case:<14} tier {r.tier}, {evidence}")
        for col in CHANNELS:
            date = r[f"strongest_unfavourable_{col}_date"]
            found = (f"strongest unfavourable change on {date} "
                     f"({r[f'strongest_unfavourable_{col}_days_before_end']:.0f} days before the window end)"
                     if date is not None else "no unfavourable change")
            print(f"     {LABELS[col]:<5} score {r[f'S_{col}']:8.2f}   {found}")
        if truth is not None:
            onset = r.true_fault_onset
            print(f"     true fault onset: {onset if isinstance(onset, str) else 'none (no fault)'}")
        print()
    print(f"Wrote summary.csv, changepoints.csv and one plot per window to {RESULTS}")


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else HERE / "data" / "synthetic_windows.csv")
