# -*- coding: utf-8 -*-
"""
Generate synthetic heat meter windows that resemble the paper's data.

The real measurements cannot be shared, so this script simulates three
six-month observation windows of daily heat meter data for a single-family
house in a district-heating network:

    no_fault      normal operation throughout,
    fault_both    from the onset, the installation stops cooling the water:
                  delta_t falls and the flow rises to deliver the same heat
                  (the insufficient-cooling pattern, first priority tier),
    fault_delta_t from the onset, delta_t falls while the flow is unchanged
                  (unfavourable evidence in one channel only, second tier).

How the data are simulated
--------------------------
Outdoor temperature follows a seasonal curve plus autocorrelated weather.
Heat demand Q (kW) rises linearly as it gets colder, plus a constant
domestic hot-water load. Supply temperature is weather-compensated. In
normal operation the installation cools the water by delta_t of about 35 K,
slightly more in cold weather, and the flow (m³/h) is whatever delivers Q at
that delta_t:  flow = Q / (1.163 * delta_t), with 1.163 kWh/(m³ K) the heat
capacity of water. Return temperature is supply minus delta_t. Energy (kWh)
and volume (m³) are cumulative meter readings, as on a real meter.

Each window ends at the window's "report date". The onset of each fault, the
ground truth that the paper never has, is written to ground_truth.csv so the
dates found by the method can be compared with it.

Output: data/synthetic_windows.csv, data/ground_truth.csv
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent / "data"
DAYS = 183                       # six-month window
END = pd.Timestamp("2026-04-01")  # window end (the "report date")
RHO_C = 1.163                    # kWh per m³ per K

CASES = {
    #  name            seed  fault onset (day index)  delta_t after onset  flow follows demand?
    "no_fault":      dict(seed=1, onset=None, dt_fault=None),
    "fault_both":    dict(seed=2, onset=160, dt_fault=14.0, flow_compensates=True),
    "fault_delta_t": dict(seed=3, onset=140, dt_fault=22.0, flow_compensates=False),
}


def simulate(seed, onset=None, dt_fault=None, flow_compensates=True):
    rng = np.random.default_rng(seed)
    t = np.arange(DAYS)
    dates = pd.date_range(end=END, periods=DAYS, freq="D")

    # Outdoor temperature: coldest in mid-January, plus AR(1) weather.
    doy = dates.dayofyear.to_numpy()
    seasonal = 8.0 + 8.5 * np.cos(2 * np.pi * (doy - 196) / 365.25)
    weather = np.zeros(DAYS)
    for i in range(1, DAYS):
        weather[i] = 0.8 * weather[i - 1] + rng.normal(0, 1.3)
    t_out = seasonal + weather

    # Heat demand (kW): space heating below 17 °C plus hot water.
    q = 0.35 + 0.25 * np.clip(17.0 - t_out, 0, None)
    q *= np.exp(rng.normal(0, 0.04, DAYS))

    # Supply temperature (°C), weather-compensated.
    t_supply = 68.0 + 0.5 * np.clip(10.0 - t_out, 0, None) + rng.normal(0, 0.6, DAYS)

    # Normal cooling, slightly better in cold weather.
    dt_normal = 35.0 + 0.15 * np.clip(10.0 - t_out, 0, None) + rng.normal(0, 0.8, DAYS)
    flow_normal = q / (RHO_C * dt_normal)

    delta_t, flow = dt_normal.copy(), flow_normal.copy()
    if onset is not None:
        after = t >= onset
        delta_t[after] = dt_fault + rng.normal(0, 0.8, after.sum())
        if flow_compensates:
            # Same heat delivered at a lower delta_t: more water.
            flow[after] = q[after] / (RHO_C * delta_t[after])
        # Otherwise the flow keeps following demand and less heat is delivered.

    t_return = t_supply - delta_t
    heat = RHO_C * flow * delta_t * 24            # kWh delivered per day
    volume_day = flow * 24                       # m³ per day

    return pd.DataFrame({
        "timestamp": dates,
        "energy": 41_250 + np.cumsum(heat),      # cumulative meter reading, kWh
        "volume": 812.0 + np.cumsum(volume_day),  # cumulative meter reading, m³
        "temp_supply": t_supply,
        "temp_return": t_return,
        "delta_t": delta_t,
        "flow": flow,
    })


def main():
    OUT.mkdir(exist_ok=True)
    frames, truth = [], []
    for name, spec in CASES.items():
        df = simulate(**spec)
        df.insert(0, "case", name)
        frames.append(df)
        onset = spec["onset"]
        truth.append({
            "case": name,
            "window_start": df.timestamp.iloc[0].date(),
            "window_end": df.timestamp.iloc[-1].date(),
            "fault_onset": df.timestamp.iloc[onset].date() if onset is not None else None,
            "fault_channels": {"no_fault": "", "fault_both": "delta_t+flow",
                               "fault_delta_t": "delta_t"}[name],
        })
    data = pd.concat(frames, ignore_index=True)
    data.round(4).to_csv(OUT / "synthetic_windows.csv", index=False)
    pd.DataFrame(truth).to_csv(OUT / "ground_truth.csv", index=False)
    print(f"Wrote {len(data)} rows for {len(CASES)} cases to {OUT}")


if __name__ == "__main__":
    main()
