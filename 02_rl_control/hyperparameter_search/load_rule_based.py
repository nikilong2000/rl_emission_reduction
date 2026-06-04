#!/usr/bin/env python3
"""
Adapt a rule-based-controller result directory into the same schema the RL
evaluation pipeline emits (``evaluation_data.csv`` + ``evaluation_metrics.json``),
so the existing comparison / plotting / per-km utilities work unchanged.

Only the **isoSOC** standard-WLTC variant is used for the fair (charge-sustaining)
comparison against the RL agents.

Column mapping (rule-based -> RL schema)
----------------------------------------
    Car_Speed_kmph_pred        -> speed_actual
    <WLTC.csv car_speed_kmph>  -> speed_target   (shared reference; v_ref ~= this)
    SOC_1_pred                 -> soc
    ICE_Torque_Nm_pred         -> ice_torque
    NOx_out_m_gps_pred  (g/s)  -> nox             (tailpipe; matches RL nox_tp)
    fuel_soll_mg               -> fuel            (commanded injection, parity field)
    tot_burned_fuel_gps_pred   -> fuel_tot_gps    (burned g/s; the comparable fuel)
    CO2_out_m_gps_pred  (g/s)  -> co2_tp_gps
    ICE_Speed_rpm              -> ice_speed_rpm   (engine_on := > 1 rpm)
    EM2_Torque_Nm              -> em2_torque_nm
    Brake_perc                 -> brake_perc

Units were verified: NOx/fuel/CO2 columns are g/s (Σ·dt matches the cycle summary
to within 0.2%).

Usage
-----
    python load_rule_based.py            # default isoSOC dir -> _adapted/eval_wltc/
"""
import os
import sys
import json
import argparse

import numpy as np
import pandas as pd

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
RL_DIR = os.path.dirname(THIS_DIR)  # 02_rl_control
sys.path.insert(0, RL_DIR)

from utils.evaluation_utils import calculate_emissions_per_km  # noqa: E402

DT = 0.5
DEFAULT_RB_DIR = os.path.join(
    RL_DIR, "rule_based_results", "20260209_151920_results_Test_WLTC_isoSOC"
)
WLTC_CSV = os.path.join(RL_DIR, "data_train", "WLTC.csv")


def _shared_target(n):
    """WLTC.csv reference speed trace, used as the common target for both controllers."""
    w = pd.read_csv(WLTC_CSV)
    w.columns = [c.strip() for c in w.columns]
    tgt = w["car_speed_kmph"].to_numpy()
    if len(tgt) >= n:
        return tgt[:n]
    # pad if rule-based is marginally longer
    return np.concatenate([tgt, np.full(n - len(tgt), tgt[-1])])


def adapt(rb_dir, out_dir):
    csv = os.path.join(rb_dir, "data_with_predictions.csv")
    df = pd.read_csv(csv, sep=";", encoding="latin1")
    df.columns = [c.strip() for c in df.columns]
    n = len(df)

    speed_actual = df["Car_Speed_kmph_pred"].to_numpy()
    speed_target = _shared_target(n)
    soc = df["SOC_1_pred"].to_numpy()
    nox = df["NOx_out_m_gps_pred"].to_numpy()
    fuel_tot_gps = df["tot_burned_fuel_gps_pred"].to_numpy()
    co2_tp_gps = df["CO2_out_m_gps_pred"].to_numpy()
    ice_speed = df["ICE_Speed_rpm"].to_numpy()
    engine_on = ice_speed > 1.0

    results = {
        "speed_actual": speed_actual,
        "speed_target": speed_target,
        "soc": soc,
        "ice_torque": df["ICE_Torque_Nm_pred"].to_numpy(),
        "nox": nox,
        "fuel": df["fuel_soll_mg"].to_numpy(),
        "fuel_tot_gps": fuel_tot_gps,
        "co2_tp_gps": co2_tp_gps,
        "engine_on": engine_on,
        "ice_speed_rpm": ice_speed,
        "em2_torque_nm": df["EM2_Torque_Nm"].to_numpy(),
        "brake_perc": df["Brake_perc"].to_numpy(),
    }

    os.makedirs(out_dir, exist_ok=True)
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, "evaluation_data.csv"), index=False
    )

    # --- metrics (same keys as models/eval.py) ---
    total_distance_km = float(np.sum(speed_actual) * DT / 3600.0)
    total_nox_g = float(np.sum(nox) * DT)
    total_fuel_burned_g = float(np.sum(fuel_tot_gps) * DT)
    total_co2_g = float(np.sum(co2_tp_gps) * DT)
    err = speed_actual - speed_target

    def per_km(g):
        return float(g / total_distance_km) if total_distance_km > 1e-6 else float("nan")

    metrics = {
        "model_path": rb_dir,
        "controller": "rule_based",
        "variant": os.path.basename(rb_dir),
        "total_fuel_burned_g": total_fuel_burned_g,
        "total_co2_g": total_co2_g,
        "total_nox_g": total_nox_g,
        "total_distance_km": total_distance_km,
        "nox_g_per_km": per_km(total_nox_g),
        "fuel_burned_g_per_km": per_km(total_fuel_burned_g),
        "co2_g_per_km": per_km(total_co2_g),
        "engine_off_pct": float(100.0 * np.mean(~engine_on)),
        "mae_speed_kmph": float(np.mean(np.abs(err))),
        "rmse_speed_kmph": float(np.sqrt(np.mean(err**2))),
        "initial_soc": float(soc[0]),
        "final_soc": float(soc[-1]),
        "delta_soc": float(soc[-1] - soc[0]),
        "max_abs_soc_drift": float(np.max(np.abs(soc - soc[0]))),
        "rms_soc_drift": float(np.sqrt(np.mean((soc - soc[0]) ** 2))),
    }
    with open(os.path.join(out_dir, "evaluation_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    calculate_emissions_per_km(results, out_dir, dt=DT)

    print(f"Adapted {os.path.basename(rb_dir)} -> {out_dir}")
    print(
        f"  distance={total_distance_km:.2f} km  NOx={total_nox_g:.3f} g "
        f"({metrics['nox_g_per_km']*1000:.1f} mg/km)  "
        f"fuel_burned={total_fuel_burned_g:.1f} g  "
        f"engine_off={metrics['engine_off_pct']:.1f}%  "
        f"RMSE={metrics['rmse_speed_kmph']:.2f} km/h  "
        f"SOC {metrics['initial_soc']:.3f}->{metrics['final_soc']:.3f}"
    )
    return metrics


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--rb_dir", default=DEFAULT_RB_DIR)
    ap.add_argument(
        "--out_dir",
        default=os.path.join(RL_DIR, "rule_based_results", "_adapted", "eval_wltc"),
    )
    args = ap.parse_args()
    adapt(args.rb_dir, args.out_dir)


if __name__ == "__main__":
    main()
