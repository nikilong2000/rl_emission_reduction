import os
import numpy as np


def calculate_emissions_per_km(results, log_dir, dt=0.5):
    """
    Compute NOx emissions per km plus overall and phase-specific WLTC statistics.

    Output file: emissions_per_km.txt

    Phase durations (seconds):
      - low: 589
      - medium: 433
      - high: 455
      - extra_high: 323
    """
    speed_actual = np.array(results.get("speed_actual", []), dtype=np.float64)
    nox_gs = np.array(results.get("nox", []), dtype=np.float64)

    n = min(len(speed_actual), len(nox_gs))
    speed_actual = speed_actual[:n]
    nox_gs = nox_gs[:n]

    if n == 0:
        out_path = os.path.join(log_dir, "emissions_per_km.txt")
        with open(out_path, "w") as f:
            f.write("No evaluation samples available for emissions calculation.\n")
        return

    distance_step_km = speed_actual * dt / 3600.0
    nox_step_mg = nox_gs * dt * 1000.0

    total_distance_km = float(np.sum(distance_step_km))
    total_nox_mg = float(np.sum(nox_step_mg))
    overall_mg_per_km = (
        total_nox_mg / total_distance_km if total_distance_km > 1e-12 else float("nan")
    )

    phase_durations_s = {
        "low": 589,
        "medium": 433,
        "high": 455,
        "extra_high": 323,
    }

    phase_ranges = {}
    start_idx = 0
    for phase_name, duration_s in phase_durations_s.items():
        phase_steps = int(round(duration_s / dt))
        end_idx = start_idx + phase_steps
        phase_ranges[phase_name] = (start_idx, min(end_idx, n))
        start_idx = end_idx

    out_path = os.path.join(log_dir, "emissions_per_km.txt")
    with open(out_path, "w") as f:
        f.write("Emissions Summary\n")
        f.write("=================\n")
        f.write(f"Samples used: {n}\n")
        f.write(f"Step size (dt): {dt:.3f} s\n")
        f.write(f"Total distance (km): {total_distance_km:.6f}\n")
        f.write(f"Total NOx (mg): {total_nox_mg:.3f}\n")
        if np.isfinite(overall_mg_per_km):
            f.write(f"Overall NOx (mg/km): {overall_mg_per_km:.3f}\n")
            f.write(f"Overall NOx pass (<=80 mg/km): {overall_mg_per_km <= 80.0}\n")
        else:
            f.write("Overall NOx (mg/km): NaN (zero distance)\n")
            f.write("Overall NOx pass (<=80 mg/km): False\n")

        f.write("\nNOx by WLTC phase\n")
        f.write("=================\n")
        f.write("phase,duration_s,distance_km,nox_mg,nox_mg_per_km,pass_80mg_per_km\n")

        for phase_name, duration_s in phase_durations_s.items():
            s, e = phase_ranges[phase_name]
            if s >= n:
                phase_distance_km = 0.0
                phase_nox_mg = 0.0
                phase_mg_per_km = float("nan")
            else:
                phase_distance_km = float(np.sum(distance_step_km[s:e]))
                phase_nox_mg = float(np.sum(nox_step_mg[s:e]))
                phase_mg_per_km = (
                    phase_nox_mg / phase_distance_km
                    if phase_distance_km > 1e-12
                    else float("nan")
                )

            phase_pass = bool(np.isfinite(phase_mg_per_km) and phase_mg_per_km <= 80.0)
            phase_mg_per_km_str = (
                f"{phase_mg_per_km:.3f}" if np.isfinite(phase_mg_per_km) else "NaN"
            )
            f.write(
                f"{phase_name},{duration_s},{phase_distance_km:.6f},"
                f"{phase_nox_mg:.3f},{phase_mg_per_km_str},{phase_pass}\n"
            )

        f.write("\nNOx per km segments\n")
        f.write("====================\n")
        f.write("Kilometer, NOx (mg/km), NOx_Pass (<=80 mg/km)\n")

        distance_km = 0.0
        accumulated_nox_mg = 0.0
        km_counter = 1

        for v, nox in zip(speed_actual, nox_gs):
            dist_step = v * dt / 3600.0
            nox_mg_step = nox * dt * 1000.0

            distance_km += dist_step
            accumulated_nox_mg += nox_mg_step

            if distance_km >= 1.0:
                nox_pass = accumulated_nox_mg <= 80.0
                f.write(f"{km_counter}, {accumulated_nox_mg:.2f}, {nox_pass}\n")
                distance_km = 0.0
                accumulated_nox_mg = 0.0
                km_counter += 1

        if distance_km > 0.1:
            nox_per_km = accumulated_nox_mg / distance_km
            nox_pass = nox_per_km <= 80.0
            f.write(f"Partial ({distance_km:.2f} km), {nox_per_km:.2f}, {nox_pass}\n")
