# max_separation_orbit_finder.py

import numpy as np
import pandas as pd
from sgp4.api import Satrec
from datetime import datetime, timedelta, timezone
from orekit.pyhelpers import datetime_to_absolutedate
from scipy.optimize import minimize

from synthetic_orbits.orbit_finder.DMT import VectorizedKeplerianOrbit
from synthetic_orbits.orbit_finder.optimum_orbit_tle import convert_kep_to_tle

def get_keplerian_array_from_tle(row):
    line1_array = np.array([row["line1"]])
    line2_array = np.array([row["line2"]])
    orbit = VectorizedKeplerianOrbit(line1_array, line2_array)
    return np.array([orbit.a[0], orbit.e[0], orbit.i[0],
                     orbit.omega[0], orbit.raan[0], 0.0])


def make_complete_orbit(opt_array):
    a, e, i, omega, raan = opt_array[:5]
    p = a * (1 - e**2)
    q = a * (1 - e)
    return np.array([a, e, i, omega, raan, q, p])


def calculate_average_epoch(df):
    epoch_times = []
    for _, row in df.iterrows():
        satrec = Satrec.twoline2rv(row["line1"], row["line2"])
        year = 2000 + satrec.epochyr if satrec.epochyr < 100 else satrec.epochyr
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=satrec.epochdays - 1)
        epoch_times.append(dt)

    if not epoch_times:
        print("No epochs found in TLE data. Using current time as average epoch.")
        return datetime.now(timezone.utc)

    timestamps = [dt.timestamp() for dt in epoch_times]
    average_timestamp = sum(timestamps) / len(timestamps)
    return datetime.fromtimestamp(average_timestamp, timezone.utc)


def min_distance_to_catalog(x, kepler_list):
    cand = VectorizedKeplerianOrbit(make_complete_orbit(x))
    dmins = []
    for k in kepler_list:
        other = VectorizedKeplerianOrbit(make_complete_orbit(k))
        d = VectorizedKeplerianOrbit.DistanceMetric(cand, other)
        dmins.append(d)
    return float(np.min(dmins))


def sample_maxmin(kepler_list, bounds, n_samples=10000):
    best_x = None
    best_r = -np.inf
    for _ in range(n_samples):
        x = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        r_x = min_distance_to_catalog(x, kepler_list)
        if r_x > best_r:
            best_r = r_x
            best_x = x
    return best_x, best_r


def refine_maxmin(x0, kepler_list, bounds):
    def objective(x):
        return -min_distance_to_catalog(x, kepler_list)

    res = minimize(objective, x0=x0, bounds=bounds, method="L-BFGS-B")
    return res.x, -res.fun


def cluster_spacing_stats(kepler_list):
    n = len(kepler_list)
    nn_dists = []
    for i in range(n):
        oi = VectorizedKeplerianOrbit(make_complete_orbit(kepler_list[i]))
        dists = []
        for j in range(n):
            if i == j:
                continue
            oj = VectorizedKeplerianOrbit(make_complete_orbit(kepler_list[j]))
            dists.append(VectorizedKeplerianOrbit.DistanceMetric(oi, oj))
        nn_dists.append(min(dists))
    nn_dists = np.array(nn_dists)
    return {
        "nn_distances": nn_dists,
        "median": float(np.median(nn_dists)),
        "p75": float(np.percentile(nn_dists, 75)),
        "p90": float(np.percentile(nn_dists, 90)),
        "p95": float(np.percentile(nn_dists, 95)),
        "max": float(np.max(nn_dists)),
    }


def evaluate_max_separation_orbit(max_separation_kepler, kepler_list, eps=1e-12):
    r_star = min_distance_to_catalog(max_separation_kepler, kepler_list)
    stats = cluster_spacing_stats(kepler_list)
    percentile = np.mean(stats["nn_distances"] < r_star) * 100.0
    median = stats["median"]
    if median < eps:
        ratio_to_median = np.inf
    else:
        ratio_to_median = r_star / median
    return {
        "r_star": float(r_star),
        "percentile_vs_cluster": float(percentile),
        "ratio_to_median_spacing": float(ratio_to_median),
        "cluster_stats": stats,
    }


def get_maximally_separated_orbit(df, n_samples=5000, return_diagnostics=True):
    all_keplers = [get_keplerian_array_from_tle(row) for _, row in df.iterrows()]
    if len(all_keplers) < 2:
        print("Not enough orbits for max-min optimization.")
        return (df, None) if return_diagnostics else df

    min_a = min(k[0] for k in all_keplers)
    min_e = min(k[1] for k in all_keplers)
    min_i = min(k[2] for k in all_keplers)
    min_omega = min(k[3] for k in all_keplers)
    min_raan = min(k[4] for k in all_keplers)
    max_a = max(k[0] for k in all_keplers)
    max_e = max(k[1] for k in all_keplers)
    max_i = max(k[2] for k in all_keplers)
    max_omega = max(k[3] for k in all_keplers)
    max_raan = max(k[4] for k in all_keplers)

    bounds = [
        (min_a, max_a),
        (min_e, max_e),
        (min_i, max_i),
        (min_omega, max_omega),
        (min_raan, max_raan),
        (0.0, 2 * np.pi),
    ]

    x0, r0 = sample_maxmin(all_keplers, bounds, n_samples=n_samples)
    print(f"Best sampled max_separation radius: {r0:.6f}")

    x_star, r_star = refine_maxmin(x0, all_keplers, bounds)
    print(f"Refined max_separation radius: {r_star:.6f}")
    print(
        "Maximally separated Keplerian Elements: "
        "{a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f; v: %.6f;}"
        % tuple(x_star)
    )

    avg_epoch = calculate_average_epoch(df)
    initialDate = datetime_to_absolutedate(avg_epoch)
    ref_row = df.iloc[0]
    satrec = Satrec.twoline2rv(ref_row["line1"], ref_row["line2"])
    mean_anomaly = satrec.mo
    x_star[5] = mean_anomaly
    line1, line2 = convert_kep_to_tle(x_star, mean_anomaly, initialDate)

    max_separation_entry = {
        "satNo": "99999",
        "name": "MaximallySeparated",
        "line1": line1,
        "line2": line2,
        "correlated": True,
        "dataset": ref_row.get("dataset") if "dataset" in ref_row else None,
    }
    df = pd.concat([df, pd.DataFrame([max_separation_entry])], ignore_index=True)

    if return_diagnostics:
        diagnostics = evaluate_max_separation_orbit(x_star, all_keplers)
        return df, diagnostics
    return df
