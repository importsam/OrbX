import numpy as np
import pandas as pd
from sgp4.api import Satrec
from datetime import datetime, timedelta, timezone
from orekit.pyhelpers import datetime_to_absolutedate
from scipy.optimize import minimize

from orbx.synthetic_orbits.orbit_finder.DMT import VectorizedKeplerianOrbit
from orbx.synthetic_orbits.orbit_finder.optimum_orbit_tle import (
    average_bstar,
    convert_kep_to_tle,
)

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


def sample_maxmin(kepler_list, bounds, n_samples=10000, top_k=5):
    """
    Randomly sample candidate orbits and keep the ``top_k`` with the largest
    min-distance to the catalog (for subsequent local refinement).
    """
    top = []  # list of (r_x, x), ascending by r_x
    for _ in range(n_samples):
        x = np.array([np.random.uniform(lo, hi) for lo, hi in bounds])
        r_x = min_distance_to_catalog(x, kepler_list)
        if len(top) < top_k:
            top.append((r_x, x))
            top.sort(key=lambda t: t[0])
        elif r_x > top[0][0]:
            top[0] = (r_x, x)
            top.sort(key=lambda t: t[0])

    # Best-first for refinement reporting.
    top.sort(key=lambda t: t[0], reverse=True)
    return [(x, r) for r, x in top]


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


def safe_bounds(lo, hi, eps=1e-9):
    if hi - lo < eps:
        return lo - eps / 2, hi + eps / 2
    return lo, hi

def unwrap_to_ref(angle, ref):
    return ((angle - ref + np.pi) % (2 * np.pi)) + ref - np.pi

def shift_keplerian(k, omega_ref, raan_ref):
    k_shifted = k.copy()
    k_shifted[3] = unwrap_to_ref(k[3], omega_ref)
    k_shifted[4] = unwrap_to_ref(k[4], raan_ref)
    return k_shifted

def get_maximally_separated_orbit(df, n_samples=5000, top_k=5, return_diagnostics=True):
    all_keplers = [get_keplerian_array_from_tle(row) for _, row in df.iterrows()]
    if len(all_keplers) < 2:
        print("Not enough orbits for max-min optimization.")
        return (None, None)

    min_a = min(k[0] for k in all_keplers)
    min_e = min(k[1] for k in all_keplers)
    min_i = min(k[2] for k in all_keplers)
    max_a = max(k[0] for k in all_keplers)
    max_e = max(k[1] for k in all_keplers)
    max_i = max(k[2] for k in all_keplers)
    
    # apply safe bounds check for equal bounds values
    min_a, max_a = safe_bounds(min_a, max_a)
    min_e, max_e = safe_bounds(min_e, max_e)
    min_i, max_i = safe_bounds(min_i, max_i)
    
    # # Angle wraparound resolution 
    omega_vals = [k[3] for k in all_keplers]
    raan_vals = [k[4] for k in all_keplers]
    # # get omega and raan reference angles as just the mean
    omega_ref = np.arctan2(np.mean(np.sin(omega_vals)), np.mean(np.cos(omega_vals)))
    raan_ref = np.arctan2(np.mean(np.sin(raan_vals)), np.mean(np.cos(raan_vals)))

    all_keplers_shifted = [shift_keplerian(k, omega_ref, raan_ref) for k in all_keplers]
    
    omega_vals_shifted = [k[3] for k in all_keplers_shifted]
    raan_vals_shifted = [k[4] for k in all_keplers_shifted]
    
    min_omega, max_omega = safe_bounds(min(omega_vals_shifted), max(omega_vals_shifted))
    min_raan, max_raan = safe_bounds(min(raan_vals_shifted), max(raan_vals_shifted))

    bounds = [
        (min_a, max_a),
        (min_e, max_e),
        (min_i, max_i),
        (min_omega, max_omega),
        (min_raan, max_raan),
    ]

    candidates = sample_maxmin(
        all_keplers_shifted, bounds, n_samples=n_samples, top_k=top_k
    )
    if not candidates:
        print("No max-separation candidates sampled.")
        return (None, None)

    print(
        "Top sampled max_separation radii: "
        + ", ".join(f"{r0:.6f}" for _, r0 in candidates)
    )

    x_star, r_star = None, -np.inf
    for i, (x0, r0) in enumerate(candidates):
        x_ref, r_ref = refine_maxmin(x0, all_keplers_shifted, bounds)
        print(f"Refined candidate {i + 1}/{len(candidates)}: {r0:.6f} -> {r_ref:.6f}")
        if r_ref > r_star:
            r_star = r_ref
            x_star = x_ref

    x_star = x_star.copy()
    x_star[3] %= (2 * np.pi)
    x_star[4] %= (2 * np.pi)
    
    
    print(f"Best refined max_separation radius: {r_star:.6f}")
    print(
        "Maximally separated Keplerian Elements: "
        "{a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f;}"
        % tuple(x_star[:5])
    )

    avg_epoch = calculate_average_epoch(df)
    initialDate = datetime_to_absolutedate(avg_epoch)
    ref_row = df.iloc[0]
    satrec = Satrec.twoline2rv(ref_row["line1"], ref_row["line2"])
    mean_anomaly = satrec.mo
    
    line1, line2 = convert_kep_to_tle(
        x_star,
        mean_anomaly,
        initialDate,
        bStar=average_bstar(df),
    )

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
