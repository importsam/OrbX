import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sgp4.api import Satrec
from datetime import datetime, timedelta, timezone

""" 
This file holds standalone functions for finding the frechet mean orbit for a cluster of orbits.
"""

try:
    from orekit.pyhelpers import datetime_to_absolutedate
except ImportError:
    raise ImportError(
        "Orekit is required for synthetic orbit generation. "
        "Please install it using: conda install -c conda-forge orekit"
    )

from orbx.synthetic_orbits.orbit_finder.DMT import VectorizedKeplerianOrbit
from orbx.synthetic_orbits.orbit_finder.optimum_orbit_tle import (
    average_bstar,
    convert_kep_to_tle,
)


# ---------- helpers ----------
def get_keplerian_array_from_tle(row):
    line1_array = np.array([row["line1"]])
    line2_array = np.array([row["line2"]])
    orbit = VectorizedKeplerianOrbit(line1_array, line2_array)
    return np.array([orbit.a[0], orbit.e[0], orbit.i[0],
                     orbit.omega[0], orbit.raan[0]])


def make_complete_orbit(opt_array):
    a, e, i, omega, raan = opt_array[:5]
    p = a * (1 - e**2)
    q = a * (1 - e)
    return np.array([a, e, i, omega, raan, q, p])


def mean_sq_distance_kepler(candidate_k, all_keplers):
    cand_orbit = VectorizedKeplerianOrbit(make_complete_orbit(candidate_k))
    d2 = []
    for kepler in all_keplers:
        other_orbit = VectorizedKeplerianOrbit(make_complete_orbit(kepler))
        d = VectorizedKeplerianOrbit.DistanceMetric(cand_orbit, other_orbit)
        d2.append(d**2)
    return np.mean(d2)

def residuals_keplerian(x, other_keplers):
    """Residuals for ordinary Fréchet: least_squares minimizes sum(r**2) = sum(q).
    """
    x_copy = x.copy()
    candidate = VectorizedKeplerianOrbit(make_complete_orbit(x_copy))
    residuals = []
    for kepler in other_keplers:
        other = VectorizedKeplerianOrbit(make_complete_orbit(kepler))
        q = VectorizedKeplerianOrbit.DistanceMetric(candidate, other)
        q = float(np.asarray(q).reshape(-1)[0])
        """q is returned as a squared distance, we root to avoid fourth-power loss."""
        residuals.append(np.sqrt(np.maximum(q, 0.0)))
    return np.asarray(residuals, dtype=float).ravel()

def find_optimum_keplerian(initial_guess, other_keplers, lower_bounds, upper_bounds):
    return least_squares(
        residuals_keplerian,
        initial_guess,
        args=(other_keplers,),
        bounds=(lower_bounds, upper_bounds),
        jac="3-point",
    )


def get_initial_candidate(df):
    print("df size: ", df.shape)
    line1 = df["line1"].values
    line2 = df["line2"].values
    orbits = VectorizedKeplerianOrbit(line1, line2)
    distance_matrix = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
    avg_distance = np.mean(distance_matrix, axis=1)
    sorted_indexes = np.argsort(avg_distance)
    initial_candidate = df.iloc[sorted_indexes[0]]
    print("Initial candidate: ", initial_candidate["line1"])
    return initial_candidate


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

# ---------- main API ----------

def optimize_frechet_kepler(all_keplers):
    """
    Pure numerical optimisation in Kepler space.
    Returns optimum_keplerian and a diagnostics dict.
    """
    if len(all_keplers) < 2:
        raise ValueError("Not enough orbits for optimisation.")

    # bounds
    all_keplers = list(all_keplers)
    
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

    """
    There's no starting point on a circle. 
    if two angles neighbour on 359 and 1 degree, there's a massive difference when 
    the circle treated as a straight line. We find the center of the cluster on this line
    and cut it at the opposite end. This addresses the wraparound issue.
    """
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
    
    lower_bounds = [min_a, min_e, min_i, min_omega, min_raan]
    upper_bounds = [max_a, max_e, max_i, max_omega, max_raan]

    # run from each initial guess
    best_result = None
    best_cost = np.inf

    for initial_guess in all_keplers_shifted:
        result = find_optimum_keplerian(initial_guess, all_keplers_shifted, lower_bounds, upper_bounds)
        print(f"Cost for initial guess {initial_guess[:5]}: {result.cost:.6f}")
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result.x

    optimum_keplerian = best_result.copy()
    optimum_keplerian[3] %= (2 * np.pi)
    optimum_keplerian[4] %= (2 * np.pi)
    
    print(
        "Optimized Keplerian Elements: "
        "{a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f;}"
        % tuple(optimum_keplerian)
    )

    # diagnostics
    means_real = [mean_sq_distance_kepler(k, all_keplers) for k in all_keplers]
    mean_opt = mean_sq_distance_kepler(optimum_keplerian, all_keplers)
    print(f"Optimized mean distance (Kepler space): {mean_opt:.6f}")
    print(f"Best real mean distance (Kepler space): {min(means_real):.6f}")

    if mean_opt <= min(means_real):
        print("Kepler-space verification PASSED (Fréchet mean found).")
    else:
        print("Kepler-space verification FAILED (local minimum or convergence issue).")

    # ranking
    kepler_candidates = all_keplers + [optimum_keplerian]
    kepler_means = [mean_sq_distance_kepler(k, all_keplers) for k in kepler_candidates]
    order = np.argsort(kepler_means)
    print("\nKepler-space ranking by mean *squared* distance:")
    for rank, idx in enumerate(order, start=1):
        label = "OPT" if idx == len(all_keplers) else f"REAL_{idx}"
        print(f"{rank:2d}. {label}  mean_sq = {kepler_means[idx]:.6f}")

    diagnostics = {
        "N": len(all_keplers),
        "best_real_cost": float(min(means_real)),
        "optimized_cost": float(mean_opt),
        "success": bool(mean_opt <= min(means_real)),
    }
    return optimum_keplerian, diagnostics


def frechet_orbit(df, return_diagnostics=False):
    """
    Cluster-level wrapper: takes a df of TLEs for one cluster.
    - If return_diagnostics=True: prints, returns diagnostics only.
    - Else: appends satNo=99999 Frechet orbit as TLE row.
    """
    all_keplers = [get_keplerian_array_from_tle(row) for _, row in df.iterrows()]
    if len(all_keplers) < 2:
        print("Not enough orbits for optimization.")
        return None

    initial_candidate = get_initial_candidate(df)
    initial_keplerian = get_keplerian_array_from_tle(initial_candidate)
    print(
        "Initial candidate Keplerian Elements: "
        "{a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f;}"
        % tuple(initial_keplerian)
    )

    optimum_keplerian, diagnostics = optimize_frechet_kepler(all_keplers)

    if return_diagnostics:
        return diagnostics

    # Mean anomaly is not optimised to we steal it from one of the real TLEs.
    avg_epoch = calculate_average_epoch(df)
    satrec = Satrec.twoline2rv(initial_candidate["line1"], initial_candidate["line2"])
    mean_anomaly = satrec.mo
    initialDate = datetime_to_absolutedate(avg_epoch)
    line1, line2 = convert_kep_to_tle(
        optimum_keplerian,
        mean_anomaly,
        initialDate,
        bStar=average_bstar(df),
    )

    opt_row = {
        "satNo": "99999",
        "name": "Fréchet Mean",
        "line1": line1,
        "line2": line2,
        "correlated": True,
        "dataset": initial_candidate.get("dataset")
        if isinstance(initial_candidate, pd.Series) and "dataset" in initial_candidate
        else None,
    }
    df = pd.concat([df, pd.DataFrame([opt_row])], ignore_index=True)
    print("Optimized orbit added to the TLE data.")
    return df
