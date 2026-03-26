# frechet_orbit_finder.py

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from sgp4.api import Satrec
from datetime import datetime, timedelta, timezone
from orekit.pyhelpers import datetime_to_absolutedate

from synthetic_orbits.orbit_finder.DMT import VectorizedKeplerianOrbit
from synthetic_orbits.orbit_finder.optimum_orbit_tle import convert_kep_to_tle


# ---------- helpers ----------
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


def mean_sq_distance_kepler(candidate_k, all_keplers):
    cand_orbit = VectorizedKeplerianOrbit(make_complete_orbit(candidate_k))
    d2 = []
    for kepler in all_keplers:
        other_orbit = VectorizedKeplerianOrbit(make_complete_orbit(kepler))
        d = VectorizedKeplerianOrbit.DistanceMetric(cand_orbit, other_orbit)
        d2.append(d**2)
    return np.mean(d2)


def residuals_keplerian(x, other_keplers):
    x_copy = x.copy()
    candidate = VectorizedKeplerianOrbit(make_complete_orbit(x_copy))
    residuals = []
    for kepler in other_keplers:
        other = VectorizedKeplerianOrbit(make_complete_orbit(kepler))
        dist = VectorizedKeplerianOrbit.DistanceMetric(candidate, other)
        residuals.append(dist)
    return np.asarray(residuals).ravel()


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
    min_omega = min(k[3] for k in all_keplers)
    min_raan = min(k[4] for k in all_keplers)
    max_a = max(k[0] for k in all_keplers)
    max_e = max(k[1] for k in all_keplers)
    max_i = max(k[2] for k in all_keplers)
    max_omega = max(k[3] for k in all_keplers)
    max_raan = max(k[4] for k in all_keplers)

    lower_bounds = [min_a, min_e, min_i, min_omega, min_raan, 0.0]
    upper_bounds = [max_a, max_e, max_i, max_omega, max_raan, 2 * np.pi]

    # run from each initial guess
    best_result = None
    best_cost = np.inf

    for initial_guess in all_keplers:
        result = find_optimum_keplerian(initial_guess, all_keplers, lower_bounds, upper_bounds)
        print(f"Cost for initial guess {initial_guess[:5]}: {result.cost:.6f}")
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result.x

    optimum_keplerian = best_result
    print(
        "Optimized Keplerian Elements: "
        "{a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f; v: %.6f;}"
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


def get_optimum_orbit(df, return_diagnostics=False):
    """
    Cluster-level wrapper: takes a df of TLEs for one cluster.
    - If return_diagnostics=True: prints, returns diagnostics only.
    - Else: appends satNo=99999 Frechet orbit as TLE row.
    """
    all_keplers = [get_keplerian_array_from_tle(row) for _, row in df.iterrows()]
    if len(all_keplers) < 2:
        print("Not enough orbits for optimization.")
        return None if return_diagnostics else df

    initial_candidate = get_initial_candidate(df)
    initial_keplerian = get_keplerian_array_from_tle(initial_candidate)
    print(
        "Initial candidate Keplerian Elements: "
        "{a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f; v: %.6f;}"
        % tuple(initial_keplerian)
    )

    optimum_keplerian, diagnostics = optimize_frechet_kepler(all_keplers)

    if return_diagnostics:
        return diagnostics

    # build TLE for the Frechet orbit
    avg_epoch = calculate_average_epoch(df)
    satrec = Satrec.twoline2rv(initial_candidate["line1"], initial_candidate["line2"])
    mean_anomaly = satrec.mo
    optimum_keplerian[5] = mean_anomaly
    initialDate = datetime_to_absolutedate(avg_epoch)
    line1, line2 = convert_kep_to_tle(optimum_keplerian, mean_anomaly, initialDate)

    opt_row = {
        "satNo": "99999",
        "name": "Optimized",
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
