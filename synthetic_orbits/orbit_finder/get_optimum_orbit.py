"""
Given a list of debris or objects, find which one has the 
smallest distance to all others and return the six 
Keplerian elements of that object.
"""

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from synthetic_orbits.orbit_finder.optimum_orbit_tle import convert_kep_to_tle, test_orbit
from synthetic_orbits.orbit_finder.DMT import VectorizedKeplerianOrbit
from sgp4.api import Satrec
from datetime import datetime, timedelta, timezone
from orekit.pyhelpers import datetime_to_absolutedate
from scipy.optimize import minimize
    
def get_keplerian_array_from_tle(row):
    """
    Given a row with the line1 and line2, return the keplerian elements as a 6-element array.
    The 6 elements are: [a, e, i, omega, raan, M]
    Here, a, e, i, omega, and raan are extracted directly and M is set to 0 as a placeholder.
    """

    line1_array = np.array([row['line1']])
    line2_array = np.array([row['line2']])
    orbit = VectorizedKeplerianOrbit(line1_array, line2_array)
    # Extract the parameters from the first (and only) satellite in the VectorizedKeplerianOrbit instance.
    return np.array([orbit.a[0], orbit.e[0], orbit.i[0], orbit.omega[0], orbit.raan[0], 0])

def get_initial_candidate(df):
    print("df size: ", df.shape)
    
    # get distance metric
    line1 = df['line1'].values
    line2 = df['line2'].values
    
    orbits = VectorizedKeplerianOrbit(line1, line2)
    
    distance_matrix = VectorizedKeplerianOrbit.DistanceMetric(orbits, orbits)
    
    # get the average distance for each orbit
    avg_distance = np.mean(distance_matrix, axis=1)
    
    # sort indexes by average distance in ascending order
    sorted_indexes = np.argsort(avg_distance)
    
    # get the first orbit as the initial candidate
    initial_candidate = df.iloc[sorted_indexes[0]]
    
    print("Initial candidate: ", initial_candidate['line1'])
    
    return initial_candidate

def calculate_average_epoch(df):
    epoch_times = []
    for _, row in df.iterrows():
        satrec = Satrec.twoline2rv(row['line1'], row['line2'])
        # Convert epoch to datetime
        year = 2000 + satrec.epochyr if satrec.epochyr < 100 else satrec.epochyr
        dt = datetime(year, 1, 1, tzinfo=timezone.utc) + timedelta(days=satrec.epochdays - 1)
        epoch_times.append(dt)
    
    # Calculate average datetime
    if not epoch_times:
        # Fallback to current time if no epochs
        print("No epochs found in TLE data. Using current time as average epoch.")	
        return datetime.now(timezone.utc)
    
    # Convert datetimes to timestamps
    timestamps = [dt.timestamp() for dt in epoch_times]
    average_timestamp = sum(timestamps) / len(timestamps)
    return datetime.fromtimestamp(average_timestamp, timezone.utc)

def get_optimum_orbit(df, return_diagnostics=False):
    """
    Find an optimized orbit that minimizes the average distance to all input orbits
    using multiple initial guesses.
    
    Parameters:
    - df: DataFrame with TLE data (line1, line2, etc.).
    
    Returns:
    - df: Updated DataFrame with the optimized orbit appended.
    """
    # Collect Keplerian elements for all orbits
    all_keplers = [get_keplerian_array_from_tle(row) for _, row in df.iterrows()]
    if len(all_keplers) < 2:
        print("Not enough orbits for optimization.")
        return None

    # Compute bounds based on all Keplerian elements
    min_a = min([k[0] for k in all_keplers])
    min_e = min([k[1] for k in all_keplers])
    min_i = min([k[2] for k in all_keplers])
    min_omega = min([k[3] for k in all_keplers])
    min_raan = min([k[4] for k in all_keplers])
    max_a = max([k[0] for k in all_keplers])
    max_e = max([k[1] for k in all_keplers])
    max_i = max([k[2] for k in all_keplers])
    max_omega = max([k[3] for k in all_keplers])
    max_raan = max([k[4] for k in all_keplers])

    lower_bounds = [min_a, min_e, min_i, min_omega, min_raan, 0]
    upper_bounds = [max_a, max_e, max_i, max_omega, max_raan, 2 * np.pi]

    # Optionally, add a margin to bounds (e.g., 10%) to widen the search space
    # margin = 0.1
    # lower_bounds = [min_a - margin * (max_a - min_a), max(0, min_e - margin * (max_e - min_e)), ...]
    # upper_bounds = [max_a + margin * (max_a - min_a), max_e + margin * (max_e - min_e), ...]

    # Run optimization from each initial guess
    best_result = None
    best_cost = np.inf

    # Optionally, include the original initial candidate explicitly
    initial_candidate = get_initial_candidate(df)
    initial_keplerian = get_keplerian_array_from_tle(initial_candidate)
    print("Initial candidate Keplerian Elements: {a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f; v: %.6f;}" % 
          tuple(initial_keplerian))

    for initial_guess in all_keplers:
        result = find_optimum_keplerian(initial_guess, all_keplers, lower_bounds, upper_bounds)
        print(f"Cost for initial guess {initial_guess[:5]}: {result.cost:.6f}")
        if result.cost < best_cost:
            best_cost = result.cost
            best_result = result.x

    optimum_keplerian = best_result
    print("Optimized Keplerian Elements: {a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f; v: %.6f;}" % 
          tuple(optimum_keplerian))


    # === CORRECT verification (same space as optimizer) ===
    means_real = [mean_sq_distance_kepler(k, all_keplers) for k in all_keplers]
    mean_opt  = mean_sq_distance_kepler(optimum_keplerian, all_keplers)

    print(f"Optimized mean distance (Kepler space): {mean_opt:.6f}")
    print(f"Best real mean distance (Kepler space): {min(means_real):.6f}")

    if mean_opt <= min(means_real):
        print("Kepler-space verification PASSED (Fréchet mean found).")
    else:
        print("Kepler-space verification FAILED (local minimum or convergence issue).")

    # Kepler-space ranking (same metric and functional as optimizer)
    kepler_candidates = all_keplers + [optimum_keplerian]
    kepler_means      = [mean_sq_distance_kepler(k, all_keplers) for k in kepler_candidates]

    order = np.argsort(kepler_means)
    print("\nKepler-space ranking by mean *squared* distance:")
    for rank, idx in enumerate(order, start=1):
        label = "OPT" if idx == len(all_keplers) else f"REAL_{idx}"
        print(f"{rank:2d}. {label}  mean_sq = {kepler_means[idx]:.6f}")



    # Verification: compare average distances
    # def verify_orbit(initial, optimized, others):
    #     initial_complete = make_complete_orbit(initial)
    #     optimized_complete = make_complete_orbit(optimized)
    #     initial_orbit = VectorizedKeplerianOrbit(initial_complete)
    #     optimized_orbit = VectorizedKeplerianOrbit(optimized_complete)
        
    #     initial_dists = []
    #     optimized_dists = []
    #     for k in others:
    #         other_complete = make_complete_orbit(k)
    #         other_orbit = VectorizedKeplerianOrbit(other_complete)
    #         initial_dists.append(VectorizedKeplerianOrbit.DistanceMetric(initial_orbit, other_orbit))
    #         optimized_dists.append(VectorizedKeplerianOrbit.DistanceMetric(optimized_orbit, other_orbit))
        
    #     return np.mean(initial_dists), np.mean(optimized_dists)

    # avg_initial, avg_optimized = verify_orbit(initial_keplerian, optimum_keplerian, all_keplers)
    # print(f"Initial candidate average distance: {avg_initial:.6f}")
    # print(f"Optimized orbit average distance: {avg_optimized:.6f}")
    
    # if avg_optimized < avg_initial:
    #     print("Verification: Optimization successful - optimized orbit has a lower average distance.")
    # else:
    #     print("Verification: Optimization unsuccessful - no improvement detected.")


    diagnostics = {
        "N": len(all_keplers),
        "best_real_cost": min(means_real),
        "optimized_cost": mean_opt,
        "success": mean_opt <= min(means_real),
    }

    if return_diagnostics:
        # For diagnostics-only use, do NOT build a TLE or modify df
        return diagnostics


    # Convert to TLE
    avg_epoch = calculate_average_epoch(df)
    year_start = datetime(avg_epoch.year, 1, 1, tzinfo=timezone.utc)
    days_since_year_start = (avg_epoch - year_start).total_seconds() / (24 * 60 * 60)
    epoch_yr = avg_epoch.year
    epoch_days = days_since_year_start + 1  # 1-indexed

    satrec = Satrec.twoline2rv(initial_candidate['line1'], initial_candidate['line2'])
    mean_anomaly = satrec.mo
    optimum_keplerian[5] = mean_anomaly

    initialDate = datetime_to_absolutedate(avg_epoch)
    line1, line2 = convert_kep_to_tle(optimum_keplerian, mean_anomaly, initialDate)

    # Append optimized orbit to DataFrame
    optimum_orbit_entry = {
        'satNo': '99999',
        'name': "Optimized",
        'line1': line1,
        'line2': line2,
        'correlated': True,
        'dataset': initial_candidate.get('dataset')
                  if isinstance(initial_candidate, pd.Series) and 'dataset' in initial_candidate
                  else None
    }

    optimum_df = pd.DataFrame([optimum_orbit_entry])
    
    df = pd.concat([df, optimum_df], ignore_index=True)

    print("Optimized orbit added to the TLE data.")

    return df
     
     
def evaluate_optimizer_all_clusters(self, df, min_cluster_size=2):
    results = []

    for label, df_cluster in df.groupby("label"):
        if label == -1:
            continue

        N = len(df_cluster)
        if N < min_cluster_size:
            continue

        try:
            diagnostics = get_optimum_orbit(
                df_cluster.copy(),
                return_diagnostics=True
            )
            diagnostics["label"] = label
            results.append(diagnostics)

        except Exception as e:
            results.append({
                "label": label,
                "N": N,
                "error": str(e),
                "success": False
            })

    return pd.DataFrame(results)

     
def residuals_keplerian(x, other_keplers):
    
    # ensure the mean anomaly is 0
    x_copy = x.copy()
    
    candidate = VectorizedKeplerianOrbit(make_complete_orbit(x_copy))
    residuals = []
    for kepler in other_keplers:
        other = VectorizedKeplerianOrbit(make_complete_orbit(kepler))
        dist = VectorizedKeplerianOrbit.DistanceMetric(candidate, other)
        residuals.append(dist)
    # Flatten the residuals to 1-D and center them
    return np.array(residuals).ravel()

def find_optimum_keplerian(initial_guess, other_keplers, lower_bounds, upper_bounds):
    """
    Optimize Keplerian elements to minimize residuals against other_keplers.
    
    Parameters:
    - initial_guess: Array of [a, e, i, omega, raan, M] to start optimization.
    - other_keplers: List of Keplerian arrays to compare against.
    - lower_bounds: List of lower bounds for [a, e, i, omega, raan, M].
    - upper_bounds: List of upper bounds for [a, e, i, omega, raan, M].
    
    Returns:
    - result: SciPy OptimizeResult object with optimized parameters and cost.
    """
    result = least_squares(
        residuals_keplerian,
        initial_guess,
        args=(other_keplers,),
        bounds=(lower_bounds, upper_bounds),
        jac='3-point'
    )
    return result

def make_complete_orbit(opt_array):
    """
    Convert a 6-element optimized orbit [a, e, i, omega, raan, M]
    into a complete 7-element array [a, e, i, omega, raan, q, p].
    Here, p = a * (1 - e**2) and q = a * (1 - e), with q serving as a proxy.
    """
    
    a, e, i, omega, raan = opt_array[:5]
    p = a * (1 - e**2)
    q = a * (1 - e)
    return np.array([a, e, i, omega, raan, q, p])

def verify_keplerian_elements(before_keplerians, after_keplerians):

    """
    Take the keplerians before put into TLE, then extract keplerians from TLE and compare the before and after to check consistency
    """
    # get after_keplerians    
    keplerian_options = ['a', 'e', 'i', 'omega', 'raan']
    
    failed_keplerians = False
    for i in range(5):
        if abs(before_keplerians[i] - after_keplerians[i]) > 1e-6:
            print(f"Verification failed: Keplerian element {keplerian_options[i]} does not match.")
            # print them out
            print("Before: ", before_keplerians[i])
            print("After: ", after_keplerians[i])
            print("\n")
            failed_keplerians = True
            
    if not failed_keplerians:
        print("Verification successful: Keplerian elements are consistent before and after optimization.")
    else:
        print("Verification failed: Keplerian elements are inconsistent before and after optimization.")

def mean_sq_distance_kepler(candidate_k, all_keplers):
    cand_orbit = VectorizedKeplerianOrbit(make_complete_orbit(candidate_k))
    d2 = []
    for kepler in all_keplers:
        other_orbit = VectorizedKeplerianOrbit(make_complete_orbit(kepler))
        d = VectorizedKeplerianOrbit.DistanceMetric(cand_orbit, other_orbit)
        d2.append(d**2)
    return np.mean(d2)





    """ 
        Everything below here is for the min-max
    """
    
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
        x = np.array([
            np.random.uniform(lo, hi)
            for lo, hi in bounds
        ])

        r_x = min_distance_to_catalog(x, kepler_list)

        if r_x > best_r:
            best_r = r_x
            best_x = x

    return best_x, best_r


def refine_maxmin(x0, kepler_list, bounds):
    def objective(x):
        return -min_distance_to_catalog(x, kepler_list)

    res = minimize(
        objective,
        x0=x0,
        bounds=bounds,
        method="L-BFGS-B"
    )

    return res.x, -res.fun

def get_maximally_separated_orbit(df, n_samples=5000, return_diagnostics=True):
    """
    Find a synthetic orbit that maximizes distance to its nearest neighbour
    (largest empty ball in Keplerian element space).
    """

    all_keplers = [
        get_keplerian_array_from_tle(row)
        for _, row in df.iterrows()
    ]

    if len(all_keplers) < 2:
        print("Not enough orbits for max-min optimization.")
        return df

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

    # --- step 1: Monte Carlo largest empty ball ---
    x0, r0 = sample_maxmin(all_keplers, bounds, n_samples=n_samples)
    print(f"Best sampled void radius: {r0:.6f}")

    # --- step 2: local refinement ---
    x_star, r_star = refine_maxmin(x0, all_keplers, bounds)
    print(f"Refined void radius: {r_star:.6f}")

    print(
        "Maximally separated Keplerian Elements: "
        "{a: %.6f; e: %.6f; i: %.6f; pa: %.6f; raan: %.6f; v: %.6f;}"
        % tuple(x_star)
    )

    # --- convert to TLE ---
    avg_epoch = calculate_average_epoch(df)
    initialDate = datetime_to_absolutedate(avg_epoch)

    # reuse mean anomaly from a real satellite
    ref_row = df.iloc[0]
    satrec = Satrec.twoline2rv(ref_row['line1'], ref_row['line2'])
    mean_anomaly = satrec.mo
    x_star[5] = mean_anomaly

    line1, line2 = convert_kep_to_tle(x_star, mean_anomaly, initialDate)

    # --- append to DataFrame ---
    void_entry = {
        'satNo': '99999',
        'name': 'MaximallySeparated',
        'line1': line1,
        'line2': line2,
        'correlated': True,
        'dataset': ref_row.get('dataset') if 'dataset' in ref_row else None
    }

    df = pd.concat([df, pd.DataFrame([void_entry])], ignore_index=True)

    # test_orbit(df)
    
    if return_diagnostics:
        diagnostics = evaluate_void_orbit(x_star, all_keplers)
        return df, diagnostics
    return df

def cluster_spacing_stats(kepler_list):
    """
    Compute nearest-neighbour distance statistics inside a cluster.
    """
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

def evaluate_void_orbit(void_kepler, kepler_list, eps=1e-12):
    """
    Quantify how 'void-like' a synthetic orbit is.
    """
    # nearest neighbour distance of synthetic orbit
    r_star = min_distance_to_catalog(void_kepler, kepler_list)

    stats = cluster_spacing_stats(kepler_list)

    percentile = np.mean(stats["nn_distances"] < r_star) * 100

    median = stats["median"]
    if median < eps:
        ratio_to_median = np.inf   # or np.nan, depending on what you prefer
    else:
        ratio_to_median = r_star / median

    return {
        "r_star": float(r_star),
        "percentile_vs_cluster": float(percentile),
        "ratio_to_median_spacing": float(ratio_to_median),
        "cluster_stats": stats,
    }

