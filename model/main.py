import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from ionop_czml import ionop_czml
from live.build_czml import build_czml
import pandas as pd

from orbx.clustering import cluster
from orbx.synthetic_orbits import synthetic_orbit
from orbx.density import density
from orbx.clustering.data_handling.DataHandler import DataHandler

def _norm_norad(x) -> str:
    return str(x).split(".")[0].zfill(5)


# None = all non-noise clusters. Use this for testing on a smaller number of clusters first before scaling.
CLUSTER_SAMPLE_SIZE = 20

if __name__ == '__main__':
    # builds the czml files
    # try:
    #     # print("Scoring started")
    #     # scoring_main()
        
    # except Exception as e:
    #     print(f"Error: {e}")
    
    # now build the czml files
    try:
        results_df = pd.read_pickle("data/satellites_with_scores.pkl")

        print(results_df.columns)

        """ 
        Apogee band defines who gets clustered; full scoring df goes to CZML with label -1
        for satellites outside that band (or unmapped). Keplerian work uses copies only.
        """
        
        print(len(results_df))

        prep = results_df.rename(columns={"TLE_LINE1": "line1", "TLE_LINE2": "line2"})
        keplerian_df = DataHandler().tle_to_keplerian(prep.copy())
        keplerian_df = keplerian_df[(keplerian_df["apogee"] >= 300) & (keplerian_df["apogee"] <= 700)]

        allowed_satnos = set(keplerian_df["satNo"].map(_norm_norad))
        cluster_df = results_df[results_df["NORAD_CAT_ID"].map(_norm_norad).isin(allowed_satnos)].copy()
        print(f"clustering subset (apogee 300-700 km): {len(cluster_df)}")

        cluster_input = cluster_df.rename(columns={"TLE_LINE1": "line1", "TLE_LINE2": "line2"})

        labels = cluster(cluster_input, verbose=True)

        dh = DataHandler()
        df_kep = dh.tle_to_keplerian(cluster_input.copy())
        sat_nos = list(df_kep["satNo"].unique())
        if len(sat_nos) != len(labels):
            raise ValueError(f"label alignment mismatch: {len(sat_nos)} sat numbers vs {len(labels)} labels")
        label_by_sat = {_norm_norad(s): int(lab) for s, lab in zip(sat_nos, labels)}
        results_df["label"] = results_df["NORAD_CAT_ID"].map(lambda nid: label_by_sat.get(_norm_norad(nid), -1))

        # Synthetic orbits per non-noise cluster
        clustered_for_synth = cluster_df.rename(
            columns={"TLE_LINE1": "line1", "TLE_LINE2": "line2"}
        ).copy()
        clustered_for_synth["label"] = clustered_for_synth["NORAD_CAT_ID"].map(
            lambda nid: label_by_sat.get(_norm_norad(nid), -1)
        )

        clustered_work = clustered_for_synth[
            clustered_for_synth["label"] != -1
        ].copy()
        all_cluster_labels = sorted(clustered_work["label"].unique())


        if CLUSTER_SAMPLE_SIZE is not None and CLUSTER_SAMPLE_SIZE > 0:
            import numpy as np

            n_sample = min(CLUSTER_SAMPLE_SIZE, len(all_cluster_labels))
            active_labels = set(
                int(l)
                for l in np.random.choice(all_cluster_labels, size=n_sample, replace=False)
            )
            clustered_work = clustered_work[
                clustered_work["label"].isin(active_labels)
            ]
            print(
                f"Cluster sample enabled: {len(active_labels)}/{len(all_cluster_labels)} clusters"
            )
        else:
            active_labels = set(int(l) for l in all_cluster_labels)
            print(f"Using all {len(all_cluster_labels)} non-noise clusters")

        density_df = density(
            clustered_work[["line1", "line2", "label"]],
            verbose=False,
        )

        max_density = density_df["density"].max()
        if max_density and max_density > 0:
            density_df["density"] = density_df["density"] / max_density

        density_by_label = {
            int(row["label"]): float(row["density"])
            for _, row in density_df.iterrows()
        }
        print(f"Computed density for {len(density_by_label)} clusters")

        print("Computing synthetic orbits (frechet + max_separation)...")
        synth_df = synthetic_orbit(
            clustered_work[["line1", "line2", "label"]],
            mode=["frechet", "max_separation"],
            verbose=True,
        )
        print(f"Generated {len(synth_df)} synthetic orbit rows")

        # Reshape synthetic rows to match results_df columns
        synth_rows = []
        for _, row in synth_df.iterrows():
            synth_rows.append({
                "NORAD_CAT_ID": f"SYN_{row['synthetic_type']}_{int(row['label'])}",
                "OBJECT_NAME": f"{row['synthetic_type'].replace('_', ' ').title()} (cluster {int(row['label'])})",
                "TLE_LINE1": row["line1"],
                "TLE_LINE2": row["line2"],
                "prop_orbit_class": "LEO",
                "prop_uniqueness": None,
                "prop_rank": None,
                "uniqueness_range": "none",
                "neighbours": [],
                "label": int(row["label"]),
                "synthetic_type": row["synthetic_type"],
                "cluster_density": density_by_label.get(int(row["label"])),
            })

        if CLUSTER_SAMPLE_SIZE is not None:
            results_for_czml = results_df[
                results_df["label"].isin(active_labels)
            ].copy()
        else:
            results_for_czml = results_df.copy()

        results_for_czml["synthetic_type"] = None
        results_for_czml["cluster_density"] = results_for_czml["label"].map(
            lambda lab: density_by_label.get(int(lab)) if lab != -1 else None
        )
        combined_df = pd.concat([results_for_czml, pd.DataFrame(synth_rows)], ignore_index=True)

        print(f"Clusters in CZML: {len(active_labels)}")
        print(f"Total entities for CZML: {len(combined_df)} ({len(results_for_czml)} real + {len(synth_rows)} synthetic)")
        build_czml(combined_df)

    except Exception as e:
        print(f"Error: {e}")
    
    ACCESSTOKEN_live = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJkYjlmZDMxMy1lY2RmLTQyMDMtYTZhZS0wMmY4MzcyZDc4ZGEiLCJpZCI6MjQwODIwLCJpYXQiOjE3NzcxOTU0NTB9.M5SBzZZAebNNCMhZbSUqozkEm8LIul3abqANtS8URrU'
    try:
        ionop_czml(ACCESSTOKEN_live, 'live')
    except Exception as e:
        print(f"Error: {e}")
