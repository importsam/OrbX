import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
_src = _repo_root / "src"
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from scoring import scoring_main
from ionop_czml import ionop_czml
from live.build_czml import build_czml_live
import pandas as pd

from orbx.clustering import cluster
from orbx.clustering.data_handling.DataHandler import DataHandler


def _norm_norad(x) -> str:
    return str(x).split(".")[0].zfill(5)


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

        """ 
        Apogee band defines who gets clustered; full scoring df goes to CZML with label -1
        for satellites outside that band (or unmapped). Keplerian work uses copies only.
        """
        
        print(len(results_df))

        prep = results_df.rename(columns={"TLE_LINE1": "line1", "TLE_LINE2": "line2"})
        keplerian_df = DataHandler().tle_to_keplerian(prep.copy())
        keplerian_df = keplerian_df[(keplerian_df["apogee"] >= 300) & (keplerian_df["apogee"] <= 700)]

        allowed_satnos = set(keplerian_df["satNo"].map(_norm_norad))
        cluster_df = results_df[
            results_df["NORAD_CAT_ID"].map(_norm_norad).isin(allowed_satnos)
        ].copy()
        print(f"clustering subset (apogee 300-700 km): {len(cluster_df)}")

        cluster_input = cluster_df.rename(
            columns={"TLE_LINE1": "line1", "TLE_LINE2": "line2"}
        )

        labels = cluster(cluster_input, verbose=True)

        dh = DataHandler()
        df_kep = dh.tle_to_keplerian(cluster_input.copy())
        sat_nos = list(df_kep["satNo"].unique())
        if len(sat_nos) != len(labels):
            raise ValueError(f"label alignment mismatch: {len(sat_nos)} sat numbers vs {len(labels)} labels")
        label_by_sat = {_norm_norad(s): int(lab) for s, lab in zip(sat_nos, labels)}
        results_df["label"] = results_df["NORAD_CAT_ID"].map(lambda nid: label_by_sat.get(_norm_norad(nid), -1))

        print("Number of clusters: ", len(results_df["label"].unique()))
        print(results_df.columns)
        build_czml_live(results_df)
        
    except Exception as e:
        print(f"Error: {e}")
    
    ACCESSTOKEN_live = 'eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJqdGkiOiJkYjlmZDMxMy1lY2RmLTQyMDMtYTZhZS0wMmY4MzcyZDc4ZGEiLCJpZCI6MjQwODIwLCJpYXQiOjE3NzcxOTU0NTB9.M5SBzZZAebNNCMhZbSUqozkEm8LIul3abqANtS8URrU'
    try:
        ionop_czml(ACCESSTOKEN_live, 'live')
    except Exception as e:
        print(f"Error: {e}")
