import pandas as pd
import pickle
from data_handler import DataHandler


if __name__ == "__main__":
    with open("data/cluster_results/hdbscan_obj.pkl", "rb") as f:
        hdbscan_obj = pickle.load(f)

    data_handler = DataHandler()
    data_dict = data_handler.load_data()

    orbit_df = data_dict["orbit_df"]

    # use the loaded hdbscan_obj, assuming it has a .labels_ attribute
    orbit_df["hdbscan_labels"] = hdbscan_obj.labels

    # save to CSV (no index column)
    orbit_df.to_csv("data/orbit_with_hdbscan_labels.csv", index=False)
