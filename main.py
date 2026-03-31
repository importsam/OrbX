from src.satellite_clustering.Core import Core 
from src.satellite_clustering.data_handling.data_loader import DataLoader
from src.satellite_clustering.data_handling.tle_parser import TLEParser
from configs import ClusterConfig

if __name__ == "__main__":
    # get the df of TLEs 
    tle_parser = TLEParser("Space-Track")
    cluster_config = ClusterConfig()
    tle_parser.spacetrack_parse_file("3le_1126")
    
    df = tle_parser.df
    
    # cut df down 
    df = df[
        (df["inclination"] >= cluster_config.inclination_range[0])
        & (df["inclination"] <= cluster_config.inclination_range[1])
    ]
    df = df[
        (df["apogee"] >= cluster_config.apogee_range[0])
        & (df["apogee"] <= cluster_config.apogee_range[1])
    ]
    
    print(f"Loaded {len(df)} satellites in range - inc: {cluster_config.inclination_range}, apogee: {cluster_config.apogee_range}")
    
    # run clustering
    cluster_result = Core(cluster_config)._cluster(df)
    print(cluster_result.df.head())
     