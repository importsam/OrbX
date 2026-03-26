

from src.satellite_clustering.data_handling.tle_parser import TLEParser


class SatelliteLoader():
    def __init__(self, cluster_config):
        self.tle_parser = TLEParser("Space-Track")
    
    