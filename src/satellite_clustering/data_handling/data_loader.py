from src.satellite_clustering.data_handling.tle_parser import TLEParser

# MAKE THIS IMPORT CLUSTER CONFIG INSTEAD OF PASSING IT DOWN

# file name doesn't match this lol
class DataLoader():
    def __init__(self, cluster_config):
        self.tle_parser = TLEParser("Space-Track")
    
    