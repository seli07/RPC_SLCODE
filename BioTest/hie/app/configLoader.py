from importlib.machinery import SourceFileLoader
import os


class HIEConfig:
    def __init__(self, configFileLocation: str = None):
        if configFileLocation:
            conf = SourceFileLoader(
                os.path.basename(configFileLocation).split(".")[0], configFileLocation
            ).load_module()

            self.world_size = conf.world_size
            self.rank = conf.rank
            self.address = conf.address
            self.port = conf.port
        else:
            self.world_size = 2
            self.rank = 1
            self.address = "localhost"
            self.port = 7732


class CentralConfig:
    def __init__(self, configFileLocation: str = None):
        self.world_size = ...
        self.epochs = ...
        self.iterations = ...
        self.batch_size = ...
        self.datapath = ...
        self.lr = ...
        self.address = ...
        self.port = ...
