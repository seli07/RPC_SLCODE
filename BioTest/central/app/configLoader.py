from importlib.machinery import SourceFileLoader
import os


class CentralConfig:
    def __init__(self, configFileLocation: str = None):
        if configFileLocation:
            conf = SourceFileLoader(
                os.path.basename(configFileLocation).split(".")[0], configFileLocation
            ).load_module()

            self.world_size = conf.world_size
            self.epochs = conf.epochs
            self.iterations = conf.iterations
            self.batch_size = conf.batch_size
            self.datapath = conf.datapath
            self.lr = conf.lr
            self.address = conf.address
            self.port = conf.port
        else:
            self.world_size = 2
            self.epochs = 4
            self.iterations = 3
            self.batch_size = 16
            self.datapath = "data/"
            self.lr = 0.001
            self.address = "localhost"
            self.port = 7732
