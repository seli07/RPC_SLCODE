import torch.distributed.rpc
import torch
import torch.nn as nn
import pandas as pd

from models import InputEncoder, OutputDecoder
from torch.distributed.rpc import RRef
import torch.distributed.autograd as dist_autograd
from torch.distributed.optim.optimizer import DistributedOptimizer
from torch.utils.data import DataLoader
import torchvision as tv


import numpy as np

# from collections import Counter
# from copy import deepcopy
from tqdm.auto import tqdm
from debugPrint import debug as print
from debugPrint import Log
from sklearn.metrics import f1_score as f1score

# from torch.profiler import profile, record_function, ProfilerActivity


class HIE(object):
    def __init__(self, server, central_model_rrefs, rank, args):
        self.client_id = rank

        # setting number of epochs to class var
        self.epochs = args.epochs

        # Remote reference to central server
        self.central = server

        # Input and output decoder model def
        self.inputModel = InputEncoder()
        self.outputModel = OutputDecoder()
        print(self.inputModel)
        print(self.outputModel)

        self.inputModel.apply(init_params)
        self.outputModel.apply(init_params)
        # Loss function definition
        self.criterion = nn.BCELoss()

        # Distributed optimizer definition
        self.dist_optimizer = DistributedOptimizer(
            torch.optim.SGD,
            list(map(RRef, self.outputModel.parameters()))
            + central_model_rrefs
            + list(map(RRef, self.inputModel.parameters())),
            lr=args.lr,
            momentum=0.9,
        )
        # self.printSize = args.printSize
        # Dataloader definition call.
        self.loadBioData()
        self.args = args

    def train(self, last_hie_rref, last_hie_id):
        """
        Forward pass function
        """

        if last_hie_rref is None:
            # If none, this is the first hie to be trained in line
            print(f"hie{self.client_id} is first client to train")

        else:
            # else, get input, output model params from the previous HIE and load them
            print(f"hie{self.client_id} receiving weights from hie{last_hie_id}")
            model1_weights, model2_weights = last_hie_rref.rpc_sync().give_weights()
            self.inputModel.load_state_dict(model1_weights)
            self.outputModel.load_state_dict(model2_weights)

        # Train loop for given number of epochs
        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1}")
            for _, data in tqdm(
                enumerate(self.train_dataloader),
                total=np.ceil(
                    len(self.train_dataloader.dataset) / self.args.batch_size
                ),
            ):
                inputData, labels = data

                with dist_autograd.context() as context_id:
                    # Input decoder forward pass
                    inputResult = self.inputModel(inputData)

                    # Central model forward pass
                    bulkOutput = self.central.rpc_sync().train(
                        inputResult
                    )  # model(activation_hie1)

                    # Output decoder forward pass
                    output = self.outputModel(bulkOutput)

                    # Calculate loss
                    loss = self.criterion(output, labels.unsqueeze(1))

                    # run the backward pass
                    dist_autograd.backward(context_id, [loss])
                    self.dist_optimizer.step(context_id)

    def give_weights(self):
        """
        Returns input and output model state parameters
        """
        # print(self.inputModel.state_dict())
        return [self.inputModel.state_dict(), self.outputModel.state_dict()]

    def eval(self):
        """
        Evaluation method, runs eval on currently trained model params.
        """
        correct = 0
        total = 0
        f1s = 0
        # since we're not training, we don't need to calculate the gradients for our outputs
        with torch.no_grad():
            for i, data in enumerate(self.test_dataloader):
                images, labels = data
                # calculate outputs by running images through the network
                activation_hie1 = self.inputModel(images)
                activation_central = self.central.rpc_sync().train(
                    activation_hie1
                )  # model(activation_hie1)
                outputs = self.outputModel(activation_central) > 0.5
                # the class with the highest energy is what we choose as prediction
                # _, predicted = torch.max(outputs.data, 1)
                # _, groundTruths = torch.max(labels, 1)
                total += labels.size(0)
                # print(str(outputs.shape) +
                #   str(predicted.shape) + str(labels.shape))
                # print(labels)
                # print(outputs)
                correct += (outputs == labels).sum().item()
                f1s += f1score(labels, outputs, average="weighted")
                # correct += ((predicted > 0.5) == groundTruths).sum().item()

        print(f"hie{self.client_id} Evaluating Data: {round(correct / total, 3)}")
        return correct, total, f1s / (i + 1)

    def loadBioData(self):
        """
        Loads and normalizes data
        """
        data = pd.read_csv("examplefile.csv")

        data.pop("BLOODPRESSURE")

        def replaceFunction(column: pd.core.series.Series):
            if (
                column.name == "PREGNANCIES"
                or column.name == "INSULIN"
                or column.name == "OUTCOME"
            ):
                return column.replace("YES", 1).replace("NO", 0)
            elif column.name == "GENDER":
                return column.replace("FEMALE", 1).replace("MALE", 0)
            else:
                return column / max(column)

        data = data.apply(replaceFunction)
        y = data.pop("OUTCOME").to_numpy()
        data = data.to_numpy()
        # print(data)
        # print(y)
        self.train_dataloader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(data, dtype=torch.float),
                torch.tensor(y, dtype=torch.float),
            ),
            shuffle=True,
            batch_size=1,
        )
        self.test_dataloader = DataLoader(
            torch.utils.data.TensorDataset(
                torch.tensor(data, dtype=torch.float),
                torch.tensor(y, dtype=torch.float),
            ),
            shuffle=True,
            batch_size=1,
        )


class Central(object):
    """
    Dummy definition of central class, to overcome NotImplemented Error.
    """

    def __init__(self, args):
        ...

    def train_request(self, client_id):
        ...

    def eval_request(self):
        ...

    def train(self, x):
        ...


def init_params(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        # Random weight initialisation
        m.weight.data = torch.randn(m.weight.size()) * 0.01
        m.bias.data = torch.zeros(m.bias.size())
