import torch.distributed.rpc as rpc
import torch
import torch.nn as nn
from models import MainModel
from torch.distributed.rpc import RRef
import os
from debugPrint import debug as print
from debugPrint import Log


class HIE(object):
    """
    Dummy HIE Class definition to avoid definition errors.
    """

    def __init__(self, server, central_model_rrefs, rank, args):
        ...

    def train(self, last_hie_rref, last_hie_id):
        ...

    def give_weights(self):
        ...

    def eval(self):
        ...

    def load_data(self, args):
        ...


class Central(object):
    """
    Central class definition
    """

    def __init__(self, args):
        # Remote reference to central server
        self.server = RRef(self)

        # Creating model object and storing it in object variable
        self.model = MainModel()
        print(self.model)
        self.model.apply(init_params)
        # Remote ref to central model parameters, used for defining backward pass in HIE
        model_rrefs = list(map(RRef, self.model.parameters()))

        # Map of HIEs' ranks to remote references to HIEs.
        self.hies = {
            rank
            + 1: rpc.remote(
                f"hie{rank+1}", HIE, (self.server, model_rrefs, rank + 1, args)
            )
            for rank in range(args.client_num_in_total)
        }

        # Previously trained HIE ID
        self.last_hie_id = None

        # World size
        self.client_num_in_total = args.client_num_in_total

        # model performance variable, used for saving the best model in EVAL.
        self.highestAcc = -1
        self.highestF1Score = -1

    def train_request(self, client_id):
        """
        call the train request from hie
        """
        # print(f"Train Request for hie{client_id}")
        if self.last_hie_id is None:
            print(f"HIE{client_id} is the first client", Log.WRN)
            # Run the forward pass on the given client with random weight initialization
            self.hies[client_id].rpc_sync(timeout=0).train(None, None)
        else:
            # Run the forward pass on the given client, use model params from last HIE
            self.hies[client_id].rpc_sync(timeout=0).train(
                self.hies[self.last_hie_id], self.last_hie_id
            )
        # Change last client ID to current for next pass
        self.last_hie_id = client_id

    def eval_request(self, itr: int):
        """
        Same as train request, but for validation evaluation
        """
        print("Initializing Evaluation of all hies")
        total = []
        num_corr = []
        f1Scores = []
        # Run the evaluation pass on all the HIEs with their current model parameters.
        check_eval = [
            self.hies[client_id].rpc_async(timeout=0).eval()
            for client_id in range(1, self.client_num_in_total + 1)
        ]

        # condense all the HIEs' results into single report.
        for check in check_eval:
            corr, tot, f1Score = check.wait()
            total.append(tot)
            num_corr.append(corr)
            f1Scores.append(f1Score)
        tempHighestAcc = sum(num_corr) / sum(total)
        currentF1Score = sum(f1Scores) / len(f1Scores)
        print("Accuracy over all data: {:.3f}".format(tempHighestAcc))

        # SaveBestModel code.
        if currentF1Score > self.highestF1Score:
            print(f"Saving current model at itr {itr}")
            os.makedirs(f"saves/itr{itr}/", exist_ok=True)
            # Save central's params
            torch.save(self.model.state_dict(), f"saves/itr{itr}/central.pt")
            # Get last HIE's weights
            m1Weights, m2Weights = (
                self.hies[self.last_hie_id].rpc_sync(timeout=0).give_weights()
            )

            # for i in [m1Weights, m2Weights, type(m1Weights), type(m2Weights)]:
            #     print(i)

            # Save last HIE's weights
            torch.save(m1Weights, f"saves/itr{itr}/m1.pt")
            torch.save(m2Weights, f"saves/itr{itr}/m2.pt")
            with open(f"saves/itr{itr}/modelPerf.txt", "w") as f:
                f.write(
                    f"Accuracy: {tempHighestAcc}\nF1Score:{currentF1Score}\nTotal Correct Predictions:{sum(num_corr)}"
                )
            # Update best accuracy.
            self.highestAcc = tempHighestAcc

    def train(self, fromInput):
        """
        Runs forward pass over the model
        """
        # print(fromInput.shape)
        return self.model(fromInput)


def init_params(m):
    if type(m) in [nn.Linear, nn.Conv2d]:
        # Random weight initialisation
        m.weight.data = torch.randn(m.weight.size()) * 0.01
        m.bias.data = torch.zeros(m.bias.size())
