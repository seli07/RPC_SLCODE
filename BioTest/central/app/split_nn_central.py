from debugPrint import debug as print
from debugPrint import Log
import os

# import argparse
import torch.distributed.rpc as rpc
from data_entities import Central
import configLoader

# Function for initializing environmental vars


def init_env(rank: int, addr: str, port: int):
    print(f"Initialize Meetup Spot, in dev rank {rank}", Log.INF)
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = str(port)


# Argument parser system
# parser = argparse.ArgumentParser(description="Split Learning Initialization")
# parser.add_argument(
#     "--world_size",
#     type=int,
#     default=3,
#     help="The world size which is equal to 1 server + (world size - 1) clients",
# )
# parser.add_argument(
#     "--epochs",
#     type=int,
#     default=1,
#     help="The number of epochs to run on the client training each iteration",
# )
# parser.add_argument(
#     "--iterations",
#     type=int,
#     default=5,
#     help="The number of iterations to communication between clients and server",
# )
# parser.add_argument(
#     "--batch_size",
#     type=int,
#     default=16,
#     help="The batch size during the epoch training",
# )
# parser.add_argument(
#     "--datapath",
#     type=str,
#     default="data/",
#     help="folder path to all the local datasets",
# )
# parser.add_argument(
#     "--lr", type=float, default=0.001, help="Learning rate of local client (SGD)"
# )
# parser.add_argument(
#     "--address", type=str, default="localhost", help="IP Address to listen on"
# )
# parser.add_argument("--port", type=int, default="5656", help="Port to listen on")


# args = parser.parse_args()

args = configLoader.CentralConfig("conf.py")

# Initializing env vars
init_env(0, args.address, args.port)


# Initializing env vars
rpc.init_rpc("central", rank=0, world_size=args.world_size)
print("Connection established to all HIEs", Log.SUC)

# Initializing Central Server Object
args.client_num_in_total = args.world_size - 1
CentralServer = Central(args)

# Creating Saves directory
os.makedirs("saves/", exist_ok=True)

# Training Loop
for itr in range(args.iterations):
    # For given number of iterations,
    print(f"Starting Iteration {itr}", Log.WRN)
    for client_id in range(1, args.world_size):
        # For all the HIEs connected to the Central server,
        print(f"Training client HIE{client_id}")
        # Run forward Pass for given client ID
        CentralServer.train_request(client_id)

    # Run Evaluation on the trained model.
    CentralServer.eval_request(itr)

# Send shutdown signal to all HIEs.
rpc.shutdown()
