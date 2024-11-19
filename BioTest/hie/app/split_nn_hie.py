from debugPrint import debug as print
from debugPrint import Log
import os

# import argparse
import torch.distributed.rpc as rpc
import configLoader

# Function for initializing environmental vars


def init_env(rank: int, addr: str, port: int):
    print(f"Initialize Meetup Spot, in dev rank {rank}", Log.INF)
    os.environ["MASTER_ADDR"] = addr
    os.environ["MASTER_PORT"] = str(port)


# Argument parser system
# parser = argparse.ArgumentParser(
#     description='Split Learning Initialization')
# parser.add_argument('--world_size', type=int, default=3,
#                     help='The world size which is equal to 1 server + (world size - 1) clients')
# parser.add_argument("--rank", type=int)
# parser.add_argument("--address", type=str, default="localhost",
#                     help="IP Address to listen on")
# parser.add_argument("--port", type=int, default="5656",
#                     help="Port to listen on")
# args = parser.parse_args()
args = configLoader.HIEConfig("conf.py")
# TODO To implement: get the data location, if not default, move/symlink the data location
# TODO to default location.
# Initializing env vars
init_env(args.rank, args.address, args.port)

# Initializing RPC Connection
rpc.init_rpc(f"hie{args.rank}", rank=args.rank, world_size=args.world_size)
print("Connected to Central", Log.SUC)

# Wait till a shutdown signal is received from central
rpc.shutdown()
