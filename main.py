from utils.utils import setupAPI, TensorBoard, Log
import torch.multiprocessing as mp
from torch.cuda import device_count
import argparse
from DeepLearning_API import PREDICTIONS_DIRECTORY, STATISTICS_DIRECTORY, DL_API_STATE
import os

def main():
    parser = argparse.ArgumentParser(description="DeepLearing API", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    with setupAPI(parser) as distributedObject:
        with Log(distributedObject.name) as log:
            world_size = device_count()
            if world_size == 0:
                world_size = 1
            distributedObject.setup(world_size)
            with TensorBoard(distributedObject.name, PREDICTIONS_DIRECTORY() if DL_API_STATE() == "PREDICTION" else STATISTICS_DIRECTORY()) as _:
                mp.spawn(distributedObject, nprocs=world_size)

if __name__ == "__main__":
    main()
 