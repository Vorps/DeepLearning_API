from DeepLearning_API.utils import setupAPI, TensorBoard, Log
import torch.multiprocessing as mp
from torch.cuda import device_count
import argparse
import sys

def main():
    parser = argparse.ArgumentParser(description="DeepLearing API", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    with setupAPI(parser) as distributedObject:
        with Log(distributedObject.name) as log:
            world_size = device_count()
            if world_size == 0:
                world_size = 1
            distributedObject.setup(world_size)
            with TensorBoard(distributedObject.name) as _:
                mp.spawn(distributedObject, nprocs=world_size)

if __name__ == "__main__":
    main()
 