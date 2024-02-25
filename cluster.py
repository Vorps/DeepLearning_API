import argparse
import submitit
import os 

from utils.utils import setupAPI, TensorBoard
from DeepLearning_API import STATISTICS_DIRECTORY, PREDICTIONS_DIRECTORY, DL_API_STATE

def main():
    parser = argparse.ArgumentParser(description="DeepLearing API", formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Cluster manager arguments
    cluster_args = parser.add_argument_group('Cluster manager arguments')
    cluster_args.add_argument('--name', type=str, help='Task name', required=True)
    cluster_args.add_argument('--num-nodes', '--num_nodes', default=1, type=int, help='Number of nodes')
    cluster_args.add_argument('--memory', type=int, default=16, help='Amount of memory per node')
    cluster_args.add_argument('--time-limit', '--time_limit', type=int, default=1440, help='Job time limit in minute')
    cluster_args.add_argument('--resubmit', action='store_true', help='Automatically resubmit job just before timout')

    with setupAPI(parser) as distributedObject:
        args = parser.parse_args()
        config = vars(args)
        os.environ["DL_API_OVERWRITE"] = "True"
        os.environ["DL_API_CLUSTER"] = "True"

        n_gpu = len(config["gpu"].split(","))
        distributedObject.setup(n_gpu*int(config["num_nodes"]))

        executor = submitit.AutoExecutor(folder="./Cluster/")
        executor.update_parameters(name=config["name"], mem_gb=config["memory"], gpus_per_node=n_gpu, tasks_per_node=n_gpu//distributedObject.size, cpus_per_task=config["num_workers"], nodes=config["num_nodes"], timeout_min=config["time_limit"])
        with TensorBoard(distributedObject.name, PREDICTIONS_DIRECTORY() if DL_API_STATE() == "PREDICTION" else STATISTICS_DIRECTORY()) as _:
            executor.submit(distributedObject)

if __name__ == "__main__":
    main()
 