from subprocess import call
from typing import Callable
import datetime
import os
import signal
import sys
import traceback
import argparse
import os

from DeepLearning_API.utils import State
from DeepLearning_API import cluster

class ClusterSubmit(object):

    def __init__(
            self,
            fnc_kwargs: str,
            time_limit: str = "02:00:00",
            num_gpus: int = 0,
            num_nodes: int = 1,
            num_workers: int = 1,
            memory: str = "16GB",
            resubmit: bool = True,
            email: str = None
    ):
        """ 
        Argument/s:
            fnc_kwargs - keyword arguments for the function.
            manager - 'slurm' (needs to be modified for other cluster managers, e.g., PBS).
            time_limit - time limit for job.
            num_gpus - number of gpus.
            num_nodes - number of nodes.
            num_workers - number of workers per GPU.
            memory - minimum memory amount.
            resubmit - automatically resubmit job just before timout.
        """
        self.fnc_kwargs = fnc_kwargs
        self.exp_dir = "./"
        self.log_err = True
        self.log_out = True
        self.time_limit = time_limit
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.num_workers = num_workers
        self.memory = memory
        self.resubmit = resubmit

        self.email = email
        self.notify_on_end = True
        self.notify_on_fail = True

        self.script_name = "run.sh"

    def submit(self, job_display_name=None):
        self.job_display_name = job_display_name
        self.log_dir()
        scripts_path = os.path.join(self.exp_dir, 'manager_scripts')
        self.schedule_experiment(self.get_max_session_number(scripts_path))

    def schedule_experiment(self, session):
        timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        timestamp = f'session_{session}_{timestamp}'

        # Generate and save cluster manager script
        manager_cmd_script_path = os.path.join(self.manager_files_log_path, f'{timestamp}.sh')
        manager_cmd = self.build_slurm_command(manager_cmd_script_path, timestamp, session)

        self.save_manager_cmd(manager_cmd, manager_cmd_script_path)

        # Run script to launch job
        print('\nLaunching experiment...')
        result = call(f'{"sbatch"} {manager_cmd_script_path}', shell=True)
        if result == 0:
            print(f'Launched experiment {manager_cmd_script_path}.')
        else:
            print('Launch failed...')

    def save_manager_cmd(self, manager_cmd, manager_cmd_script_path):
        with open(manager_cmd_script_path, mode='w') as file:
            file.write(manager_cmd)

    def get_max_session_number(self, path):
        files = os.listdir(path)
        session_files = [f for f in files if 'session_' in f]
        if len(session_files) > 0:
            sessions = [int(f_name.split('_')[1]) for f_name in session_files]
            max_session = max(sessions)
            return max_session + 1
        else:
            return 0

    def log_dir(self):
        out_path = os.path.join(self.exp_dir)
        self.exp_dir = out_path
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        self.manager_files_log_path = os.path.join(out_path, 'manager_scripts')
        if not os.path.exists(self.manager_files_log_path):
            os.makedirs(self.manager_files_log_path)
        if self.log_err:
            err_path = os.path.join(out_path, 'error_logs')
            if not os.path.exists(err_path):
                os.makedirs(err_path)
            self.err_log_path = err_path
        if self.log_out:
            out_path = os.path.join(out_path, 'out_logs')
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            self.out_log_path = out_path

    def build_slurm_command(self, manager_cmd_script_path, timestamp, session):

        sub_commands = ['#!/bin/bash -l']
        sub_commands.append(f'#SBATCH --job-name={self.job_display_name}session_{session}')

        if self.log_out:
            out_path = os.path.join(self.out_log_path, f'{timestamp}_%j.out')
            sub_commands.append(f'#SBATCH --output={out_path}')

        if self.log_err:
            err_path = os.path.join(self.err_log_path, f'{timestamp}_%j.err')
            sub_commands.append(f'#SBATCH --error={err_path}')
        sub_commands.append(f'#SBATCH --time={self.time_limit:s}')

        if self.num_gpus:
            sub_commands.append(f'#SBATCH --gres=gpu:{self.num_gpus}')

        if self.num_workers > 0:
            sub_commands.append(f'#SBATCH --cpus-per-task={self.num_workers}')

        sub_commands.append(f'#SBATCH --nodes={self.num_nodes}')
        sub_commands.append(f'#SBATCH --mem={self.memory}')
        sub_commands.append('#SBATCH --signal=B:USR1@360')
        sub_commands.append('#SBATCH --open-mode=append')
        sub_commands.append("test()\n{\n echo \"Requeing job $SLURM_JOB_ID\" \n export DEEP_LEARNING_API_RESUBMIT=\"TRUE\" \n scontrol requeue $SLURM_JOB_ID \n}")
        sub_commands.append("trap 'test' USR1")

        mail_type = []
        if self.notify_on_end:
            mail_type.append('END')
        if self.notify_on_fail:
            mail_type.append('FAIL')
        if len(mail_type) > 0 and self.email is not None:
            sub_commands.append(f'#SBATCH --mail-type={",".join(mail_type)}')
            sub_commands.append(f'#SBATCH --mail-user={self.email}')

        sub_commands = [x.lstrip() for x in sub_commands]

        cmd = f'{self.script_name} {self.fnc_kwargs}'
        sub_commands.append(cmd)
        return '\n'.join(sub_commands)



def main():
    parser = argparse.ArgumentParser(description="DeepLearing API", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    api_args = parser.add_argument_group('API arguments')
    
    api_args.add_argument("type", type=State, choices=list(State))
    api_args.add_argument('-y', action='store_true', help="Accept overwrite")
    api_args.add_argument("-c", "--config", type=str, default="None", help="Configuration file location")
    api_args.add_argument("-models_dir", "--MODELS_DIRECTORY", type=str, default="./Models/", help="Models location")
    api_args.add_argument("-checkpoints_dir", "--CHECKPOINTS_DIRECTORY", type=str, default="./Checkpoints/", help="Checkpoints location")
    api_args.add_argument("-url", "--URL_MODEL", type=str, default=None, help="URL Model")
    api_args.add_argument("-predictions_dir", "--PREDICTIONS_DIRECTORY", type=str, default="./Predictions/", help="Predictions location")
    api_args.add_argument("-statistics_dir", "--STATISTICS_DIRECTORY", type=str, default="./Statistics/", help="Statistics location")
    api_args.add_argument("-setups_dir", "--SETUPS_DIRECTORY", type=str, default="./Setups/", help="Setups location")
    
    # Distributed computing arguments
    distributed_args = parser.add_argument_group('Distributed computing arguments')
    distributed_args.add_argument('--num-workers', '--num_workers', default=0, type=int, help='No. of workers per DataLoader & GPU')
    distributed_args.add_argument('--devices', default=1, type=int, help='Number of devices per node')
    distributed_args.add_argument('--num-nodes', '--num_nodes', default=1, type=int, help='Number of nodes')

    # Cluster manager arguments
    cluster_args = parser.add_argument_group('Cluster manager arguments')
    
    cluster_args.add_argument('--name', type=str, help='Task name', required=True)
    cluster_args.add_argument('--memory', type=str, default="16Gb", help='Amount of memory per node')
    cluster_args.add_argument('--time-limit', '--time_limit', type=str, default="01:00:00", help='Job time limit')
    cluster_args.add_argument('--resubmit', action='store_true', help='Automatically resubmit job just before timout')
    cluster_args.add_argument('--email', type=str, help='Email for cluster manager notifications')


    args = parser.parse_args()
    config = vars(args)

    clusterSubmit = cluster.ClusterSubmit("{} {}--config {} -models_dir {} -checkpoints_dir {} -url {} -predictions_dir {} -statistics_dir {} -setups_dir {} {}".format(config["type"], "-y " if config["y"] else "", config["config"], config["MODELS_DIRECTORY"], config["CHECKPOINTS_DIRECTORY"], config["URL_MODEL"], config["PREDICTIONS_DIRECTORY"], config["STATISTICS_DIRECTORY"], config["SETUPS_DIRECTORY"], "--resubmit" if config["resubmit"] else ""), time_limit=config["time_limit"], num_gpus=config["devices"], num_workers=config["num_workers"], memory=config["memory"], resubmit=config["resubmit"], email=config["email"])
    clusterSubmit.submit(job_display_name=config["name"] + '_')

if __name__ == "__main__":
    main()
 