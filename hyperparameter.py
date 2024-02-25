
from utils.utils import DistributedObject, State, getMaxGPUMemory
from DeepLearning_API.config import config
from DeepLearning_API.networks.network import ModelLoader, Network, CPU_Model, ModuleArgsDict
from DeepLearning_API.dataset import DataHyperparameter
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
import torch
import os
import copy

class Hyperparameter(DistributedObject):

    @config("Hyperparameter")
    def __init__(self, model: ModelLoader = ModelLoader(), dataset: DataHyperparameter = DataHyperparameter()):
        super().__init__("")
        self.dataset = dataset
        self.model = model.getModel(train=True)
        self.gpu_alloc = self._getNbCopy(self.model)
        self.nameNetworks = {}
        for k, v in self.gpu_alloc.items():
            for l in v:
                self.nameNetworks[l] = k
        os.environ["DL_API_DEBUG"] = "True"


    def _getNbCopy(self, network: Network):
        result: dict[str, list[str]] = {}
        for name, module in network.items():
            if isinstance(module, Network):
                if module._get_name() in result:
                    result[module._get_name()].append(name)
                else:
                    result[module._get_name()] = [name]
                for k,v in self._getNbCopy(module):
                    if module._get_name() in result:
                        result[k].append(v)
                    else:
                        result[k] = [v]
        return result
    
    def setup(self, world_size: int):
        self.model.init(False, State.TRAIN)
        self.model.init_outputsGroup()
        self.model.load({}, init=True, ema=False)
        self.world_size = world_size
        self.dataloader = self.dataset.getData(1)

    def getInput(self, data_dict : dict[str, tuple[torch.Tensor, int, int, int, str, bool]]) -> dict[tuple[str, bool], torch.Tensor]:
        return {(k, v[5][0].item()) : v[0] for k, v in data_dict.items()}

    def isModuleArgsDict(self, name):
        module = self.model
        for n in name.split("."):
            if n not in module.keys():
                return False
            if not isinstance(module[n], ModuleArgsDict):
                return False
            module = module[n]
        return True
    
    def _optim_gradient_checkpoints(self, input: dict[tuple[str, bool], torch.Tensor], batchsize: int) -> list[str]:
        input = {k: v.repeat([batchsize]+[1 for _ in range(len(v.shape)-1)]) for k,v in input.items()}
        gradient_checkpoints = []
        while True:
            if "device" in os.environ:
                del os.environ["device"]
            if "DL_API_DEBUG_LAST_LAYER" in os.environ:
                del os.environ["DL_API_DEBUG_LAST_LAYER"]
            model = copy.deepcopy(self.model)
            model._compute_channels_trace(model, model.in_channels, None, gradient_checkpoints)
            
            torch.cuda.empty_cache()
            model = Network.to(model, 0)
            model = DDP(model) if torch.cuda.is_available() else CPU_Model(model)
            try:
                model(input)
                model.module.backward(model.module)
                break
            except KeyboardInterrupt:
                exit(0)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    nbGPU = len(gradient_checkpoints)
                    if nbGPU == torch.cuda.device_count()-1:
                        raise NameError("Model too large")
                    name_tmp = {}
                    data = [(v.split(":")[0],  v.split(":")[0].split(".")[0], float(v.split(":")[1]), int(v.split(":")[2])) for v in os.environ["DL_API_DEBUG_LAST_LAYER"].split("|")]
                    for i, (name, preName, GPUMemory, gpu) in enumerate(data):
                        if gpu == nbGPU:
                            if preName in name_tmp:
                                if GPUMemory-name_tmp[preName] > (getMaxGPUMemory(gpu)-GPUMemory)/len(self.gpu_alloc[self.nameNetworks[name.split(".")[0]]]):
                                    for it in range(i-1):
                                        if self.isModuleArgsDict(data[i-it-1][0]):
                                            gradient_checkpoints.append(data[i-it-1][0])
                                            print(gradient_checkpoints)
                                            exit(0)
                                            break
                                    if it == 0:
                                        raise NameError("No ModuleArgsDict")
                                    break

                        if preName in self.nameNetworks and preName not in name_tmp:
                            name_tmp[preName] = GPUMemory
                        print(name, GPUMemory)
                    if gpu == len(gradient_checkpoints):
                        for it in range(i-1):
                            if self.isModuleArgsDict(data[i-it-1][0]):
                                gradient_checkpoints.append(data[i-it-1][0])
                                break
                        if it == 0:
                            raise NameError("No ModuleArgsDict")
                else:
                    print(e)
                    exit(0)
        return gradient_checkpoints
    
    def run_process(self, world_size: int, global_rank: int, local_rank: int, dataloaders: list[DataLoader]):
        gradient_checkpoints = self._optim_gradient_checkpoints(self.getInput(next(iter(dataloaders[0]))), 1)
        print("OKKKK")
        print(gradient_checkpoints)
        exit(0)
        batchSize = 2
        if len(gradient_checkpoints) > torch.cuda.device_count()//2-1:
            while True:
                try:
                    gradient_checkpoints = self._optim_gradient_checkpoints(self.getInput(next(iter(dataloaders[0]))), batchSize)
                    batchSize+=1
                    print(batchSize)
                except KeyboardInterrupt:
                    exit(0)
                except:
                    batchSize-=1
                    break
        else:
            while True:
                try:
                    gradient_checkpoints_tmp = self._optim_gradient_checkpoints(self.getInput(next(iter(dataloaders[0]))), batchSize)
                    if len(gradient_checkpoints_tmp) == len(gradient_checkpoints):
                        gradient_checkpoints = gradient_checkpoints_tmp
                    else:
                        break
                    batchSize+=1
                    print(batchSize)
                except KeyboardInterrupt:
                    exit(0)
                except:
                    batchSize-=1
                    break

        print(gradient_checkpoints)
        print(batchSize)
        
