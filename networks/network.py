from functools import partial, wraps
import functools
import importlib
from operator import mod
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple
import typing
from typing_extensions import Self
import torch
from abc import ABC
import numpy as np
import torch.nn.functional as F
from torch._jit_internal import _copy_to_script_wrapper

from DeepLearning_API.config import config
from DeepLearning_API.measure import Measure, TargetCriterionsLoader
from DeepLearning_API.utils import NeedDevice, State, _getModule
from DeepLearning_API.dataset import Patch
from collections import OrderedDict

class OptimizerLoader():
    
    @config("Optimizer")
    def __init__(self, name: str = "AdamW") -> None:
        self.name = name
    
    def getOptimizer(self, key: str, parameter: Iterator[torch.nn.parameter.Parameter]) -> torch.optim.Optimizer:
        return config("Trainer.Model.{}.Optimizer".format(key))(getattr(importlib.import_module('torch.optim'), self.name))(parameter, config = None)
        
class SchedulerStep():
    
    @config(None)
    def __init__(self, nb_step : int = 0) -> None:
        self.nb_step = nb_step

class SchedulersLoader():
        
    @config("Schedulers")
    def __init__(self, params: Dict[str, SchedulerStep] = {"default:ReduceLROnPlateau" : SchedulerStep(0)}) -> None:
        self.params = params

    def getShedulers(self, key: str, optimizer: torch.optim.Optimizer) -> Dict[torch.optim.lr_scheduler._LRScheduler, int]:
        shedulers : Dict[torch.optim.lr_scheduler._LRScheduler, int] = {}
        for name, step in self.params.items():
            if name:
                shedulers[config("Trainer.Model.{}.Schedulers.{}".format(key, name))(getattr(importlib.import_module('torch.optim.lr_scheduler'), name))(optimizer, config = None)] = step.nb_step
        return shedulers

class ModuleArgsDict(torch.nn.Module, ABC):
   
    class ModuleArgs:

        def __init__(self, in_branch: List[int], out_branch: List[int], pretrained : bool, alias : List[str]) -> None:
            super().__init__()
            self.alias : List[str] = alias
            self.pretrained = pretrained
            self.in_branch = in_branch
            self.out_branch = out_branch

    def __init__(self) -> None:
       super().__init__()
       self._modulesArgs : dict[str, ModuleArgsDict.ModuleArgs] = dict()
    
    def _addindent(self, s_: str, numSpaces : int):
        s = s_.split('\n')
        if len(s) == 1:
            return s_
        first = s.pop(0)
        s = [(numSpaces * ' ') + line for line in s]
        s = '\n'.join(s)
        s = first + '\n' + s
        return s

    def __repr__(self):
        extra_lines = []

        extra_repr = self.extra_repr()
        if extra_repr:
            extra_lines = extra_repr.split('\n')

        child_lines = []
        is_simple_branch = lambda x : len(x) > 1 or x[0] != 0 
        for key, module in self._modules.items():
            mod_str = repr(module)
            mod_str = self._addindent(mod_str, 2)
            desc = ""
            if is_simple_branch(self._modulesArgs[key].in_branch) or is_simple_branch(self._modulesArgs[key].out_branch):
                desc += ", {}->{}".format(self._modulesArgs[key].in_branch, self._modulesArgs[key].out_branch)
            if not self._modulesArgs[key].pretrained:
                desc += ", pretrained=False"
            if self._modulesArgs[key].alias:
                desc += ", alias={}".format(self._modulesArgs[key].alias)
            child_lines.append("({}{}) {}".format(key, desc, mod_str))
            
        lines = extra_lines + child_lines

        desc = ""
        if lines:
            if len(extra_lines) == 1 and not child_lines:
                desc += extra_lines[0]
            else:
                desc += '\n  ' + '\n  '.join(lines) + '\n'

        return "{}({})".format(self._get_name(), desc)

    def __getitem__(self, key: str) -> torch.nn.Module:
        module = self._modules[key]
        assert module, "Error {} is None".format(key)
        return module 

    @_copy_to_script_wrapper
    def keys(self) -> Iterable[str]:
        return self._modules.keys()

    @_copy_to_script_wrapper
    def items(self) -> Iterable[Tuple[str, Optional[torch.nn.Module]]]:
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[Optional[torch.nn.Module]]:
        return self._modules.values()

    def add_module(self, name: str, module : torch.nn.Module, in_branch: List[int] = [0], out_branch: List[int] = [0], pretrained : bool = True, alias : List[str] = []) -> None:
        super().add_module(name, module)
        self._modulesArgs[name] = ModuleArgsDict.ModuleArgs(in_branch, out_branch, pretrained, alias)
    
    def getMap(self):
        results : Dict[str, str] = {}
        for name, moduleArgs in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                if len(moduleArgs.alias):
                    count = {k : 0 for k in set(module.getMap().values())}
                    for k, v in module.getMap().items():
                        alias_name = moduleArgs.alias[count[v]]
                        if k == "":
                            results.update({alias_name : name+"."+v})
                        else:
                            results.update({alias_name+"."+k : name+"."+v})
                        count[v]+=1
                else:
                    results.update({k : name+"."+v for k, v in module.getMap().items()})
            else:
                for alias in moduleArgs.alias:
                    results[alias] = name
        return results

    @staticmethod
    def init_func(module: torch.nn.Module, init_type: str, init_gain: float):
        if not isinstance(module, Network):
            if isinstance(module, torch.nn.modules.conv._ConvNd) or isinstance(module, torch.nn.Linear):
                if init_type == 'normal':
                    torch.nn.init.normal_(module.weight, 0.0, init_gain)
                elif init_type == 'xavier':
                    torch.nn.init.xavier_normal_(module.weight, gain=init_gain)
                elif init_type == 'kaiming':
                    torch.nn.init.kaiming_normal_(module.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    torch.nn.init.orthogonal_(module.weight, gain=init_gain)
                elif init_type == "trunc_normal":
                    torch.nn.init.trunc_normal_(module.weight, std=init_gain)
                else:
                    raise NotImplementedError('Initialization method {} is not implemented'.format(init_type))
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

            elif isinstance(module, torch.nn.modules.batchnorm._BatchNorm):
                torch.nn.init.normal_(module.weight, 0.0, std = init_gain)
                torch.nn.init.constant_(module.bias, 0.0)

    def named_forward(self, *inputs: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        branchs: Dict[int, torch.Tensor] = {}
        for i, input in enumerate(inputs):
            branchs[i] = input
        out = inputs[0]
        for name, module in self.items():
            for ib in self._modulesArgs[name].in_branch:
                if ib not in branchs:
                    branchs[ib] = inputs[0]
            if isinstance(module, ModuleArgsDict):
                for k, out in module.named_forward(*[branchs[i] for i in self._modulesArgs[name].in_branch]):
                    yield name+"."+k, out
            elif isinstance(module, torch.nn.Module):  
                out = module(*[branchs[i] for i in self._modulesArgs[name].in_branch])
                yield name, out
            for ob in self._modulesArgs[name].out_branch:
                branchs[ob] = out
        del branchs

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        v = input
        for _, v in self.named_forward(input):
            pass
        return v

    def named_parameters(self, pretrained: bool = False) -> Iterator[Tuple[str, torch.nn.parameter.Parameter]]:
        for name, moduleArgs in self._modulesArgs.items():
            if isinstance(self[name], Network):
                continue
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                for k, v in module.named_parameters(pretrained):
                    yield name+"."+k, v
            elif isinstance(module, torch.nn.Module):
                if not pretrained or not moduleArgs.pretrained:
                    for k, v in module.named_parameters():
                        yield name+"."+k, v

    def parameters(self, pretrained: bool = False):
        for _, v in self.named_parameters(pretrained):
            yield v
            
class Network(ModuleArgsDict, NeedDevice, ABC):

    def _apply_network(self, function: Callable) -> Dict[str, object]:
        results : Dict[str, object] = {}
        for module in self.values():
            if isinstance(module, Network):
                for k, v in module._apply_network(function).items():
                    results.update({self.getName()+"."+k : v})
        results[self.getName()] = function()
        return results

    def _function_network(function : Callable): # type: ignore
        @wraps(function)
        def new_function(self : Self, *args, **kwargs) -> Dict[str, object]:
            return self._apply_network(partial(function, self, *args,  **kwargs))
        return new_function

    def __init__(   self,
                    optimizer: Optional[OptimizerLoader] = None, 
                    schedulers: Optional[SchedulersLoader] = None, 
                    outputsCriterions: Optional[Dict[str, TargetCriterionsLoader]] = None,
                    patch : Optional[Patch] = None,
                    nb_batch_per_step : int = 1,
                    init_type : str = "normal",
                    init_gain : float = 0.02,
                    padding : int = 0,
                    paddingMode : str = "constant",
                    dim : int = 3) -> None:
        super().__init__()
        self.optimizerLoader  = optimizer
        self.optimizer : Optional[torch.optim.Optimizer] = None

        self.schedulersLoader  = schedulers
        self.schedulers : Optional[Dict[torch.optim.lr_scheduler._LRScheduler, int]] = None

        self.outputsCriterionsLoader = outputsCriterions
        self.measure : Optional[Measure] = None

        self.patch = patch

        self.nb_batch_per_step = nb_batch_per_step
        self.init_type  = init_type
        self.init_gain  = init_gain
        self.padding = padding
        self.paddingMode = paddingMode
        self.dim = dim
        
        self.scaler : Optional[torch.cuda.amp.grad_scaler.GradScaler] = None
        self._it = 0
        
    @_function_network
    def setDevice(self, device: torch.device):
        super().setDevice(device)
        
    @_function_network
    def load(self, state_dict : Dict[str, Dict[str, torch.Tensor]], init: bool = True, ema : bool =False):
        if init:
            self.apply(partial(ModuleArgsDict.init_func, init_type=self.init_type, init_gain=self.init_gain))
            
        name = self.getName() + ("_EMA" if ema else "")
        if name in state_dict:
            model_state_dict_tmp = state_dict[name]
            map = self.getMap()
            model_state_dict : OrderedDict[str, torch.Tensor] = OrderedDict()
            
            for alias in model_state_dict_tmp.keys():
                prefix = ".".join(alias.split(".")[:-1])
                if prefix in map.keys():
                    model_state_dict[alias.replace(prefix, map[prefix])] = model_state_dict_tmp[alias]
                if prefix in map.values():
                    model_state_dict[alias] = model_state_dict_tmp[alias]
            self.load_state_dict(model_state_dict, strict=True)
            if "{}_optimizer_state_dict".format(name) in state_dict and self.optimizer:
                self.optimizer.load_state_dict(state_dict['{}_optimizer_state_dict'.format(name)])

    @_function_network
    def init(self, autocast : bool, state : State) -> None:
        if state != State.PREDICTION:
            if self.optimizerLoader:
                self.optimizer = self.optimizerLoader.getOptimizer(self.getName(), self.parameters(state == State.TRANSFER_LEARNING))
                self.optimizer.zero_grad()
            if self.schedulersLoader and self.optimizer:
                self.schedulers = self.schedulersLoader.getShedulers(self.getName(), self.optimizer)
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=autocast)
        if self.outputsCriterionsLoader:
            self.measure = Measure(self.getName(), self.outputsCriterionsLoader, state != State.PREDICTION)
            if self.device:
                self.measure.setDevice(self.device)
    
    def getInput(self, data_dict : Dict[str, torch.Tensor]) -> torch.Tensor:
        input = torch.cat(list(data_dict.values()), dim=1)
        if self.padding > 0:
            input = F.pad(input, tuple([0, 0]*self.dim+[0,self.padding]), self.paddingMode)
        return input
        
    def forward(self, data_dict : Dict[Tuple[str, bool], torch.Tensor]) -> torch.Tensor:
        input = self.getInput({k[0] : torch.unsqueeze(v, dim=1) for k, v in data_dict.items() if k[1]})
        output_layer = input
        outputsGroup = list(self.measure.outputsCriterions.keys()).copy() if self.measure else list("None")
        if self.patch:
            self.patch.load(input.shape[2:])
            output_layer_accumulator = {outputGroup : list() for outputGroup in outputsGroup}

            for patch_input in self.patch.disassemble(input):
                outputsGroup_tmp = outputsGroup.copy()
                for (name, output_layer) in self.named_forward(patch_input):
                    if name in outputsGroup_tmp:
                        outputsGroup_tmp.remove(name)
                        output_layer_accumulator[name].append(output_layer)
                    if not len(outputsGroup_tmp):
                        break

            for outputGroup in outputsGroup:
                if self.measure: 
                    self.measure.update(outputGroup, self.patch.assemble(output_layer_accumulator[outputGroup]), {k[0] : v for k, v in data_dict.items()})
        else:
            for (name, output_layer) in self.named_forward(input):
                if name in outputsGroup:
                    outputsGroup.remove(name)
                    if self.measure:
                        self.measure.update(name, output_layer, {k[0] : v for k, v in data_dict.items()})
                if not len(outputsGroup):
                    break
        return output_layer
    
    @_function_network
    def backward(self):
        assert self.measure and self.scaler and self.optimizer, "Error init model with [optimizer, criterions]"
        
        self.measure.loss /= self.nb_batch_per_step
        if self._it % self.nb_batch_per_step == 0:
            if self.measure.loss is not None:
                self.scaler.scale(self.measure.loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        self._it += 1

    @_function_network
    def update_lr(self):
        step = 0
        scheduler = None
        if self.schedulers:
            for scheduler, value in self.schedulers.items():
                if value is None or (self._it >= step  and self._it < step+value):
                    break
                step += value
        if scheduler:
            if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                if self.measure:
                    scheduler.step(np.mean(self.measure.value))
            else:
                scheduler.step()     

    @_function_network
    def measureClear(self):
        if self.measure:
            self.measure.clear()
    
    @_function_network
    def getNetworks(self) -> Self:
        return self

    @_function_network
    def logImage(self, data_dict : Dict[str, torch.Tensor], output : Dict[str, torch.Tensor]) -> Optional[Dict[str, np.ndarray]]:
        return None

    def getName(self):
        return self.__class__.__name__

class ModelLoader():

    @config("Model")
    def __init__(self, classpath : str = "default:segmentation.UNet") -> None:
        self.module, self.name = _getModule(classpath.split(".")[-1] if len(classpath.split(".")) > 1 else classpath, "networks" + "."+".".join(classpath.split(".")[:-1]) if len(classpath.split(".")) > 1 else "")
        
    def getModel(self, train : bool = True) -> Network:
        model = partial(getattr(importlib.import_module(self.module), self.name), config = None, DL_args="{}.Model".format("Trainer" if train else "Predictor"))
        if not train: 
            model = partial(model, DL_without = ["optimizer", "scheduler", "nb_batch_per_step", "init_type", "init_gain"])
        return model()