from functools import partial, wraps
import importlib
import inspect
import os
from posixpath import split
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple, Union
from typing_extensions import Self
import torch
from abc import ABC
import numpy as np
from torch._jit_internal import _copy_to_script_wrapper

from DeepLearning_API.config import config
from DeepLearning_API.utils import NeedDevice, State, _getModule
from DeepLearning_API.HDF5 import ModelPatch
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

class CriterionsAttr():

    @config()
    def __init__(self, l: float = 1.0, isLoss: bool = True, stepStart:int = 0, stepStop:Optional[int] = None) -> None:
        self.l = l
        self.isTorchCriterion = True
        self.isLoss = isLoss
        self.stepStart = stepStart
        self.stepStop = stepStop
        
class CriterionsLoader():

    @config()
    def __init__(self, criterionsLoader: Dict[str, CriterionsAttr] = {"default:torch_nn_CrossEntropyLoss:Dice:NCC": CriterionsAttr()}) -> None:
        self.criterionsLoader = criterionsLoader

    def getCriterions(self, model_classname : str, output_group : str, target_group : str, train : bool) -> Dict[torch.nn.Module, CriterionsAttr]:
        criterions = {}
        for module_classpath, criterionsAttr in self.criterionsLoader.items():
            module, name = _getModule(module_classpath, "measure")
            criterionsAttr.isTorchCriterion = not module.startswith("DeepLearning_API.measure")
            criterions[config("{}.Model.{}.outputsCriterions.{}.targetsCriterions.{}.criterionsLoader.{}".format("Trainer" if train else "Predictor", model_classname, output_group, target_group, module_classpath))(getattr(importlib.import_module(module), name))(config = None)] = criterionsAttr
        return criterions

class TargetCriterionsLoader():

    @config()
    def __init__(self, targetsCriterions : Dict[str, CriterionsLoader] = {"default" : CriterionsLoader()}) -> None:
        self.targetsCriterions = targetsCriterions
        
    def getTargetsCriterions(self, output_group : str, model_classname : str, train : bool) -> Dict[str, Dict[torch.nn.Module, float]]:
        targetsCriterions = {}
        for target_group, criterionsLoader in self.targetsCriterions.items():
            targetsCriterions[target_group] = criterionsLoader.getCriterions(model_classname, output_group, target_group, train)
        return targetsCriterions

class Measure(NeedDevice):

    def __init__(self, model_classname : str, outputsCriterions: Dict[str, TargetCriterionsLoader], train : bool):
        super().__init__()
        self.outputsCriterions = {}
        for output_group, targetCriterionsLoader in outputsCriterions.items():
            self.outputsCriterions[output_group.replace(":", ".")] = targetCriterionsLoader.getTargetsCriterions(output_group, model_classname, train)
        self.values : Dict[str, List[float]] = dict()
        self.loss :Optional[torch.Tensor] = None
        self._it = 0
        self._nb_loss = 0

    def setDevice(self, device: torch.device):
        super().setDevice(device)
        for output_group in self.outputsCriterions:
            for target_group in self.outputsCriterions[output_group]:
                for criterion in self.outputsCriterions[output_group][target_group]:
                    if isinstance(criterion, NeedDevice):
                        criterion.setDevice(device)
        
    def init(self, model : torch.nn.Module):
        outputs_group_rename = {}
        for output_group in self.outputsCriterions.keys():
            for target_group in self.outputsCriterions[output_group]:
                for criterion in self.outputsCriterions[output_group][target_group]:
                    if not self.outputsCriterions[output_group][target_group][criterion].isTorchCriterion:
                        outputs_group_rename[output_group] = criterion.init(model, output_group, target_group)

        outputsCriterions_bak = self.outputsCriterions.copy()
        for old, new in outputs_group_rename.items():
            self.outputsCriterions.pop(old)
            self.outputsCriterions[new] = outputsCriterions_bak[old]
        for output_group in self.outputsCriterions:
            for target_group in self.outputsCriterions[output_group]:
                for criterion, criterionsAttr in self.outputsCriterions[output_group][target_group].items():
                    self.values["{}:{}:{}".format(output_group, target_group, criterion.__class__.__name__)] = []
                    if criterionsAttr.isLoss:
                        self._nb_loss+=1

    def update(self, output_group, output : torch.Tensor, data_dict: Dict[str, torch.Tensor], it: int):
        for target_group in self.outputsCriterions[output_group]:
            target = data_dict[target_group].to(self.device, non_blocking=False) if target_group in data_dict else None
            for criterion, criterionsAttr in self.outputsCriterions[output_group][target_group].items():
                if it >= criterionsAttr.stepStart and (criterionsAttr.stepStop is None or it <= criterionsAttr.stepStop):
                    result = criterion(output, target)
                    if criterionsAttr.isLoss:
                        self.loss += criterionsAttr.l*result
                        self._it +=1 
                    self.values["{}:{}:{}".format(output_group, target_group, criterion.__class__.__name__)].append(criterionsAttr.l*result.item())
                    
    def isDone(self):
        return self._it == self._nb_loss

    def resetLoss(self):
        self._it = 0
        self.loss = torch.zeros((1), requires_grad = True).to(self.device, non_blocking=False)

    def getLastValue(self):
        return self.loss.item() if self.loss is not None else 0
    
    def getLastMetrics(self):
        return {name : value[-1] if len(value) else 0 for name, value in self.values.items()}

    def format(self, isLoss) -> Dict[str, float]:
        result = dict()
        for name in self.values:
            output_group, target_group = name.split(":")[:2]
            for _, criterionsAttr in self.outputsCriterions[output_group][target_group].items():
                if criterionsAttr.isLoss == isLoss:
                    result[name] = np.mean(self.values[name]) if len(self.values[name]) else 0
        return result

    def mean(self) -> float:
        value = 0
        for name in self.values:
            value += np.mean(self.values[name])
        return value
    
    def clear(self) -> None:
        for name in self.values:
            self.values[name].clear()


class ModuleArgsDict(torch.nn.Module, ABC):
   
    class ModuleArgs:

        def __init__(self, in_branch: List[str], out_branch: List[str], pretrained : bool, alias : List[str], requires_grad: Optional[bool]) -> None:
            super().__init__()
            self.alias : List[str] = alias
            self.pretrained = pretrained
            self.in_branch = in_branch
            self.out_branch = out_branch
            self.in_channels = None
            self.in_is_channel = True
            self.out_channels = None
            self.out_is_channel = True
            self.requires_grad = requires_grad

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
            desc += ", in_channels={}".format(self._modulesArgs[key].in_channels)
            desc += ", in_is_channel={}".format(self._modulesArgs[key].in_is_channel)
            desc += ", out_channels={}".format(self._modulesArgs[key].out_channels)
            desc += ", out_is_channel={}".format(self._modulesArgs[key].out_is_channel)
            
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

    def add_module(self, name: str, module : torch.nn.Module, in_branch: List[Union[int, str]] = [0], out_branch: List[Union[int, str]] = [0], pretrained : bool = True, alias : List[str] = [], requires_grad:Optional[bool] = None) -> None:
        super().add_module(name, module)
        self._modulesArgs[name] = ModuleArgsDict.ModuleArgs([str(value) for value in in_branch], [str(value) for value in out_branch], pretrained, alias, requires_grad)
    
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
        branchs: Dict[str, torch.Tensor] = {}
        for i, input in enumerate(inputs):
            branchs[str(i)] = input
        out = inputs[0]
        for name, module in self.items():
            requires_grad = self._modulesArgs[name].requires_grad
            if requires_grad is not None and module:
                module.requires_grad_(requires_grad)
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

    def _apply_network(self, networks: List[torch.nn.Module], key: str, function: Callable, *args, **kwargs) -> Dict[str, object]:
        results : Dict[str, object] = {}
        for module in self.values():
            if isinstance(module, Network):
                if module not in networks:
                    networks.append(module)
                    for k, v in module._apply_network(networks, key+"."+module.getName(), function, *args, **kwargs).items():
                        results.update({self.getName()+"."+k : v})

        if len([param.name for param in list(inspect.signature(function).parameters.values()) if param.name == "key"]):
            function = partial(function, key=key)

        results[self.getName()] = function(self, *args, **kwargs)
        return results

    def _function_network(function : Callable): # type: ignore
        def new_function(self : Self, *args, **kwargs) -> Dict[str, object]:
            return self._apply_network([], self.getName(), function, *args, **kwargs)
        return new_function

    def __init__(   self,
                    in_channels : int = 1,
                    optimizer: Optional[OptimizerLoader] = None, 
                    schedulers: Optional[SchedulersLoader] = None, 
                    outputsCriterions: Optional[Dict[str, TargetCriterionsLoader]] = None,
                    patch : Optional[ModelPatch] = None,
                    nb_batch_per_step : int = 1,
                    init_type : str = "normal",
                    init_gain : float = 0.02,
                    dim : int = 3) -> None:
        super().__init__()
        self.in_channels = in_channels
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
        self.dim = dim
        
        self.scaler : Optional[torch.cuda.amp.grad_scaler.GradScaler] = None
        self._it = 0
        self.state_dict()
    
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
                else:
                    model_state_dict[alias] = model_state_dict_tmp[alias]
            self.load_state_dict(model_state_dict, strict=True)
        if "{}_optimizer_state_dict".format(name) in state_dict and self.optimizer:
            self.optimizer.load_state_dict(state_dict['{}_optimizer_state_dict'.format(name)])

    def _compute_channels_trace(self, module : ModuleArgsDict, in_channels : int, in_is_channel = True, out_channels : Optional[int] = None, out_is_channel = True) -> Tuple[int, bool, int, bool]:
        for k, v in module.items():
            if hasattr(v, "in_channels"):
                if v.in_channels:
                    in_channels = v.in_channels
            if hasattr(v, "in_features"):
                if v.in_features:
                    in_channels = v.in_features

            module._modulesArgs[k].in_channels = in_channels
            module._modulesArgs[k].in_is_channel = in_is_channel
            
            if isinstance(v, ModuleArgsDict):
                in_channels, in_is_channel, out_channels, out_is_channel = self._compute_channels_trace(v, in_channels, in_is_channel, out_channels, out_is_channel)

            if v.__class__.__name__ == "ToChannels":
                out_is_channel = True
            
            if v.__class__.__name__ == "ToFeatures":
                out_is_channel = False

            if hasattr(v, "out_channels"):
                if v.out_channels:
                    out_channels = v.out_channels
            if hasattr(v, "out_features"):
                if v.out_features:
                    out_channels = v.out_features

            module._modulesArgs[k].out_channels = out_channels
            module._modulesArgs[k].out_is_channel = out_is_channel

            in_channels = out_channels
            in_is_channel = out_is_channel
            
        return in_channels, in_is_channel, out_channels, out_is_channel

    @_function_network
    def init(self, autocast : bool, state : State, key: str) -> None:
        if state != State.PREDICTION:
            self.scaler = torch.cuda.amp.grad_scaler.GradScaler(enabled=autocast)
            if self.optimizerLoader:
                self.optimizer = self.optimizerLoader.getOptimizer(key, self.parameters(state == State.TRANSFER_LEARNING))
                self.optimizer.zero_grad()
            if self.schedulersLoader and self.optimizer:
                self.schedulers = self.schedulersLoader.getShedulers(key, self.optimizer)
        if self.outputsCriterionsLoader:
            self.measure = Measure(key, self.outputsCriterionsLoader, state != State.PREDICTION)
            if self.device:
                self.measure.setDevice(self.device)
                self.measure.init(self)
        self._compute_channels_trace(self, self.in_channels)
    
    
    def named_forward(self, *inputs: torch.Tensor) -> Iterator[Tuple[str, torch.Tensor]]:
        if self.patch:
            
            self.patch.load(inputs[0].shape[2:])
            patchIterator = self.patch.disassemble(*inputs)
            for patch_input in patchIterator:
                for (name, output_layer) in super().named_forward(*patch_input):
                    yield "{}{}".format("accu:", name), output_layer
                self.patch.addLayer(output_layer)

            yield name, self.patch.assemble(self.device)
        else:
            for (name, output_layer) in super().named_forward(*inputs):
                yield name, output_layer
    
    def get_layers(self, data_dict : Dict[Tuple[str, bool], torch.Tensor], layers_name: List[str]) -> Iterator[Tuple[str, torch.Tensor]]:
        inputs = [v for k, v in data_dict.items() if k[1]]

        output_layer_accumulator = {outputName : list() for outputName in layers_name}

        for (nameTmp, output_layer) in self.named_forward(*inputs):
            name = nameTmp.replace("accu:", "")
            if name in layers_name:
                accu = "accu:" in nameTmp
                if accu:
                    output_layer_accumulator[name].append(output_layer) 
                    networkName = nameTmp.split("accu:")[-2].split(".")[-1]
                    module = self
                    network = None
                    for n in name.split("."):
                        module = module[n]
                        if isinstance(module, Network) and n == networkName:
                            network = module
                            break

                    if network and network.patch and output_layer_accumulator[name] == len(network.patch):
                        output_layer = network.patch.assemble_accumulator(output_layer_accumulator[name], self.device)
                        output_layer_accumulator[name].clear()

                layers_name.remove(name)
                yield name, output_layer

            if not len(layers_name):
                break
            
    def _layer_function(self, name: str, output_layer: torch.Tensor, data_dict: Dict[Tuple[str, bool], torch.Tensor]):
        self.measure.update(name, output_layer, {k[0] : v for k, v in data_dict.items()}, self._it)
        self._backward()

    def forward(self, data_dict: Dict[Tuple[str, bool], torch.Tensor], output_layers: List[str] = []) -> List[Tuple[str, torch.Tensor]]:
        metric_tmp = {network : network.measure.outputsCriterions.keys() for network in self.getNetworks().values() if network.measure}

        outputsGroup : Dict[str, Self]= {}
        for k, v in metric_tmp.items():
            for a in v:
                outputsGroup[a] = k

        if not len(outputsGroup):
            return None
        
        self.resetLoss()
        results = []
        for name, layer in self.get_layers(data_dict, list(set(list(outputsGroup.keys())+output_layers))):
            if name in outputsGroup:
                outputsGroup[name]._layer_function(name, layer, data_dict)
            if name in output_layers:
                results.append((name, layer))
        return results 

    @_function_network
    def resetLoss(self):
        if self.measure:
            self.measure.resetLoss()

    def _backward(self):
        if self.measure:
            if self.measure.isDone() and self.measure.loss is not None:
                if self.scaler and self.optimizer:
                    self.measure.loss /= self.nb_batch_per_step
                    if self._it % self.nb_batch_per_step == 0:
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

    def getName(self):
        return self.__class__.__name__

class ModelLoader():

    @config("Model")
    def __init__(self, classpath : str = "default:segmentation.UNet") -> None:
        self.module, self.name = _getModule(classpath.split(".")[-1] if len(classpath.split(".")) > 1 else classpath, "networks" + "."+".".join(classpath.split(".")[:-1]) if len(classpath.split(".")) > 1 else "")
        
    def getModel(self, train : bool = True, DL_args: Optional[str] = None, DL_without=["optimizer", "schedulers", "nb_batch_per_step", "init_type", "init_gain"]) -> Network:
        if not DL_args:
            DL_args="{}.Model".format(os.environ["DEEP_LEARNING_API_ROOT"])
        model = partial(getattr(importlib.import_module(self.module), self.name), config = None, DL_args=DL_args)
        if not train: 
            model = partial(model, DL_without = DL_without)
        return model()