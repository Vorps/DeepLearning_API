from functools import partial
import importlib
import inspect
import os
from typing import Iterable, Iterator, Callable
from typing_extensions import Self
import torch
from abc import ABC
import numpy as np
from torch._jit_internal import _copy_to_script_wrapper

from DeepLearning_API.config import config
from DeepLearning_API.utils import State, _getModule
from DeepLearning_API.HDF5 import Accumulator, ModelPatch
from collections import OrderedDict
from torch.utils.checkpoint import checkpoint
from typing import Union

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
    def __init__(self, params: dict[str, SchedulerStep] = {"default:ReduceLROnPlateau" : SchedulerStep(0)}) -> None:
        self.params = params

    def getShedulers(self, key: str, optimizer: torch.optim.Optimizer) -> dict[torch.optim.lr_scheduler._LRScheduler, int]:
        shedulers : dict[torch.optim.lr_scheduler._LRScheduler, int] = {}
        for name, step in self.params.items():
            if name:
                shedulers[config("Trainer.Model.{}.Schedulers.{}".format(key, name))(getattr(importlib.import_module('torch.optim.lr_scheduler'), name))(optimizer, config = None)] = step.nb_step
        return shedulers

class CriterionsAttr():

    @config()
    def __init__(self, l: float = 1.0, isLoss: bool = True, stepStart:int = 0, stepStop: Union[int, None] = None) -> None:
        self.l = l
        self.isTorchCriterion = True
        self.isLoss = isLoss
        self.stepStart = stepStart
        self.stepStop = stepStop
        
class CriterionsLoader():

    @config()
    def __init__(self, criterionsLoader: dict[str, CriterionsAttr] = {"default:torch_nn_CrossEntropyLoss:Dice:NCC": CriterionsAttr()}) -> None:
        self.criterionsLoader = criterionsLoader

    def getCriterions(self, model_classname : str, output_group : str, target_group : str, train : bool) -> dict[torch.nn.Module, CriterionsAttr]:
        criterions = {}
        for module_classpath, criterionsAttr in self.criterionsLoader.items():
            module, name = _getModule(module_classpath, "measure")
            criterionsAttr.isTorchCriterion = module.startswith("torch")
            criterions[config("{}.Model.{}.outputsCriterions.{}.targetsCriterions.{}.criterionsLoader.{}".format("Trainer" if train else "Predictor", model_classname, output_group, target_group, module_classpath))(getattr(importlib.import_module(module), name))(config = None)] = criterionsAttr
        return criterions

class TargetCriterionsLoader():

    @config()
    def __init__(self, targetsCriterions : dict[str, CriterionsLoader] = {"default" : CriterionsLoader()}) -> None:
        self.targetsCriterions = targetsCriterions
        
    def getTargetsCriterions(self, output_group : str, model_classname : str, train : bool) -> dict[str, dict[torch.nn.Module, float]]:
        targetsCriterions = {}
        for target_group, criterionsLoader in self.targetsCriterions.items():
            targetsCriterions[target_group] = criterionsLoader.getCriterions(model_classname, output_group, target_group, train)
        return targetsCriterions

class Measure():

    def __init__(self, model_classname : str, outputsCriterions: dict[str, TargetCriterionsLoader], train : bool) -> None:
        super().__init__()
        self.outputsCriterions = {}
        for output_group, targetCriterionsLoader in outputsCriterions.items():
            self.outputsCriterions[output_group.replace(":", ".")] = targetCriterionsLoader.getTargetsCriterions(output_group, model_classname, train)
        self.values : dict[str, list[float]] = dict()
        self.loss : Union[torch.Tensor, None] = None
        self._it = 0
        self._nb_loss = 0
        self.saved_loss = 0

    def init(self, model : torch.nn.Module) -> None:
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

    def update(self, output_group: str, output : torch.Tensor, data_dict: dict[str, torch.Tensor], it: int) -> None:
        for target_group in self.outputsCriterions[output_group]:
            target = [data_dict[group].to(output.device) for group in target_group.split("/") if group in data_dict]
            for criterion, criterionsAttr in self.outputsCriterions[output_group][target_group].items():
                
                if it >= criterionsAttr.stepStart and (criterionsAttr.stepStop is None or it <= criterionsAttr.stepStop):
                    result = criterion(output, *target)
                    if criterionsAttr.isLoss:
                        self.loss = self.loss.to(result.device)+criterionsAttr.l*result
                        self._it +=1
                    self.values["{}:{}:{}".format(output_group, target_group, criterion.__class__.__name__)].append(criterionsAttr.l*result.item())
                elif criterionsAttr.isLoss:
                    self._it +=1

    def isDone(self) -> bool:
        return self._it == self._nb_loss
    
    def resetLoss(self) -> None:
        self._it = 0
        self.saved_loss = self.getLastValue()
        self.loss = torch.zeros((1), requires_grad = True)
        
    def getLastValue(self) -> float:
        return self.loss.item() if self.loss is not None else 0
    
    def getLastMetrics(self) -> dict[str, float]:
        return {name : value[-1] if len(value) else 0 for name, value in self.values.items()}

    def format(self, isLoss) -> dict[str, float]:
        result = dict()
        for name in self.values:
            output_group, target_group = name.split(":")[:2]
            for _, criterionsAttr in self.outputsCriterions[output_group][target_group].items():
                if criterionsAttr.isLoss == isLoss:
                    result[name] = np.mean(self.values[name]) if len(self.values[name]) else 0
        return result

    def mean(self) -> float:
        value = 0.0
        for name in self.values:
            value += np.mean(self.values[name])
        return value
    
    def clear(self) -> None:
        for name in self.values:
            self.values[name].clear()


class ModuleArgsDict(torch.nn.Module, ABC):
   
    class ModuleArgs:

        def __init__(self, in_branch: list[str], out_branch: list[str], pretrained : bool, alias : list[str], requires_grad: Union[bool, None]) -> None:
            super().__init__()
            self.alias= alias
            self.pretrained = pretrained
            self.in_branch = in_branch
            self.out_branch = out_branch
            self.in_channels = None
            self.in_is_channel = True
            self.out_channels = None
            self.out_is_channel = True
            self.requires_grad = requires_grad
            self.isCheckpoint = False
            self.isGPU_Checkpoint = False
            self.gpu = "CPU"

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
            desc += ", isInCheckpoint={}".format(self._modulesArgs[key].isCheckpoint)
            desc += ", isInGPU_Checkpoint={}".format(self._modulesArgs[key].isGPU_Checkpoint)
            desc += ", requires_grad={}".format(self._modulesArgs[key].requires_grad)
            desc += ", device={}".format(self._modulesArgs[key].gpu)
            
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
    def items(self) -> Iterable[tuple[str, Union[torch.nn.Module, None]]]:
        return self._modules.items()

    @_copy_to_script_wrapper
    def values(self) -> Iterable[Union[torch.nn.Module, None]]:
        return self._modules.values()

    def add_module(self, name: str, module : torch.nn.Module, in_branch: list[Union[int, str]] = [0], out_branch: list[Union[int, str]] = [0], pretrained : bool = True, alias : list[str] = [], requires_grad: Union[bool, None] = None) -> None:
        super().add_module(name, module)
        self._modulesArgs[name] = ModuleArgsDict.ModuleArgs([str(value) for value in in_branch], [str(value) for value in out_branch], pretrained, alias, requires_grad)
    
    def getMap(self):
        results : dict[str, str] = {}
        for name, moduleArgs in self._modulesArgs.items():
            module = self[name]
            if isinstance(module, ModuleArgsDict):
                if len(moduleArgs.alias):
                    count = {k : 0 for k in set(module.getMap().values())}
                    if len(count):
                        for k, v in module.getMap().items():
                            alias_name = moduleArgs.alias[count[v]]
                            if k == "":
                                results.update({alias_name : name+"."+v})
                            else:
                                results.update({alias_name+"."+k : name+"."+v})
                            count[v]+=1
                    else:
                        for alias in moduleArgs.alias:
                            results.update({alias : name})
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
                if module.weight is not None:
                    torch.nn.init.normal_(module.weight, 0.0, std = init_gain)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0.0)

    def named_forward(self, *inputs: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:    
        if len(inputs) > 0:
            branchs: dict[str, torch.Tensor] = {}
            for i, sinput in enumerate(inputs):
                branchs[str(i)] = sinput
            out = inputs[0]
            for name, module in self.items():
                requires_grad = self._modulesArgs[name].requires_grad
                if requires_grad is not None and module:
                    module.requires_grad_(requires_grad)
                for ib in self._modulesArgs[name].in_branch:
                    if ib not in branchs:
                        branchs[ib] = inputs[0]
                for branchs_key in branchs.keys():
                    if str(branchs[branchs_key].device) != "cuda:"+self._modulesArgs[name].gpu:
                        branchs[branchs_key] = branchs[branchs_key].to(int(self._modulesArgs[name].gpu))
                if self._modulesArgs[name].isCheckpoint:
                    out = checkpoint(module, *[branchs[i] for i in self._modulesArgs[name].in_branch])
                    yield name, out
                else:
                    if isinstance(module, ModuleArgsDict):
                        for k, out in module.named_forward(*[branchs[i] for i in self._modulesArgs[name].in_branch]):
                            yield name+"."+k, out
                    elif isinstance(module, torch.nn.Module):
                        out = module(*[branchs[i] for i in self._modulesArgs[name].in_branch])                    
                        yield name, out
                for ob in self._modulesArgs[name].out_branch:
                    branchs[ob] = out
            del branchs

    def forward(self, *input: torch.Tensor) -> torch.Tensor:
        v = input
        for k, v in self.named_forward(*input):
            pass
        return v

    def named_parameters(self, pretrained: bool = False) -> Iterator[tuple[str, torch.nn.parameter.Parameter]]:
        for name, moduleArgs in self._modulesArgs.items():
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
    
    def named_ModuleArgsDict(self) -> Iterator[tuple[str, torch.nn.Module, ModuleArgs]]:
        for name, module in self._modules.items():
            yield name, module, self._modulesArgs[name]
            if isinstance(module, ModuleArgsDict):
                for k, v, u in module.named_ModuleArgsDict():
                    yield name+"."+k, v, u
            

    def _requires_grad(self, keys: list[str]):
        keys = keys.copy()
        for name, module, args in self.named_ModuleArgsDict():
            requires_grad = args.requires_grad
            if requires_grad is not None:
                module.requires_grad_(requires_grad)
            if name in keys:
                keys.remove(name)
                if len(keys) == 0:
                    break
        
class Network(ModuleArgsDict, ABC):

    def _apply_network(self, name_function : Callable[[Self], str], networks: list[str], key: str, function: Callable, *args, **kwargs) -> dict[str, object]:
        results : dict[str, object] = {}
        for module in self.values():
            if isinstance(module, Network):
                if name_function(module) not in networks:
                    networks.append(name_function(module))
                    for k, v in module._apply_network(name_function, networks, key+"."+name_function(module), function, *args, **kwargs).items():
                        results.update({name_function(self)+"."+k : v})
        if len([param.name for param in list(inspect.signature(function).parameters.values()) if param.name == "key"]):
            function = partial(function, key=key)

        results[name_function(self)] = function(self, *args, **kwargs)
        return results
    
    def _function_network(t : bool = False):
        def _function_network_d(function : Callable):
            def new_function(self : Self, *args, **kwargs) -> dict[str, object]:
                return self._apply_network(lambda network: network._getName() if t else network.getName(), [], self.getName(), function, *args, **kwargs)
            return new_function
        return _function_network_d

    def __init__(   self,
                    in_channels : int = 1,
                    optimizer: Union[OptimizerLoader, None] = None, 
                    schedulers: Union[SchedulersLoader, None] = None, 
                    outputsCriterions: Union[dict[str, TargetCriterionsLoader], None] = None,
                    patch : Union[ModelPatch, None] = None,
                    nb_batch_per_step : int = 1,
                    init_type : str = "normal",
                    init_gain : float = 0.02,
                    dim : int = 3) -> None:
        super().__init__()
        self.name = self.__class__.__name__
        self.in_channels = in_channels
        self.optimizerLoader  = optimizer
        self.optimizer : Union[torch.optim.Optimizer, None] = None

        self.schedulersLoader  = schedulers
        self.schedulers : Union[dict[torch.optim.lr_scheduler._LRScheduler, int], None] = None

        self.outputsCriterionsLoader = outputsCriterions
        self.measure : Union[Measure, None] = None

        self.patch = patch

        self.nb_batch_per_step = nb_batch_per_step
        self.init_type  = init_type
        self.init_gain  = init_gain
        self.dim = dim
        self._it = 0
        self.outputsGroup : dict[str, Self]= {}

    @_function_network(True)
    def state_dict(self) -> dict[str, OrderedDict]:
        destination = OrderedDict()
        destination._metadata = OrderedDict()
        destination._metadata[""] = local_metadata = dict(version=self._version)
        self._save_to_state_dict(destination, "", False)
        for name, module in self._modules.items():
            if module is not None:
                if not isinstance(module, Network):   
                    module.state_dict(destination, "" + name + '.', keep_vars=False)
        for hook in self._state_dict_hooks.values():
            hook_result = hook(self, destination, "", local_metadata)
            if hook_result is not None:
                destination = hook_result
        return destination
    
    def load_state_dict(self, state_dict: dict[str, torch.Tensor]):
        missing_keys: list[str] = []
        unexpected_keys: list[str] = []
        error_msgs: list[str] = []

        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata 

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    if not isinstance(child, Network):
                        load(child, prefix + name + '.')

        load(self)
        del load

        if len(unexpected_keys) > 0:
            error_msgs.insert(
                0, 'Unexpected key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in unexpected_keys)))
        if len(missing_keys) > 0:
            error_msgs.insert(
                0, 'Missing key(s) in state_dict: {}. '.format(
                    ', '.join('"{}"'.format(k) for k in missing_keys)))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               self.__class__.__name__, "\n\t".join(error_msgs)))
        
    @_function_network(True)
    def load(self, state_dict : dict[str, dict[str, torch.Tensor]], init: bool = True, ema : bool =False):
        if init:
            self.apply(partial(ModuleArgsDict.init_func, init_type=self.init_type, init_gain=self.init_gain))
        name = "Model" + ("_EMA" if ema else "")
        if name in state_dict:
            model_state_dict_tmp = {k.split(".")[-1] : v for k, v in state_dict[name].items()}[self._getName()]

            map = self.getMap()
            model_state_dict : OrderedDict[str, torch.Tensor] = OrderedDict()
            
            for alias in model_state_dict_tmp.keys():
                prefix = ".".join(alias.split(".")[:-1])
                alias_list = [(".".join(prefix.split(".")[:len(i.split("."))]), v) for i, v in map.items() if prefix.startswith(i)]

                if len(alias_list):
                    for a, b in alias_list:
                        model_state_dict[alias.replace(a, b)] = model_state_dict_tmp[alias]
                        break
                else:
                    model_state_dict[alias] = model_state_dict_tmp[alias]
            self.load_state_dict(model_state_dict)
        if "{}_optimizer_state_dict".format(name) in state_dict and self.optimizer:
            self.optimizer.load_state_dict(state_dict['{}_optimizer_state_dict'.format(name)])
        self.initialized()

    def _compute_channels_trace(self, module : ModuleArgsDict, in_channels : int, gradient_checkpoints: Union[list[str], None], gpu_checkpoints: Union[list[str], None], name: Union[str, None] = None, in_is_channel = True, out_channels : Union[int, None] = None, out_is_channel = True) -> tuple[int, bool, int, bool]:
        for k, v in module.items():
            if hasattr(v, "in_channels"):
                if v.in_channels:
                    in_channels = v.in_channels
            if hasattr(v, "in_features"):
                if v.in_features:
                    in_channels = v.in_features
            key = name+"."+k if name else k
            isCheckpoint = False
            isGPU_Checkpoint = False
            if gradient_checkpoints:
                isCheckpoint = key in gradient_checkpoints
            if gpu_checkpoints:
                isGPU_Checkpoint = key in gpu_checkpoints

            module._modulesArgs[k].isCheckpoint = isCheckpoint
            module._modulesArgs[k].isGPU_Checkpoint = isGPU_Checkpoint
            module._modulesArgs[k].in_channels = in_channels
            module._modulesArgs[k].in_is_channel = in_is_channel
            
            if isinstance(v, ModuleArgsDict):
                in_channels, in_is_channel, out_channels, out_is_channel = self._compute_channels_trace(v, in_channels, gradient_checkpoints, gpu_checkpoints, key, in_is_channel, out_channels, out_is_channel)
            
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

    @_function_network()
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
            self.measure.init(self)
    
    def initialized(self):
        pass

    def named_forward(self, *inputs: torch.Tensor) -> Iterator[tuple[str, torch.Tensor]]:
        if self.patch:
            self.patch.load(inputs[0].shape[2:])
            acc = Accumulator(self.patch.patch_slices, self.patch.patchCombine)


            patchIterator = self.patch.disassemble(*inputs)
            for i, patch_input in enumerate(patchIterator):
                for (name, output_layer) in super().named_forward(*patch_input):
                    yield "{}{}".format("accu:", name), output_layer
                acc.addLayer(i, output_layer)
            yield name, acc.assemble()
        else:
            for (name, output_layer) in super().named_forward(*inputs):
                yield name, output_layer
    
    def get_layers(self, inputs : list[torch.Tensor], layers_name: list[str]) -> Iterator[tuple[str, torch.Tensor]]:
        layers_name = layers_name.copy()
        output_layer_accumulator : dict[str, Accumulator] = {}
        output_layer_index : dict[str, int] = {}
        
        for (nameTmp, output_layer) in self.named_forward(*inputs):
            name = nameTmp.replace("accu:", "")
            
            if name in layers_name:
                accu = "accu:" in nameTmp
                if accu:
                    if name not in output_layer_accumulator:
                        networkName = nameTmp.split("accu:")[-2].split(".")[-1]
                        module = self
                        network = None

                        if networkName == "":
                            network = module
                        else:
                            for n in name.split("."):
                                module = module[n]
                                if isinstance(module, Network) and n == networkName:
                                    network = module
                                    break
                        if network and network.patch:
                            output_layer_accumulator[name] = Accumulator(network.patch.patch_slices, network.patch.patchCombine)
                            output_layer_index[name] = 0
                    
                    output_layer_accumulator[name].addLayer(output_layer_index[name], output_layer)
                    del output_layer
                    output_layer_index[name] += 1
                    if output_layer_accumulator[name].isFull():
                        output_layer = output_layer_accumulator[name].assemble()
                        output_layer_accumulator.pop(name)
                        output_layer_index.pop(name)
                        layers_name.remove(name)
                        yield name, output_layer
                else:
                    layers_name.remove(name)
                    yield name, output_layer

            if not len(layers_name):
                break
            
    def _layer_function(self, name: str, output_layer: torch.Tensor, data_dict: dict[tuple[str, bool], torch.Tensor]):
        self.measure.update(name, output_layer, {k[0] : v for k, v in data_dict.items()}, self._it)

    def init_outputsGroup(self):
        metric_tmp = {network : network.measure.outputsCriterions.keys() for network in self.getNetworks().values() if network.measure}
        for k, v in metric_tmp.items():
            for a in v:
                self.outputsGroup[a] = k

    def forward(self, data_dict: dict[tuple[str, bool], torch.Tensor], output_layers: list[str] = []) -> list[tuple[str, torch.Tensor]]:
        if not len(self.outputsGroup) and not len(output_layers):
            return []

        self.resetLoss()
        results = []
        for name, layer in self.get_layers([v for k, v in data_dict.items() if k[1]], list(set(list(self.outputsGroup.keys())+output_layers))):
            if name in self.outputsGroup:
                self.outputsGroup[name]._layer_function(name, layer, data_dict)
            if name in output_layers:
                results.append((name, layer))
        return results

    @_function_network()
    def resetLoss(self):
        if self.measure:
            self.measure.resetLoss()

    @_function_network()
    def backward(self, model: Self):
        if self.measure:
            if self.measure.isDone() and self.measure.loss is not None:
                if self.scaler and self.optimizer:
                    model._requires_grad(list(self.measure.outputsCriterions.keys()))
                    self.scaler.scale(self.measure.loss / self.nb_batch_per_step).backward()
                    if self._it % self.nb_batch_per_step == 0:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                        self.optimizer.zero_grad(set_to_none=True)
                self._it += 1

    @_function_network()
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
                    scheduler.step(self.measure.mean())
            else:
                scheduler.step()     

    @_function_network()
    def measureClear(self):
        if self.measure:
            self.measure.clear()
    
    @_function_network()
    def getNetworks(self) -> Self:
        return self

    def to(module : ModuleArgsDict, device: int):
        for k, v in module.items():
            if module._modulesArgs[k].isGPU_Checkpoint:
                device+=1
            module._modulesArgs[k].gpu = str(device)

            if isinstance(v, ModuleArgsDict):
                v = Network.to(v, device)
            else:
                v = v.to(device)
        return module
                
    def getName(self) -> str:
        return self.__class__.__name__
    
    def setName(self, name: str) -> Self:
        self.name = name
        return self
    
    def _getName(self) -> str:
        return self.name

class ModelLoader():

    @config("Model")
    def __init__(self, classpath : str = "default:segmentation.UNet") -> None:
        self.module, self.name = _getModule(classpath.split(".")[-1] if len(classpath.split(".")) > 1 else classpath, "models" + "."+".".join(classpath.split(".")[:-1]) if len(classpath.split(".")) > 1 else "")
        
    def getModel(self, train : bool = True, DL_args: Union[str, None] = None, DL_without=["optimizer", "schedulers", "nb_batch_per_step", "init_type", "init_gain"]) -> Network:
        if not DL_args:
            DL_args="{}.Model".format(os.environ["DEEP_LEARNING_API_ROOT"])
        model = partial(getattr(importlib.import_module(self.module), self.name), config = None, DL_args=DL_args)
        if not train: 
            model = partial(model, DL_without = DL_without)
        return model()
    