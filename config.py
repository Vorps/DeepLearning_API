from ast import Param
import os
from typing import Dict, List, Union
import torch
import yaml
import inspect 
import collections
from copy import deepcopy

class ConfigError(Exception):

    def __init__(self, message : str = "The config only supports types : config(Object), int, str, bool, float, List[int], List[str], List[bool], List[float], Dict[str, Object]"):
        self.message = message
        super().__init__(self.message)

class Config():

    def __init__(self, filename, key) -> None:
        self.filename = filename
        self.keys = key.split(".")

    def __enter__(self):
        if not os.path.exists(self.filename):
            with  open(self.filename, "w") as f:
                pass    
        self.yml = open(self.filename, 'r')

        self.data = yaml.load(self.yml, Loader=yaml.FullLoader)
        if self.data == None:
            self.data = {}
        
        self.config = self.data

        for key in self.keys:
            if self.config == None or key not in self.config:
                self.config = {key : {}}
            
            self.config = self.config[key]
        return self

    def createDictionary(self, data, keys, i):
        if keys[i] not in data:
            data = {keys[i]: data}
        if i == 0:
            return data
        else:
            i -= 1
            return self.createDictionary(data, keys, i)
    
    def merge(self, dict1, dict2):
        result = deepcopy(dict1)

        for key, value in dict2.items():
            if isinstance(value, collections.Mapping):
                result[key] = self.merge(result.get(key, {}), value)
            else:
                if not dict2[key] == None:
                    result[key] = deepcopy(dict2[key])
        return result


    def __exit__(self, type, value, traceback):
        self.yml.close()
        with open(self.filename, 'r') as yml:
            data = yaml.load(yml, Loader=yaml.FullLoader)
            if data == None:
                data = {}
        with open(self.filename, 'w') as yml:
            yaml.dump(self.merge(data, self.createDictionary(self.config, self.keys, len(self.keys)-1)), yml)
        
        
    def getValue(self, name, default):
        default = None if default == inspect._empty else default
        value = default
        if name in self.config and self.config[name] is not None:
            value = self.config[name]
            if value == None:
                value = default
            value_config = value
        else:
            value_config = value
            if type(value) == tuple:
                value_config = list(value)
            if type(value) == dict:
                value_config = {}
                for key in value:
                    value_config[key] = None 
        self.config[name] = value_config
        return value

def config(key : str = None):
    def decorator(function):
        def new_function(*args, **kwargs):
            if "config" in kwargs:
                filename =  kwargs["config"] 
                key_tmp =  kwargs["args"] if "args" in kwargs else key
                with Config(filename, key_tmp) as config:
                    kwargs = {} 
                    for param in list(inspect.signature(function).parameters.values())[len(args):]:
                        if not param.annotation == inspect._empty:
                            if param.annotation not in [int, str, bool, float, Union]:
                                if "__extra__" in param.annotation.__dict__:
                                    if param.annotation.__extra__ == list or param.annotation.__extra__ == tuple:
                                        if param.annotation.__args__[0] in [int, str, bool, float]:
                                            values = config.getValue(param.name, param.default)
                                            kwargs[param.name] = values
                                        else:
                                            raise ConfigError()
                                    elif param.annotation.__extra__ == dict:
                                        if param.annotation.__args__[0] == str:
                                            values = config.getValue(param.name, param.default)
                                            if param.annotation.__args__[1] not in [int, str, bool, float]:
                                                kwargs[param.name] = {value : param.annotation.__args__[1](config = filename, args = key_tmp+"."+param.name+"."+value) for value in values}
                                            else:
                                                kwargs[param.name] = values
                                        else: 
                                            raise ConfigError()
                                    else:
                                        raise ConfigError()
                                else:
                                    kwargs[param.name] = param.annotation(config = filename)
                            else:
                                kwargs[param.name] = config.getValue(param.name, param.default)
                        elif param.name != "self":
                            kwargs[param.name] = config.getValue(param.name, param.default)
            result = function(*args, **kwargs)
            return result
        return new_function
    return decorator
    