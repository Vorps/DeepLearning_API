import importlib
import torch
import tqdm
from typing import List, Dict, Tuple
import numpy as np
import datetime
import os
from enum import Enum
from DeepLearning_API import DataTrain, config, MODELS_DIRECTORY, CHECKPOINTS_DIRECTORY, STATISTICS_DIRECTORY, SETUPS_DIRECTORY, CONFIG_FILE, gpuInfo, getDevice, _getModule
import datetime

DATE = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

import shutil
import pynvml

from torch.utils.tensorboard import SummaryWriter

class State(Enum):
    TRAIN = "TRAIN"
    RESUME = "RESUME"
    PREDICTION = "PREDICTION"
    
    def __str__(self) -> str:
        return self.value

class Optimizer():
    
    @config("Optimizer")
    def __init__(self, name : str = "AdamW") -> None:
        self.name = name
    
    def getOptimizer(self, model : str) -> torch.nn.Module:
        return config("Trainer.Optimizer")(getattr(importlib.import_module('torch.optim'), self.name))(model.parameters(), config = None)

class Scheduler():

    @config("Scheduler")
    def __init__(self, name : str = "ReduceLROnPlateau") -> None:
        self.name = name

    def getSheduler(self, optimizer : str) -> torch.nn.Module:
        return config("Trainer.Scheduler")(getattr(importlib.import_module('torch.optim.lr_scheduler'), self.name))(optimizer, config = None)

class Criterion():

    @config(None)
    def __init__(self, group : str = "Default", l : float = 1) -> None:
        self.l = l
        self.group = group

    def getCriterion(self, classpath : str, group : str) -> torch.nn.Module:
        module, name = _getModule(classpath, "criterion")

        return config("Trainer.criterions."+group+".criterion."+classpath)(getattr(importlib.import_module(module), name))(config = None)

class Criterions():

    @config(None)
    def __init__(self, criterion : Dict[str, Criterion] = {"default:torch_nn_CrossEntropyLoss:Dice:NCC" : Criterion()}) -> None:
        self.criterions = criterion

    def getCriterions(self, group) -> Dict[str, Tuple[float, torch.nn.Module]]:
        for key in self.criterions:
            self.criterions[key] = (self.criterions[key].group, self.criterions[key].l, self.criterions[key].getCriterion(key, group))
        return self.criterions

class Model():

    @config("Model")
    def __init__(self, classpath : str = "default:UNet") -> None:
        self.module, self.name = _getModule(classpath, "network")
        
    def getModel(self) -> torch.nn.Module:
        return getattr(importlib.import_module(self.module), self.name)(config = None, args="Trainer.Model")
    
class Trainer:

    @config("Trainer")
    def __init__(   self,
                    model : Model = Model(),
                    dataset : DataTrain = DataTrain(), 
                    groupsInput : List[str] = ["default"],
                    train_name : str = "default",
                    device : int = None,
                    optimizer : Optimizer = Optimizer(),
                    scheduler : Scheduler = Scheduler(),
                    criterions: Dict[str, Criterions] = {"default" : Criterions()},
                    nb_batch_per_step : int = 1,
                    manual_seed : int = None,
                    epochs: int = 100,
                    it_validation : int = None,
                    autocast : bool = False) -> None:
        if os.environ["DEEP_LEANING_API_CONFIG_MODE"] != "Done":
            exit(0)
            
        self.manual_seed = manual_seed

        if self.manual_seed is not None:
            torch.manual_seed(self.manual_seed)
        
        torch.backends.cudnn.deterministic = self.manual_seed is not None
        torch.backends.cudnn.benchmark = self.manual_seed is None
        
        self.device         = getDevice(device)
        self.train_name     = train_name
        
        self.dataset        = dataset
        self.groupsInput    = groupsInput
        self.autocast       = autocast
        
        self.model          = model.getModel().to(self.device)
        
        self.optimizer      = optimizer.getOptimizer(self.model)
        self.scheduler      = scheduler.getSheduler(self.optimizer)
        self.criterions     = criterions
        for classpath in self.criterions:
            self.criterions[classpath] = self.criterions[classpath].getCriterions(classpath)
        
        self.epochs = epochs
        self.nb_batch_per_step = nb_batch_per_step
        self.epoch = 0
        self.it = 0
        self.epoch_it = 0
        self.it_validation = it_validation
        self.checkpoints : Dict[str, float] = {}
        self.tb = None

    def __enter__(self) -> None:
        pynvml.nvmlInit()
        self.dataset.__enter__()
        self.dataloader_training, self.dataloader_validation = self.dataset.getData(self.manual_seed)
        if self.it_validation is None:
            self.it_validation = len(self.dataloader_training) 
        return self
    
    def __exit__(self, type, value, traceback) -> None:
        if len(self.checkpoints) > 0 and self.epoch_it > 0:
            self.save(checkpoint_filename=list(self.checkpoints)[np.argmin(list(self.checkpoints.values()))])
        self.dataset.__exit__(type, value, traceback)
        if self.tb is not None:
            self.tb.close()
        pynvml.nvmlShutdown()

    def train(self, state : State) -> None:
        if state == State.RESUME:
            self.checkpoint_load()
        if state == State.TRAIN and os.path.exists(STATISTICS_DIRECTORY()+self.train_name+"/"):
            accept = input("The model {} already exists ! Do you want to overwrite it (yes,no) : ".format(self.train_name))
            if accept != "yes":
                return
            else:
                for directory_path in [STATISTICS_DIRECTORY(), MODELS_DIRECTORY(), CHECKPOINTS_DIRECTORY(), SETUPS_DIRECTORY()]:
                    if os.path.exists(directory_path+self.train_name+"/"):
                        shutil.rmtree(directory_path+self.train_name+"/")

        self.tb = SummaryWriter(log_dir = STATISTICS_DIRECTORY()+self.train_name+"/")
        
        config_namefile_src = CONFIG_FILE().replace(".yml", "")
        self.config_namefile = SETUPS_DIRECTORY()+self.train_name+"/"+config_namefile_src.split("/")[-1]+"_"+str(self.it)+".yml"
        if not os.path.exists(SETUPS_DIRECTORY()+self.train_name+"/"):
            os.makedirs(SETUPS_DIRECTORY()+self.train_name+"/")
        shutil.copyfile(config_namefile_src+".yml", self.config_namefile)

        self.dataset.load()
        self._run()

    def _run(self) -> None:
        with tqdm.tqdm(iterable = range(self.epoch, self.epochs), total=self.epochs, initial=self.epoch, desc="Progress") as epoch:
            for self.epoch in epoch:
                self._train()
    
    def _loss(self, out_dict : torch.Tensor, data_dict : Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0
        losses = np.array([])
        for group in self.criterions:
            output = out_dict[group] if group in out_dict else None
            for true_group, l, criterion in self.criterions[group].values():
                target = data_dict[true_group].to(self.device, non_blocking=False) if true_group in data_dict else None
                result = criterion(output, target)
                losses = np.append(losses, result.item())  
                loss = loss + l*result
        return loss, losses
    
    def getInput(self, data_dict : Dict[str, List[torch.Tensor]]) -> torch.Tensor:    
        input = torch.cat([data_dict[group] for group in self.groupsInput], dim=0)
        return torch.unsqueeze(input, dim=0)

    def _train(self) -> None:
        self.model.train()
        scaler = torch.cuda.amp.GradScaler(enabled=self.autocast)

        training_losses = []
        training_loss = []
        self.optimizer.zero_grad()
        description = lambda loss : "Training : (Loss {:.4f}) ".format(loss)+gpuInfo(self.device)

        with tqdm.tqdm(iterable = enumerate(self.dataloader_training), desc = description(0), total=len(self.dataloader_training), leave=False) as batch_iter:
            for i, data_dict in batch_iter:
                with torch.cuda.amp.autocast(enabled=self.autocast):
                    input = self.getInput(data_dict)
                    out = self.model(input.to(self.device))
                    
                    loss, losses = self._loss(out, data_dict)
                    loss = loss / self.nb_batch_per_step

                    if (i+1) % self.nb_batch_per_step == 0 or (i+1) % self.it_validation == 0:
                        scaler.scale(loss).backward()
                        scaler.step(self.optimizer)
                        scaler.update()
                        self.optimizer.zero_grad()

                    if (self.it+1) % self.it_validation == 0:
                        name = self.checkpoint_save()
                        if self.dataloader_validation is not None:
                            mean_validation_loss = self._validate()
                            self.checkpoints[name] = mean_validation_loss
                        else:
                            self.checkpoints[name] = np.mean(training_loss)
                        
                        self._train_log(out, training_losses)
                        training_losses = []
                        self.model.train()
                        self.epoch_it += 1

                batch_iter.set_description(description(loss.item()*self.nb_batch_per_step))
                training_losses.append(losses)
                training_loss.append(loss.item())
                
                self.it += 1

        return np.mean(training_loss)

    @torch.no_grad()
    def _validate(self) -> None:

        self.model.eval()
        validation_losses = []
        validation_loss = []
        description = lambda loss : "Validation : (Loss {:.4f}) ".format(loss)+gpuInfo(self.device)
        with tqdm.tqdm(iterable = enumerate(self.dataloader_validation), desc = description(0), total=len(self.dataloader_validation), leave=False) as batch_iter:
            for _, data_dict in batch_iter:
                input = self.getInput(data_dict)
                out = self.model(input.to(self.device))
                loss, losses = self._loss(out, data_dict)
                    
                validation_losses.append(losses)
                validation_loss.append(loss.item())

                batch_iter.set_description(description(loss.item()))

        if self.scheduler is not None:
            if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                self.scheduler.step(np.mean(validation_loss))
            else:
                self.scheduler.step()
        
        self._validation_log(out, validation_losses)
        
        return np.mean(validation_loss)


    def checkpoint_save(self) -> None:
        path = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
        name = DATE()+".pt"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
            'epoch': self.epoch,
            'it': self.it,
            'checkpoints' : self.checkpoints,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, 
            path+name)
        return path+name
        
    def save(self, checkpoint_filename) -> None:
        path = MODELS_DIRECTORY()+self.train_name+"/"
        name = DATE()+".pt"
        if not os.path.exists(path):
            os.makedirs(path)
        checkpoint = torch.load(checkpoint_filename)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        torch.save(self.model, path+name)
        os.rename(self.config_namefile, self.config_namefile.replace(".yml", "")+"_"+str(self.it)+".yml")

    def checkpoint_load(self) -> None:
        path = CHECKPOINTS_DIRECTORY()+self.train_name+"/"
        if os.path.exists(path) and os.listdir(path):
            name = sorted(os.listdir(path))[-1]
            checkpoint = torch.load(path+name)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.epoch = checkpoint['epoch']
            self.it = checkpoint['it']
            self.checkpoints = checkpoint['checkpoints']
        else:
            raise Exception("Model : {} does not exist !".format(self.train_name))

    def load(self) -> None:
        path = MODELS_DIRECTORY()+self.train_name+"/"
        if os.path.exists(path) and os.listdir(path):
            name = sorted(os.listdir(path))[-1]
            self.model.load_state_dict(torch.load(path+name))
        else:
            raise Exception("Model : {} does not exist !".format(self.train_name))


    def _loss(self, out_dict : torch.Tensor, data_dict : Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0
        losses = np.array([])
        for group in self.criterions:
            for true_group, l, criterion in self.criterions[group].values():
                input = data_dict[true_group].to(self.device, non_blocking=False) if true_group in data_dict else None
                target = out_dict[group] if group in out_dict else None
                result = criterion(input, target)
                losses = np.append(losses, result.item())  
                loss = loss + l*result
        return loss, losses
    

    def _loss_format(self, losses : List[float]):
        training_loss = dict()
        i = 0
        for group in self.criterions:
            for true_group, l, criterion in self.criterions[group].values():
                training_loss[group+":"+criterion.__class__.__name__] = losses[i]
                i+=1
        return training_loss
        
    def _train_log(self, out, losses):
        print(np.mean(np.array(losses), axis=0))
        self.tb.add_scalars("Loss/Trainning", self._loss_format(np.mean(np.array(losses), axis=0)), self.it)

        """for name, weight in self.model.named_parameters():
            print(name)
            self.tb.add_histogram(name, weight, self.it)
            self.tb.add_histogram(f'{name}.grad',weight.grad, self.it)"""
        image =  self.model.logImage(out)
        if image is not None:
            self.tb.add_image("result/Train", image, self.it, dataformats='HW') 


    def _validation_log(self, out, losses):
        self.tb.add_scalars("Loss/Validation", self._loss_format(np.mean(np.array(losses), axis=0)), self.it)
        image =  self.model.logImage(out)
        if image is not None:
            self.tb.add_image("result/Validation", image, self.it, dataformats='HW') 
        self.tb.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], self.it)

