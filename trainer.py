import importlib
from time import sleep
from xmlrpc.client import boolean
import torch
import tqdm
from typing import List, Dict
import numpy as np
import datetime
import os
from enum import Enum
import time
from . import Data
from . import config

from torch.utils.tensorboard import SummaryWriter

MODELS_DIRECTORY = "./Models/"
CHECKPOINTS_DIRECTORY = "./Checkpoints/"
PREDICTION_DIRECTORY = "./Predictions/"

date = lambda : datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

class State(Enum):
    TRAIN = 0,
    RESUME = 1

class Optimizer():
    
    @config("Trainer.Optimizer")
    def __init__(self, name : str = "AdamW") -> None:
        self.name = name
    
    def getOptimizer(self, model):
        return config("Trainer.Optimizer")(getattr(importlib.import_module('torch.optim'), self.name))(model.parameters(), config = "Config.yml")

class Scheduler():

    @config("Trainer.Scheduler")
    def __init__(self, name : str = "ReduceLROnPlateau") -> None:
        self.name = name

    def getSheduler(self, optimizer):
        return config("Trainer.Scheduler")(getattr(importlib.import_module('torch.optim.lr_scheduler'), self.name))(optimizer, config = "Config.yml")

class Criterion():

    @config("Criterion")
    def __init__(self, module : str = "torch.nn", l : float = 1) -> None:
        self.module = module
        self.l = l

    def getCriterion(self, name):
        return getattr(importlib.import_module(self.module), name)()#config("Trainer.Criterion")(criterion)(config = "Config.yml")

class Criterions():

    @config("Trainer")
    def __init__(self, criterions : Dict[str, Criterion] = {"CrossEntropyLoss" : Criterion()}) -> None:
        self.criterions = criterions

    def getCriterions(self):
        for key in self.criterions:
            self.criterions[key] = (self.criterions[key].l, self.criterions[key].getCriterion(key))
        return self.criterions

class Model():

    @config("Model")
    def __init__(self, classpath : str = "network.UNet") -> None:
        self.module = ".".join(classpath.split(".")[:-1])
        self.name = classpath.split(".")[-1]
        
    def getModel(self):
        return getattr(importlib.import_module(self.module), self.name)(config = "Config.yml")

class Trainer:

    @config("Trainer")
    def __init__(   self,
                    model : Model = Model(),
                    dataset : Data = Data(), 
                    groupsInput : List[str] = ["group"],
                    train_name : str = "name",
                    device : int = 0,
                    optimizer : Optimizer = Optimizer(),
                    scheduler : Scheduler = Scheduler(),
                    criterions: Dict[str, Criterions] = {"Group" : Criterions()},
                    nb_batch_per_step : int = 1,
                    epochs: int = 100,
                    log_dir : str = "./Statistics/") -> None:
        
        if False: #torch.cuda.is_available():
            device = torch.device('cuda:'+str(device))
            torch.backends.cudnn.benchmark = True
        else:
            device = torch.device('cpu')
        
        self.model = model.getModel().to(device)
        self.train_name = train_name
        self.device = device
        self.dataset = dataset
        self.groupsInput = groupsInput

        self.optimizer = optimizer.getOptimizer(self.model)
        self.scheduler = scheduler.getSheduler(self.optimizer)
        self.criterions = criterions
        for key in self.criterions:
            self.criterions[key] = self.criterions[key].getCriterions()

        self.epochs = epochs
        self.nb_batch_per_step = nb_batch_per_step
        self.epoch = 0
        
        self.tb = SummaryWriter(log_dir = log_dir+self.train_name+"/")

    def __enter__(self):
        self.dataset.__enter__()
        self.dataloader_training, self.dataloader_validation = self.dataset.getData()
        return self
    
    def __exit__(self, type, value, traceback):
        self.dataset.__exit__(type, value, traceback)

    def train(self, state : State):
        if state == State.RESUME:
            self.checkpoint_load()
            self._run()
        if state == State.TRAIN:
            self._run()

    def _run(self) -> None:
        with tqdm.tqdm(iterable = range(self.epoch, self.epochs), total=self.epochs, initial=self.epoch, desc="Progress") as epoch:
            for _ in epoch:
                self._train()
                eval_score = self._validate()
                if self.scheduler is not None:
                    if self.scheduler.__class__.__name__ == 'ReduceLROnPlateau':
                        self.scheduler.step(eval_score)
                    else:
                        self.scheduler.step()
                self.tb.add_scalar("Learning Rate", self.optimizer.param_groups[0]['lr'], self.epoch)
                self.epoch += 1  
                self.checkpoint_save()
        self.save()
        self.tb.close()
    
    def _train(self) -> None:
        self.model.train()
        scaler = torch.cuda.amp.GradScaler()

        training_losses = []
        self.optimizer.zero_grad()

        with tqdm.tqdm(iterable = enumerate(self.dataloader_training), desc = "Training", total=len(self.dataloader_training), leave=False) as batch_iter:
            for i, data_dict in batch_iter:
                #with torch.cuda.amp.autocast():
                if len(self.groupsInput) > 1:
                    input = torch.cat((torch.unsqueeze(data_dict[group]) for group in self.groupInput), dim=0)
                else:
                    input = data_dict[self.groupsInput[0]]

                out = self.model(input.to(self.device))
                loss, losses = self._loss(out, data_dict)
                loss = loss / self.nb_batch_per_step
    
                #scaler.scale(loss).backward()
                loss.backward()

                if (i+1) % self.nb_batch_per_step == 0:
                    #scaler.step(self.optimizer)
                    #scaler.update()
                    self.optimizer.step()
                    self.optimizer.zero_grad() 

                
                batch_iter.set_description(f'Training: (loss {loss.item()*self.nb_batch_per_step:.4f})')
                training_losses.append(losses)
        training_losses =  np.mean(np.array(training_losses), axis=0)
        
        training_loss = dict()
        i = 0
        for type in self.criterions:
            for criterion, _ in self.criterions[type]:
                training_loss[type.name+":"+criterion.__class__.__name__] = training_losses[i]
                i+=1
        
        self.tb.add_scalars("Loss/Trainning", training_loss, self.epoch)

        for name, weight in self.model.named_parameters():
            self.tb.add_histogram(name,weight, self.epoch)
            self.tb.add_histogram(f'{name}.grad',weight.grad, self.epoch)
        self.tb.add_image("result/Train", np.add.reduce(torch.argmax(torch.softmax(out[0, ...], dim = 0), dim=0).cpu().numpy(), axis = 2), self.epoch, dataformats='HW') 
        return np.sum(training_losses)

    def _loss(self, out : torch.Tensor, data_dict : Dict[str, torch.Tensor]) -> torch.Tensor:
        loss = 0
        losses = np.array([])
        for group in self.criterions:
            data = data_dict[group].to(self.device, non_blocking=False)
            print(self.criterions[group].values())

            for l, criterion in self.criterions[group].values():
                print(data.shape)
                print(out.shape)
                result = criterion(out, data)
                losses = np.append(losses, result.item())  
                loss = loss + l*result
        return loss, losses

    @torch.no_grad()
    def _validate(self) -> None:

        self.model.eval()
        validation_losses = []
        with tqdm.tqdm(iterable = enumerate(self.dataloader_validation), desc = "Validation", total=len(self.dataloader_validation), leave=False) as batch_iter:
            for i, data_dict in batch_iter:

                out = self.model(data_dict[self.inputType].to(self.device))
                loss, losses = self._loss(out, data_dict)
                    
                validation_losses.append(losses)

                batch_iter.set_description(f'Validation: (loss {loss.item():.4f})')
        validation_losses = np.mean(np.array(validation_losses), axis=0)
        validation_loss = dict()
        i = 0
        for type in self.criterions:
            for criterion, _ in self.criterions[type]:
                validation_loss[type.name+":"+criterion.__class__.__name__] = validation_losses[i]
                i+=1
 
        self.tb.add_scalars("Loss/Validation", validation_loss, self.epoch)
        self.tb.add_image("result/Validation", np.add.reduce(torch.argmax(torch.softmax(out[0, ...], dim = 0), dim=0).cpu().numpy(), axis = 2), self.epoch, dataformats='HW') 
        return np.sum(validation_losses)


    def checkpoint_save(self, name : str = "date", path : str = "default") -> None:
        path = (CHECKPOINTS_DIRECTORY+self.model_name+"/" if path == "default" else path)
        name = (date() if name == "date" else name)+".pt"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save({
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict()}, 
            path+name)
        
    def save(self, name : str = "date", path : str = "default") -> None:
        path = (MODELS_DIRECTORY+self.model_name+"/" if path == "default" else path)
        name = (date() if name == "date" else name)+".pt"
        if not os.path.exists(path):
            os.makedirs(path)
        torch.save(self.model.state_dict(), path+name)

    def checkpoint_load(self, name : str = "last", path : str = "default") -> None:
        path = (CHECKPOINTS_DIRECTORY+self.model_name+"/" if path == "default" else path)
        name = (sorted(os.listdir(path))[-1] if name == "last" else name+".pt")
        checkpoint = torch.load(path+name)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        
    def load(self, name : str = "last", path : str = "default") -> None:
        path = MODELS_DIRECTORY+self.model_name+"/" if path == "default" else path
        name = (sorted(os.listdir(path))[-1] if name == "last" else name+".pt")
        self.model.load_state_dict(torch.load(path+name))