# -*- coding: utf-8 -*- 
import torch.nn as nn 
import torch.optim as optim 

from revo import config
from revo.nlu.model.domainModel import DomainModel
from revo.util.dataloader.domainLoader import DomainDataLoader

class Trainer:
    def __init__(self, optim_method, schedule):
        pass 

    def training(self, model, dataloader):
        pass 

    def train_step(self, ):
        pass 

    def evaluate(self, ):
        pass

    def save(self, ):
        pass  

    def record(self, ):
        pass 

        

if __name__ == "__main__":


    tasks = ["nlu_domain"] 