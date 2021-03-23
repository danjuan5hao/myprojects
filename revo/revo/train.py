# -*- coding: utf-8 -*- 
import torch.nn as nn 
import torch.optim as optim 

from revo import config
from revo.nlu.model.domainModel import DomainModel
from revo.util.dataloader.domainLoader import DomainDataLoader 


def train_nlu_domain(domains)
    domain_model = DomainModel.load()
    # domain_model.embedding.save_stoi()
    stoi = domain_model.embedding.stoi
    tokenizer = domain_model.embedding.tokenizer 
    domain_names = domains.names
    domain_dataloader = DomainDataLoader(stoi, tokenizer, domain_names, data_path_dir)

    domain_model.train()

    cce = nn.CrossEntropyLoss()
    optimer = optim.Adam(domain_model.parameters)

    loss = 
    for i in range(10):
        for batch in domain_dataloader.dataloader:
            texts, labels = batch 
            pred  = domain_model(texts)
            loss = cce(pred, labels)

            loss.backwards()
            optimer.step()
            
            




    # -training
    # val
    # save 






if __name__ == "__main__":


    tasks = ["nlu_domain"] 