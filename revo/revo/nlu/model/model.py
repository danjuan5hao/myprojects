# -*- coding: utf-8 -*- 
import torch.nn as nn 

from revo.nlu.featurizer.embeddingFeaturizer import FasttextEmbedding 

class NLUModel(nn.Module):
    def __init__(self):
        super(NLUModel, self).__init__()
        self.embedding = FasttextEmbedding.load()

    def forward(self, x):
        pass 

    def tokenize(self, x):
        rst = self.embedding.serialize(x)
        return rst

    def forward(self, x):
        pass 
    
        

    