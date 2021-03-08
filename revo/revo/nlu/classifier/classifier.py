# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 

class BaseClassifier(nn.Module):
    def __init__(self):
        super(BaseClassifier, self).__init__()

class SimpleClassifier(BaseClassifier):
     def __init__(self):
        super(SimpleClassifier, self).__init__()
