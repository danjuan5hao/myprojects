# -*- coding: utf-8 -*-
from itertools import chain

import torch 
import torch.nn as nn 

class TextCnnClassifier(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(TextCnnClassifier, self).__init__()
        self.conv3s = nn.ModuleList([nn.Sequential(
            collections.OrderedDict([
                ('conv1', nn.Conv1d(in_dim, out_dim, kernel_size=2) ),
                ('relu1', nn.ReLU() ),
                ('maxpool',  nn.AdaptiveMaxPool1d(1) ),
            ])
        ) for _ in range(60)])

        self.conv4s = nn.ModuleList([nn.Sequential(
            collections.OrderedDict([
                ('conv1', nn.Conv1d(in_dim, out_dim, 3) ),
                ('relu1', nn.ReLU() ),
                ('maxpool',  nn.AdaptiveMaxPool1d(1) ),
            ])
        ) for _ in range(60)])

        self.conv5s = nn.ModuleList([nn.Sequential(
            collections.OrderedDict([
                ('conv1', nn.Conv1d(in_dim, out_dim, 4) ),
                ('relu1', nn.ReLU() ),
                ('maxpool',  nn.AdaptiveMaxPool1d(1) ),
            ])
        ) for _ in range(60)])

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = x.permute(0,2,1)
        conv3s_outputs = [layer(x) for layer in self.conv3s ]
        conv4s_outputs = [layer(x) for layer in self.conv4s ]
        conv5s_outputs = [layer(x) for layer in self.conv5s ]
        output = torch.cat([*chain(conv3s_outputs, conv4s_outputs, conv5s_outputs)], dim=2)
        output.squeeze_()
        return self.dropout(output) 

    def save(self, path):
        pass 



