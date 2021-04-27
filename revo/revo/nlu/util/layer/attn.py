# -*- coding: utf-8 -*-
import torch 
import torch.nn as nn 

class SelfAttn(nn.Module):
    def __init__(self, ):
        super(selfAttn, self).__init__()
        self.wq = nn.Linear(emb_dim, 1)
        self.wk = nn.Linear(emb_dim, 1)
        


    def forward(self, x): 
        qs = torch.mm(x, self.wq)


if __name__ == "__main__":
    batch_size = 6
    seq_len = 10
    emb_dim = 50 

    test_input = torch.randn(batch_size, seq_len, emb_dim)

    selfAttnLayer = SelfAttn()