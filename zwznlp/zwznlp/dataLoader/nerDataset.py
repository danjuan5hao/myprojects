# -*- coding: utf-8 -*-
import torch 
from torch.utils.data import Dataset

class NerDataset(Dataset):
    def __init__(self, features, labels):
        super(NerDataset, self).__init__()
        self.all_input_ids = torch.from_numpy(features[0]).type(torch.long)
        self.all_attention_mask = torch.from_numpy(features[1]).type(torch.long)
        self.all_token_type_ids = torch.from_numpy(features[2]).type(torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.all_input_ids[idx], self.all_attention_mask[idx], self.all_token_type_ids[idx], self.labels[idx]