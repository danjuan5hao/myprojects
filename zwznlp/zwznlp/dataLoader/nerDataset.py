# -*- coding: utf-8 -*-
import torch 
from torch.utils.data import Dataset

class BertNerDataset(Dataset):
    def __init__(self, features, labels):
        self.all_input_ids = torch.tensor(features[0], dtype=torch.long)
        self.all_attention_mask = torch.tensor(features[1], dtype=torch.long)
        self.all_token_type_ids = torch.tensor(features[2], dtype=torch.long)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.self.all_input_ids[idx], self.all_attention_mask[idx], self.all_token_type_ids[idx], self.labels[idx]