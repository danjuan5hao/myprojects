# -*- coding:utf-8 -*- 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from revo import config 


class DomainDataLoader:
    def __init__(self, stoi, tokenizer, domain_name, data_path_dir):
        
        dataset = DomainDataset(stoi, tokenizer, domain_name, data_path_dir)
        self.dataloader = DataLoader(dataset, batch_size=6)

class DomainDataset(Dataset):
    def __init__(self, stoi, tokenizer, domain_name, data_path_dir, max_len=20):
        super(DomainDataset, self).__init__()
        self.tokenizer = tokenizer
        self.stoi = stoi
        self.domain_names = domain_names
        self.max_len = max_len


        with open(data_path_dir, "r", encoding="utf-8") as f:
            text_and_labels = f.readlines()

        all_labels = []
        all_texts = []

        for t_and_l in text_and_labels:
            t, l = t_and_l.split() 
            token_id_seq = self._get_token_id_seq(t) 
            token_id_seq = self._pad_and_truncate(token_id_seq)
            all_labels.append(token_id_seq)
            label_id = self._get_domain_id() 
            all_labels.append(label_id)
        
        self.all_labels = torch.tensor(all_labels)
        self.all_texts = torch.tensor(all_texts)


    def _get_token_id_seq(self, text):
        return [self.stoi(i) for i in self.tokenizer(text)]


    def _get_domain_id(self, domain_name):
        # TODO 此处有误
        return self.domain_names.find(eval(domain_name)[0])

    def _pad_and_truncate(self, token_ids):
        token_ids = token_ids[:self.max_len] + [0]*(len(token_ids)-self.max_len)
        return token_ids


    def __len__(self):
        return self.all_labels.size(0)
        

    def __getitem__(self, idx):
        return  self.all_texts[idx], self.all_labels[idx]

