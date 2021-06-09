# -*- coding: utf-8 -*- 
import torch 
from torch.utils.data import Dataset, DataLoader

from transformers import AutoTokenizer

class TxtClsfyDataset(Dataset):
    def __init__(self, texts, labels, pretrained_weight, max_length):
        super(TxtClsfyDataset, self).__init__()
        self.texts = texts 
        self.labels = labels 
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_weight) 
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        return self._prepare_text(text) +  (self._prepare_label(label), )

    def _prepare_text(self, text):
        inputs = self.tokenizer(text, max_length=self.max_length, padding="max_length",  add_special_tokens=False, truncation=True)
        
        input_ids =  torch.tensor(inputs["input_ids"], dtype=torch.long)
        token_type_ids = torch.tensor(inputs["token_type_ids"], dtype=torch.long)
        attention_mask = torch.tensor(inputs["attention_mask"], dtype=torch.long)
        # print(f"input_ids {input_ids.size()}")
        return input_ids, token_type_ids, attention_mask

    def _prepare_label(self, label):
        # label = self._padding_and_truncation(label)
        label =  torch.tensor(int(label), dtype=torch.long)
        # print(f"label {label.size()}")
        return label

if __name__ == "__main__":
    pretrained_weight_path = r"D:\data\预训练模型\chinese-bert-wwm"

    data_path = r"applications\littleFund\littleFund\littleNLP\data\sample_dataset.txt"
    with open(data_path, "r", encoding='utf-8') as f:
        lines = [line.strip() for line in f]
        texts, labels = zip(*[line.split("\t") for line in lines])
    
    dataset = TxtClsfyDataset(texts, labels, pretrained_weight_path, max_length=60)
    dataloader = DataLoader(dataset, batch_size=2)

    
        

