# -*- coding: utf-8 -*- 
import torch  
import torch.nn as nn  
import torch.nn.functional as F 

class TextCnn(nn.Module):
    def __init__(self, bert_embedding, bert_embedding_dim, num_classes):
        super(TextCnn, self).__init__()
        self.bert_embedding = bert_embedding
        self.convs = nn.ModuleList(
            [nn.Conv1d(bert_embedding_dim, 2, k) for k in[3,4,5]])
        self.dropout = nn.Dropout()
        self.fc = nn.Linear(6, num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x))
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_ids, token_type_ids, attention_mask, label=None):
        out = self.bertembedding(input_ids, token_type_ids, attention_mask)[0]
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out) 

        if label is None:
            return F.softmax(out)
        else:
            lsfnc=nn.CrossEntropyLoss()
            return lsfnc(out, label), F.softmax(out) 
sel