# -*- coding: utf-8 -*-
from typing import Optional, Union
import os, sys

import torch 
import torch.nn as nn
from zwznlp.model.ner.basicNerModel import BasicNerModel


class NerModel(BasicNerModel):
    """Bert model for NER. Support using CRF layer.
        We suggest you to train bert on machines with GPU cause it will be very slow to be trained with
        cpu.
    """

    def __init__(self,
                 num_class: int,
                 bert_pretrain_weight="hfl/chinese_wwm_pytorch",
                 max_len: int):
        """
        Args:
            num_class: int. Number of entity type.
            bert_config_file: str. Path to bert's configuration file.
            bert_checkpoint_file: str. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            max_len: Optional int, can be None. Max length of one sequence.
            dropout: float. The dropout rate applied to embedding layer.
            fc_dim: int. Dimensionality of fully-connected layer.
            activation: str. Activation function to use in fully-connected layer.
            use_crf: Boolean. Whether to use crf layer.
            optimizer: str or instance of `tf.keras.optimizers.Optimizer`. Which optimizer to
                use during training.
            **kwargs:
        """
        # bert
        self.num_class = num_class
        self.max_len = max_len
        super(NerModel, self).__init__()

    def build_model(self) -> nn.Module:
        embeddings = self.build_embedding()
        bert_embedding = embeddings.get("bert_emb")
        model = BertNerTorchModel(bert_embedding, self.max_len, self.num_class, self.bert_dim)
        return model

class BertNerTorchModel(nn.Module):
    def __init__(self, bert_embedding, max_len, num_class, bert_dim):
        super(BertNerTorchModel, self).__init__()
        self.bert_embeding = bert_embedding
        self.dense = nn.Linear(bert_dim, num_class)
        
    def forward(self, xs):
        xs = self.bert_embedding(xs)
        return  self.dense(xs)


