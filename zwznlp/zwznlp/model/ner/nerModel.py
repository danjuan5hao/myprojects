# -*- coding: utf-8 -*-
from typing import Optional, Union
import os, sys

import torch 
import torch.nn as nn
from torchcrf import CRF

from zwznlp.model.ner.basicNerModel import BasicNerModel
from zwznlp.layer.crf import CRFLayer 

class NerModel(BasicNerModel):
    """Bert model for NER. Support using CRF layer.
        We suggest you to train bert on machines with GPU cause it will be very slow to be trained with
        cpu.
    """

    def __init__(self, 
                 num_class: int,
                 max_len: int,
                 bert_pretrain_weight="hfl/chinese_wwm_pytorch",):
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
        super(NerModel, self).__init__(num_class = num_class,
                                       max_len=max_len,
                                       bert_pretrain_weight=bert_pretrain_weight)
        self.num_class = num_class

    def build_model(self) -> nn.Module:
        bert_embedding = self.build_embedding()
        model = NerTorchModel(bert_embedding, self.num_class)
        return model

class NerTorchModel(nn.Module):
    def __init__(self, bert_embedding, num_class):
        super(NerTorchModel, self).__init__()
        self.bert_embedding = bert_embedding
        self.crflayer = CRFLayer(num_class)
        
        # self.dense = nn.Linear(bert_dim, num_class)
        
    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None):
        inputs = {"input_ids": input_ids,
                  "attention_mask": attention_mask,
                  "token_type_ids": token_type_ids}
        outputs = self.bert_embedding(**inputs)[0]
        return outputs, self.crflayer.decode(outputs)
          


