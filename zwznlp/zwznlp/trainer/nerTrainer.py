# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import os
from typing import List, Optional

import torch
import torch.nn as nn
from absl import logging

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.utils.data import DataLoader
import torch.nn as nn 
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    BertConfig,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertTokenizer,
    get_linear_schedule_with_warmup,
)

from zwznlp.preprocessor.nerPreprocessor import NerPreprocessor
from zwznlp.dataLoader.nerDataset import NerDataset
from zwznlp.loss.crf import CRFloss

class  NerTrainer:
    """NER Trainer, which is used to
    1) train ner model with given training dataset
    2) evaluate ner model with given validation dataset

    """

    def __init__(self,
                 model, #: nn.Module
                 preprocessor: NerPreprocessor) -> None:
        """

        Args:
            model: Instance of tf.keras Model. The ner model to be trained.
            preprocessor: Instance of NERPreprocessor, which helps to prepare feature input for
                ner model.
        """
        self.model = model
        self.preprocessor = preprocessor

    def fit(self,
            train_dataloader,
            num_class, 
            dev_dataloader = None,
            epochs: int = 50) -> None:
        """Train ner model with provided training dataset. If validation dataset is provided,
        evaluate ner model with it after training.
        """

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": 0.0,},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},]
        
        optimizer = AdamW(optimizer_grouped_parameters)
        criterion = CRFloss(num_class)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        for _ in range(epochs):
            for _, batch in enumerate(train_dataloader):
                self.model.train() 
                # inputs  = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2]}
                inputs  = {"input_ids": batch[0],  "token_type_ids":batch[2]}

                outputs = self.model(**inputs)
                llk = criterion(outputs, batch[3])  # TODO mask
                loss = -llk
                loss.backward()
                # tr_loss += loss.item()
                optimizer.step()
                self.model.zero_grad()
                global_step += 1
            print(loss.item())

                # print(tr_loss)

                    


    
