# -*- coding: utf-8 -*-
# -*- coding: utf-8 -*-

import os
from typing import List, Optional

import torch
import torch.nn as nn
from absl import logging

from sklearn.metrics import f1_score

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
from zwznlp.loss.crf import crfLoss

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
            train_texts,
            train_labels, 
            num_class,  # 这一项放在哪里比较好
            dev_texts = None,
            dev_labels = None,
            batch_size = 32,
            epochs: int = 50) -> None:

        """Train ner model with provided training dataset. If validation dataset is provided,
        evaluate ner model with it after training.
        """
        features, labels =  self.preprocessor.prepare_input(train_texts, train_labels)
        trainNerDataLoader = DataLoader(NerDataset(features, labels), batch_size=batch_size)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": 0.0,},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},]
        
        optimizer = AdamW(optimizer_grouped_parameters)
        # criterion = CRFloss(num_class)
        criterion = crfLoss(self.model)

        self.model.zero_grad()

        print('Training start...')
        for _ in range(epochs):
            for _, batch in enumerate(trainNerDataLoader):
                self.model.train() 
                # inputs  = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2]}
                inputs  = {"input_ids": batch[0],  "token_type_ids":batch[2]}
                tags = batch[3]

                emission, _ = self.model(**inputs)
                llk = criterion(emission, tags)  # TODO 在loss中添加mask, 并且mask的形式需要和CRFloss匹配
                loss = -llk
                loss.backward()
                optimizer.step()
                self.model.zero_grad()
                # global_step += 1

            if dev_texts and dev_labels:
                self.evaluate(dev_texts, dev_labels)
                # dev_dataLoader = DataLoader(dev_dataset, batch_size=len(dev_dataset))
                pass 
        print('Training end...')

    def evaluate(self, dev_texts, dev_labels):
        features, labels =  self.preprocessor.prepare_input(dev_texts, dev_labels)
        devNerDataLoader = DataLoader(NerDataset(features, labels), batch_size=2)
        self.model.eval()
        all_preds = []
        all_trues = []
        for batch in devNerDataLoader:
            inputs  = {"input_ids": batch[0],  "token_type_ids":batch[2]}
            tags = batch[3]
            _, outputs = self.model(**inputs)
            
            # TODO 需要把结果对比以后添加到列表
        return f1_score(all_trues, all_preds)

       


    def save(self, saving_path=None):
        torch.save(self.model.state_dict(),  saving_path)
         


                    


    
