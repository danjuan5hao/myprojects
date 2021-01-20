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
            dev_dataloader,
            batch_size: int = 32,
            epochs: int = 50) -> None:
        """Train ner model with provided training dataset. If validation dataset is provided,
        evaluate ner model with it after training.

        Args:
            train_data: List of List of str. List of tokenized (in char level) texts for training,
                like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            train_labels: List of List of str. The labels of train_data, usually in BIO or BIOES
                format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.
            valid_data: Optional List of List of str, can be None. List of tokenized (in char
                level) texts for evaluation.
            valid_labels: Optional List of List of str, can be None. The labels of valid_data.
                We can use fancy_nlp.utils.load_ner_data_and_labels() function to get training
                or validation data and labels from raw dataset in CoNLL format.
            batch_size: int. Number of samples per gradient update.
            epochs: int. Number of epochs to train the model
            callback_list: Optional List of str or instance of `keras.callbacks.Callback`,
                can be None. Each item indicates the callback to apply during training. Currently,
                we support using 'modelcheckpoint' for `ModelCheckpoint` callback, 'earlystopping`
                for 'Earlystopping` callback, 'swa' for 'SWA' callback. We will automatically add
                `NERMetric` callback when valid_data and valid_labels are both provided.
            checkpoint_dir: Optional str, can be None. Directory to save the ner model. It must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training.
            model_name: Optional str, can be None. Prefix of ner model's weights file. I must be
                provided when using `ModelCheckpoint` or `SWA` callback, since these callbacks needs
                to save ner model after training. For example, if checkpoint_dir is 'ckpt' and
                model_name is 'model', the weights of ner model saved by `ModelCheckpoint` callback
                will be 'ckpt/model.hdf5' and by `SWA` callback will be 'ckpt/model_swa.hdf5'.
            swa_model: Instance of `tf.keras.model.Model`. The ner model which is used in `SWA`
                callback to keep track of weight averaging during training. It has the same architecture as
                self.model. Only pass it when using `SWA` callback.
            load_swa_model: Boolean. Whether to load swa model, only apply when using `SWA`
                Callback. We suggest set it to True when using `SWA` Callback since swa model
                performs better than the original model at most cases.

        """

        train_features, train_y = self.preprocessor.prepare_input(train_data, train_labels)
        dataset = NerDataset(train_features, train_y)
        dataloader = DataLoader(dataset)

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {"params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],"weight_decay": 0.0,},
            {"params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], "weight_decay": 0.0},]
        
        optimizer = AdamW(optimizer_grouped_parameters)# AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        for _ in range(60):
            for _, batch in enumerate(dataloader):
                self.model.train() 
                inputs  = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}

                loss.backward()
                tr_loss += loss.item()

                optimizer.step()
                  
                self.model.zero_grad()
                global_step += 1

                    


    
