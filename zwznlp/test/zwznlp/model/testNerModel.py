# -*- coding: utf-8 -*-

import os,sys
ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "../../..")) 

from torch.utils.data import DataLoader

from zwznlp.preprocessor import NerPreprocessor
from zwznlp.util.data_loader import load_ner_data_and_labels
from zwznlp.dataLoader import NerDataset
from zwznlp.trainer import NerTrainer
from zwznlp.model.ner import NerModel

class TestNerModel:

    text_file_path = r"testData\ner\msra\example.txt"

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.text_file_path)
        preprocessor = NerPreprocessor(x_train, y_train, bert_pretrain_weight="hfl/chinese-bert-wwm", max_len=50)
        features, labels = preprocessor.prepare_input(preprocessor.train_data,
                                                       preprocessor.train_labels)
        self.ner_dataset = NerDataset(features, labels)
        self.ner_dataloader = DataLoader(self.ner_dataset)

    def test_ner_model(self):
        pass 
