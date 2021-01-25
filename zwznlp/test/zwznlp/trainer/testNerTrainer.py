# -*- coding: utf-8 -*-

import os,sys
ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "../../..")) 

from torch.utils.data import DataLoader

from zwznlp.preprocessor import NerPreprocessor
from zwznlp.util.data_loader import load_ner_data_and_labels
from zwznlp.dataLoader import NerDataset
from zwznlp.trainer import NerTrainer
from zwznlp.model.ner.nerModel import NerModel

class TestNerTrainer:

    text_file_path = r"testData\ner\msra\example.txt"
    batch_size = 11
    bert_pretrain_weight="hfl/chinese-bert-wwm"
    max_len = 50

    def setup_class(self):
        self.x_train, self.y_train = load_ner_data_and_labels(self.text_file_path)
        self.preprocessor = NerPreprocessor(self.x_train, self.y_train, bert_pretrain_weight="hfl/chinese-bert-wwm", max_len=50)
        # features, labels =  self.preprocessor.prepare_input(preprocessor.train_data,
        #                                                preprocessor.train_labels)
        self.num_class = self.preprocessor.num_class
        # ner_dataset = NerDataset(features, labels)
        # self.ner_dataloader = DataLoader(ner_dataset, batch_size=self.batch_size)

        model_builder = NerModel(self.num_class, bert_pretrain_weight=self.bert_pretrain_weight, max_len=self.max_len)
        model = model_builder.build_model() 

        self.ner_trainer = NerTrainer(model, self.preprocessor)

    def test_train_fit(self):
        # self.ner_trainer.fit(self.ner_dataloader, self.num_class)
        self.ner_trainer.fit(self.x_train, self.y_train, self.preprocessor,  self.num_class)

if __name__ == "__main__":
    test_nerTrainer = TestNerTrainer()
    test_nerTrainer.setup_class()
    test_nerTrainer.test_train_fit()
    print("Done")