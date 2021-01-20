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
    batch_size = 32
    bert_pretrain_weight="hfl/chinese-bert-wwm"
    max_len = 50

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.text_file_path)
        preprocessor = NerPreprocessor(x_train, y_train, bert_pretrain_weight="hfl/chinese-bert-wwm", max_len=50)
        features, labels =  preprocessor.prepare_input(preprocessor.train_data,
                                                       preprocessor.train_labels)
        num_class = preprocessor.num_class
        ner_dataset = NerDataset(features, labels)
        ner_dataloader = DataLoader(ner_dataset)

        model_builder = NerModel(num_class, bert_pretrain_weight=self.bert_pretrain_weight, max_len=self.max_len)
        model = NerModel.build_model() 

        self.ner_trainer = NerTrainer(model, preprocessor)

    def test_train(self):
        pass 


if __name__ == "__main__":
    test_nerTrainer = TestNerTrainer()
    test_nerTrainer.setup_class()
    test_nerTrainer.test_get_one()
    print("Done")