# -*- coding: utf-8 -*-

import os,sys
ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "../../..")) 

from zwznlp.preprocessor import NerPreprocessor
from zwznlp.util.data_loader import load_ner_data_and_labels
from zwznlp.dataLoader import NerDataset

class TestNerDataLoader:
    text_file_path = r"testData\ner\msra\example.txt"

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.text_file_path)
        preprocessor = NerPreprocessor(x_train, y_train, bert_pretrain_weight="hfl/chinese-bert-wwm", max_len=50)
        features, labels =  preprocessor.prepare_input(preprocessor.train_data,
                                                            preprocessor.train_labels)
        self.ner_dataset = NerDataset(features, labels)

    def test_get_one(self):
        idx = 1
        a = self.ner_dataset[idx]
        print(a[0])
        print(a[1])
        print(a[2])
        print(a[3])


if __name__ == "__main__":
    test_nerTrainer = TestNerDataLoader()
    test_nerTrainer.setup_class()
    test_nerTrainer.test_get_one()
    print("Done")