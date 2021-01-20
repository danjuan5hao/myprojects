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

class TestNerModel:

    text_file_path = r"testData\ner\msra\example.txt"
    batch_size = 32
    bert_pretrain_weight="hfl/chinese-bert-wwm"
    max_len = 50

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.text_file_path)
        preprocessor = NerPreprocessor(x_train, y_train, bert_pretrain_weight=self.bert_pretrain_weight, max_len=self.max_len)
        features, labels = preprocessor.prepare_input(preprocessor.train_data,
                                                      preprocessor.train_labels)
        num_class = preprocessor.num_class
        self.ner_dataset = NerDataset(features, labels)
        self.ner_dataloader = DataLoader(self.ner_dataset, batch_size=self.batch_size)

        self.model_builder = NerModel(num_class, bert_pretrain_weight=self.bert_pretrain_weight, max_len=self.max_len)

    def test_ner_model_build_embedding(self):
        bert_embedding = self.model_builder.build_embedding()
        for batch in  self.ner_dataloader:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2]}
            # inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2], "labels": batch[3]}

            outputs = bert_embedding(**inputs)[0]
            print(outputs.size())
    
    def test_ner_model_build_model(self):
        model = self.model_builder.build_model()
        for batch in  self.ner_dataloader:
            inputs = {"input_ids": batch[0], "attention_mask": batch[1], "token_type_ids":batch[2]}
            outputs = model(**inputs)
            print(outputs.size())

if __name__ == "__main__":
    test_ner_model = TestNerModel()
    test_ner_model.setup_class()
    # test_ner_model.test_ner_model_build_embedding()
    test_ner_model.test_ner_model_build_model()