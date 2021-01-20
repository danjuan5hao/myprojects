import os,sys
ROOT = os.path.dirname(__file__)
sys.path.append(os.path.join(ROOT, "../../..")) 

from zwznlp.preprocessor import NerPreprocessor
from zwznlp.util.data_loader import load_ner_data_and_labels

class TestNerPreprocessor:
    text_file_path = r"testData\ner\msra\example.txt"

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.text_file_path)
        self.preprocessor = NerPreprocessor(x_train, y_train, bert_pretrain_weight="hfl/chinese-bert-wwm", max_len=50)

    def test_prepare_input(self):
        features, labels = self.preprocessor.prepare_input(self.preprocessor.train_data,
                                                           self.preprocessor.train_labels)
        print(len(features))
        print(len(labels))
        
        pass 

if __name__ == "__main__":
    test_ner_preprocessor = TestNerPreprocessor()
    test_ner_preprocessor.setup_class()
    test_ner_preprocessor.test_prepare_input()

    print("DONE")