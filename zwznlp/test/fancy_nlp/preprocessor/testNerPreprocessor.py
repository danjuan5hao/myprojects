# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))

import jieba
import numpy as np

from zwznlp.util.data_loader import load_ner_data_and_labels
from zwznlp.preprocessor.nerPreprocessor import NerPreprocessor


class TestNERPreprocessor:
    test_file = os.path.join(os.path.dirname(__file__), '../../../testData/ner/msra/example.txt')
    bert_vocab_file = os.path.join(os.path.dirname(__file__),
                                   '../../../testData/embeddings/bert_sample_model/vocab.txt')

    def setup_class(self):
        x_train, y_train = load_ner_data_and_labels(self.test_file)
        self.preprocessor = NerPreprocessor(x_train, y_train, use_char=True, use_bert=True,
                                            use_word=True, bert_vocab_file=self.bert_vocab_file,
                                            external_word_dict=['比特币'],
                                            char_embed_type='word2vec', max_len=16)

    def test_init(self):
        assert len(self.preprocessor.char_vocab_count) + 4 == len(self.preprocessor.char_vocab) \
            == len(self.preprocessor.id2char)
        assert list(self.preprocessor.id2char.keys())[0] == 0
        for cnt in self.preprocessor.char_vocab_count.values():
            assert cnt >= 2

        assert self.preprocessor.char_embeddings.shape[0] == len(self.preprocessor.char_vocab)
        assert self.preprocessor.char_embeddings.shape[1] == 300
        assert not np.any(self.preprocessor.char_embeddings[0])

        assert len(self.preprocessor.word_vocab_count) + 4 == len(self.preprocessor.word_vocab) \
            == len(self.preprocessor.id2word)
        assert list(self.preprocessor.id2word.keys())[0] == 0
        for cnt in self.preprocessor.word_vocab_count.values():
            assert cnt >= 2
        assert self.preprocessor.word_embeddings is None

        assert len(self.preprocessor.label_vocab) == len(self.preprocessor.id2label)
        assert list(self.preprocessor.id2label.keys())[0] == 0

    def test_prepare_input(self):
  
        features, y = self.preprocessor.prepare_input(self.preprocessor.train_data,
                                                      self.preprocessor.train_labels)
        assert len(features) == 4
        assert features[0].shape == features[1].shape == features[2].shape == features[3].shape == \
               (len(self.preprocessor.train_data), self.preprocessor.max_len)
        assert self.preprocessor.id2char[features[0][0][0]] == self.preprocessor.cls_token
        assert self.preprocessor.id2word[features[0][0][0]] == self.preprocessor.cls_token
        assert y.shape == (len(self.preprocessor.train_data), self.preprocessor.max_len,
                           self.preprocessor.num_class)

    def test_get_word_ids(self):
        example_text = ''.join(self.preprocessor.train_data[0])
        word_cut = jieba.lcut(example_text)
        word_ids = self.preprocessor.get_word_ids(word_cut)
        assert len(word_ids) == len(example_text) + 2

        start = 1
        for word in word_cut:
            if start > len(word_ids):
                break
            assert len(set(word_ids[start:start+len(word)])) == 1
            start += len(word)

    def test_label_decode(self):
        rand_pred_probs = np.random.rand(2, 10, self.preprocessor.num_class)
        lengths = [8, 9]
        pred_labels = self.preprocessor.label_decode(rand_pred_probs, lengths)
        assert len(pred_labels) == len(lengths)
        for i, length in enumerate(lengths):
            assert len(pred_labels[i]) == length

    def test_save_load(self):
        pkl_file = 'test_preprocessor.pkl'
        self.preprocessor.save(pkl_file)
        assert os.path.exists(pkl_file)
        new_preprocessor = NerPreprocessor.load(pkl_file)
        assert new_preprocessor.num_class == self.preprocessor.num_class
        os.remove(pkl_file)


if __name__ == "__main__":
    a = TestNERPreprocessor()
    a.setup_class()
    a.test_prepare_input()
