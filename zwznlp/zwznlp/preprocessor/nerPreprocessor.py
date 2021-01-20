# -*- coding=utf-8 -*-
import pickle
from typing import List, Optional, Tuple, Dict, Any
import os,sys

from absl import logging
import jieba
import numpy as np
from transformers import BertTokenizer

from zwznlp.preprocessor.basicPreprocessor import BasicPreprocessor
from zwznlp.util.other import get_len_from_corpus, pad_sequences_2d, one_hot_label

class NerPreprocessor(BasicPreprocessor):
    """NER preprocessor, which is used to
    1) build all kinds of vocabulary (char, word , label) from training data;
    2) pre-trained embedding matrix using training corpus;
    3) prepare feature input for ner model;
    4) decode model predictions to tagging sequence.
    """

    def __init__(self,
                 train_data: List[List[str]],
                 train_labels: List[List[str]],
                 max_len: int = None, 
                 bert_pretrain_weight = "chinese_wwm_pytorch") -> None:
        """
        """
        super(NerPreprocessor, self).__init__(max_len)

        self.train_data = train_data
        self.train_labels = train_labels
        self.max_len = max_len

        self.bert_pretrain_weight = bert_pretrain_weight
        self.bert_tokenizer = BertTokenizer.from_pretrained(bert_pretrain_weight)

        # build label vocabulary
        self.label_vocab, self.id2label = self.build_label_vocab(self.train_labels)
        self.num_class = len(self.label_vocab)

        if self.max_len is None:
            self.max_len = get_len_from_corpus(self.train_data)
            self.max_len = min(self.max_len + 2, 512)

        return 
·
    def build_label_vocab(self,
                          labels: List[List[str]]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """Build label vocabulary.
        """
        label_count = {}
        for sequence in labels:
            for label in sequence:
                label_count[label] = label_count.get(label, 0) + 1

        # sorted by frequency, so that the label with the highest frequency will be given
        # id of 0, which is the default id for unknown labels
        sorted_label_count = sorted(label_count.items(), key=lambda x: x[1], reverse=True)
        sorted_label_count = dict(sorted_label_count)

        label_vocab = {}
        for label in sorted_label_count:
            label_vocab[label] = len(label_vocab)

        id2label = dict((idx, label) for label, idx in label_vocab.items())

        logging.info('Build label vocabulary finished, '
                     'vocabulary size: {}'.format(len(label_vocab)))
        return label_vocab, id2label

    def prepare_input(self,
                      texts: List[List[str]],
                      labels: Optional[List[List[str]]] = None) -> Tuple[np.ndarray, Any]:
        """Prepare input (features and labels) for NER model.
        Here we not only use character embeddings (or bert embeddings) as main input, but also
        support word embeddings and other hand-crafted features embeddings as additional input.

        Args:
            data: List of List of str. List of tokenized (in char level) texts for training,
                like ``[['我', '在', '上', '海', '上'， '学'], ...]``.
            labels: Optional List of List of str, can be None. The labels of train_data, usually in
            BIO or BIOES format, like ``[['O', 'O', 'B-LOC', 'I-LOC', 'O', 'O'], ...]``.

        Returns: Tuple:
            features: id matrix
            y: label id matrix only if labels is provided, otherwise None,

        """
        batch_bert_ids, batch_bert_seg_ids,  batch_attention_masks = [], [], []
        batch_label_ids = []
        for i, char_text in enumerate(texts):

            indices = self.bert_tokenizer.encode(text=''.join(char_text)) # , segments 
            text_len = len(indices)
            batch_bert_ids.append(indices)
                
            attention_mask = [1]*text_len+[0]*(self.max_len-text_len)
            batch_attention_masks.append(attention_mask)

            segments = [0]*text_len
            batch_bert_seg_ids.append(segments)

            if labels is not None:              
                label_str = [self.cls_token] + labels[i] + [self.cls_token]
                label_ids = [self.label_vocab.get(l, self.get_unk_label_id()) for l in label_str]
                label_ids = one_hot_label(label_ids, self.num_class)
                batch_label_ids.append(label_ids)
                

        features = []
        if self.use_char:
            features.append(self.pad_sequence(batch_char_ids))
        if self.use_bert:
            features.append(self.pad_sequence(batch_bert_ids))
            features.append(self.pad_sequence(batch_bert_seg_ids))
            features.append(self.pad_sequence(batch_attention_masks))

        if self.use_word:
            features.append(self.pad_sequence(batch_word_ids))

        if len(features) == 1:
            features = features[0]

        if not batch_label_ids:
            return features, None
        else:
            y = pad_sequences_2d(batch_label_ids, max_len_1=self.max_len, max_len_2=self.num_class,
                                 padding=self.padding_mode, truncating=self.truncating_mode)
            return features, y

    def get_word_ids(self, word_cut: List[str]) -> List[int]:
        """Given a word-level tokenized text, return the corresponding word ids in char-level
           sequence. We add the same word id to each character in the word.

        Args:
            word_cut: List of str, like ['我', '是'. '中国人']

        Returns: List of int, id sequence

        """
        word_ids = []
        for word in word_cut:
            for _ in word:
                word_ids.append(self.word_vocab.get(word, self.word_vocab[self.unk_token]))
        if self.use_bert:
            word_ids = [self.word_vocab[self.cls_token]] + word_ids + \
                       [self.word_vocab[self.seq_token]]
        return word_ids

    def label_decode(self,
                     pred_probs: np.ndarray,
                     lengths: Optional[List[int]] = None) -> List[List[str]]:
        """Decode model predictions to label strings

        Args:
            pred_probs: np.ndarray, shaped [num_samples, max_len, num_class], the ner model's
                predictions
            lengths: Optional List of int. Length of each sample;

        Returns：
            List of List of str, the tagging sequences of each sample.

        """
        pred_ids = np.argmax(pred_probs, axis=-1)
        pred_labels = [[self.id2label[label_id] for label_id in ids] for ids in pred_ids]
        if lengths is not None:
            pred_labels = [labels[:length] for labels, length in zip(pred_labels, lengths)]
        return pred_labels

    def get_unk_label_id(self):
        """return a default id for label that does not exist in the label vocab

        Returns: int

        """
        if 'O' in self.label_vocab:
            return self.label_vocab['O']
        elif 'o' in self.label_vocab:
            return self.label_vocab['o']
        else:
            return 0  # id of 0 is the label with the highest frequency

    def save(self, preprocessor_file: str):
        """Save preprocessor to disk

        Args:
            preprocessor_file: str, path to save preprocessor

        Returns:

        """
        pickle.dump(self, open(preprocessor_file, 'wb'))

    @classmethod
    def load(cls, preprocessor_file):
        """Load preprocessor from disk

        Args:
            preprocessor_file: str, path to load preprocessor.

        Returns:

        """
        p = pickle.load(open(preprocessor_file, 'rb'))
        p.load_word_dict()  # reload external word dict into jieba
        return p

class BertNerPreprocessor(BasicPreprocessor):
    def __init__(self,
                 train_data: List[List[str]],
                 train_labels: List[List[str]],
                 min_count: int = 2,
                 max_len: Optional[int] = None,
                 padding_mode: str = 'post',
                 truncating_mode: str = 'post') -> None: 
        super(BertNerPreprocessor, self).__init__()