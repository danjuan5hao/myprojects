# -*- coding: utf-8 -*-
from typing import Callable, List, Tuple, Dict, Optional

from absl import logging
import numpy as np 
from torch.nn.utils.rnn import pad_sequence

from zwznlp.util.embedding import load_pre_trained, train_w2v, train_fasttext

class BasicPreprocessor:
    """Basic class for Fancy-NLP Preprocessor. All the preprocessor will inherit from it.

    Preprocessor is used to
    1) build vocabulary from training data;
    2) pre-trained embedding matrix using training corpus;
    3) prepare feature input for model;
    4) decode model predictions to label string

    """

    def __init__(self,
                 max_len: int = None) -> None:
        """
        """
        self.max_len = max_len
        self.cls_token = "<CLS>"


    @staticmethod
    def build_corpus(untokenized_texts: List[str],
                     cut_func: Callable[[str], List[str]]) -> List[List[str]]:
        """Build corpus from untokenized texts.
        """
        corpus = []
        for text in untokenized_texts:
            corpus.append(cut_func(text))
        return corpus 
    
    
    def build_label_vocab(self, labels):
        """Build label vocabulary.
        """
        raise NotImplementedError

    
    def prepare_input(self, data, label=None):
        """Prepare feature input for neural model training, evaluating and testing.
        """
        raise NotImplementedError

    @staticmethod  # TODO 为什么用staticmethod
    def build_id_sequence(tokenized_text: List[str],
                          vocabulary: Dict[str, int],
                          unk_idx: int = 1) -> List[int]:
        """Given a token list, return the corresponding id sequence.

        Args:
            tokenized_text: List of str, like `['我', '是', '中', '国', '人']`.
            vocabulary: Dict[str, int]. A mapping of tokens to indices.
            unk_idx: int. The index of tokens that do not appear in vocabulary. We usually set it
                to 1.

        Returns:
            List of indices.

        """
        return [vocabulary.get(token, unk_idx) for token in tokenized_text]

    @staticmethod
    def build_id_matrix(tokenized_texts: List[List[str]],
                        vocabulary, unk_idx=1):
        """Given a list, each item is a token list, return the corresponding id matrix.

        Args:
            tokenized_texts: List of List of str. List of tokenized texts, like ``[['我', '是', '中',
                '国', '人'], ...]``.
            vocabulary: Dict[str, int]. A mapping of tokens to indices
            unk_idx: int. The index of tokens that do not appear in vocabulary. We usually set it
                to 1.

        Returns:
            List of List of indices

        """
        return [[vocabulary.get(token, unk_idx) for token in text] for text in tokenized_texts]

    def pad_sequence(self,
                     sequence_list: List[List[int]]) -> np.ndarray:
        """Given a list, each item is a id sequence, return the padded sequence.

        Args:
            sequence_list: List of List of int, where each element is a sequence.

        Returns:
            a 2D Numpy array of shape `(num_samples, num_timesteps)`

        """
        padded_trunct_matrix = np.zeros((len(sequence_list), self.max_len), dtype=int)
        for i, sequence in enumerate(sequence_list):
            sequence = sequence[:self.max_len] 
            padded_trunct_matrix[i, :self.max_len] = sequence
        return padded_trunct_matrix

    def label_decode(self, predictions, label_dict):
        """Decode model predictions to label strings
        """
        raise NotImplementedError

    

