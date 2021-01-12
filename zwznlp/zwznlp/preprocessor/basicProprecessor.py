# -*- coding: utf-8 -*-
from typing import Callable, List, Tuple, Dict, Optional

import numpy as np 

class BasicPreprocessor:
    """
    preprocessor should help dataset to output texts and labels in tensor form. 
    and dataset/dataloader can use 
    
    Preprocessor is used to
    1) build vocabulary from training data;
    2) pre-trained embedding matrix using training corpus;
    3) prepare feature input for model;
    4) decode model predictions to label string
    """

    def __init__(self, texts, labels):
        pass

    def build_vocab(self, texts):
        pass 

    def build_embedding(self):
        pass 

    def build_label_vocab(self, labels):
        pass 

    def prepare_input(self, data, label=None):
        """Prepare feature input for neural model training, evaluating and testing.
        """
        raise NotImplementedError 

    @staticmethod
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
                        vocabulary: Dict[str, int],
                        unk_idx: int = 1) -> List[List[int]]:
        return [[vocabulary.get(token, unk_idx) for token in text] for text in tokenized_texts] 

    def pad_sequence(self, sequence_list: List[List[int]]) -> np.ndarray:
        """Given a list, each item is a id sequence, return the padded sequence.

        Args:
            sequence_list: List of List of int, where each element is a sequence.

        Returns:
            a 2D Numpy array of shape `(num_samples, num_timesteps)`

        """
        pass 

    def label_decode(self, predictions, label_dict):
        """Decode model predictions to label strings
        """
        raise NotImplementedError

    

