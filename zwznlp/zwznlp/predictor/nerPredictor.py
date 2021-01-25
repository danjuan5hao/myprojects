# -*- coding: utf-8 -*-

from collections import defaultdict
from typing import Union, List, Dict, Any

import numpy as np
from absl import logging
import torch 
from seqeval.metrics import sequence_labeling
from zwznlp.preprocessor import NerPreprocessor
from zwznlp.loss.crf import CRF


class NERPredictor(object):
    """NER predictor, which is used to
    1) output predictive probability sequence for given text;
    2) output predictive tag sequence for given text;
    3) output recognized entities with detailed information in pretty format for given text.

    """

    def __init__(self,
                 model,
                 preprocessor: NerPreprocessor) -> None:
        """
        Args:
            model: instance of tf.keras model, the trained ner model.
            preprocessor: instance of `NERPreprocessor`, which helps to prepare feature input for
                ner model.
        """
        self.model = model
        self.preprocessor = preprocessor

    def tag(self, text):
        features, _ = self.preprocessor.prepare_input([text])
        inputs = 
        _, outputs = self.model()
        
        pass 

    def tag_batch(self, texts):
        pass 
