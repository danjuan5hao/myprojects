# -*- coding: utf-8 -*-
class TransformerFeaturizer:
    """Featurizer using transformer-based language models.

    The transformers(https://github.com/huggingface/transformers) library
    is used to load pre-trained language models like BERT, GPT-2, etc.
    The component also tokenizes and featurizes dense featurizable attributes of
    each message.
    """

    defaults = {
        # name of the language model to load.
        "model_name": "bert",
        # Pre-Trained weights to be loaded(string)
        "model_weights": None,
        # an optional path to a specific directory to download
        # and cache the pre-trained model weights.
        "cache_dir": None,
    }


    def train