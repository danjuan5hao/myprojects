from typing import Optional

import numpy as np
import torch.nn as nn
from transformers import BertForTokenClassification, BertForPreTraining, BertConfig

from zwznlp.model.basicModel import BasicModel

class BasicNerModel(BasicModel):
    """The basic class for ner models. All the ner models will inherit from it.

    """
    def __init__(self,
                 num_class,
                 bert_pretrain_weight: str = "hfl/chinese_wwm_pytorch",
                 max_len: Optional[int] = None) -> None:
        """
        Args:
            use_char: Boolean. Whether to use character embedding as input.
            char_embeddings: Optional np.ndarray. Char embedding matrix, shaped
                [char_vocab_size, char_embed_dim]. There are 2 cases when char_embeddings is None:
                1)  use_char is False, do not use char embedding as input; 2) user did not
                provide valid pre-trained embedding file or any embedding training method. In
                this case, use randomly initialized embedding instead.
            char_vocab_size: int. The size of char vocabulary.
            char_embed_dim: int. Dimensionality of char embedding.
            char_embed_trainable: Boolean. Whether to update char embedding during training.
            use_bert: Boolean. Whether to use bert embedding as input.
            bert_config_file: Optional str, can be None. Path to bert's configuration file.
            bert_checkpoint_file: Optional str, can be None. Path to bert's checkpoint file.
            bert_trainable: Boolean. Whether to update bert during training.
            use_word: Boolean. Whether to use word as additional input.
            word_embeddings: Optional np.ndarray. Similar as char_embeddings.
            word_vocab_size: int. Similar as char_vocab_size.
            word_embed_dim: int. Similar as char_embed_dim.
            word_embed_trainable: Boolean. Similar as char_embed_trainable.
            max_len: Optional int, can be None. Max length of one sequence.
            dropout: float. The dropout rate applied to embedding layer.
        """
        super(BasicNerModel, self).__init__()
        self.bert_pretrain_weight = bert_pretrain_weight 
        self.max_len = max_len
        self.num_class = num_class

    def build_embedding(self): 
        """Build input placeholder and prepare embedding for ner model
        """
       
        config = BertConfig.from_pretrained(self.bert_pretrain_weight,
                                            num_labels=self.num_class)
        bert_model = BertForTokenClassification.from_pretrained(self.bert_pretrain_weight, config=config) 
        # bert_model = BertForPreTraining.from_pretrained(self.bert_pretrain_weight, config=config) 

        return bert_model

    def build_model(self):
        """Build ner model's architecture."""
        raise NotImplementedError