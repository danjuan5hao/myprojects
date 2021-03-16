# -*- coding: utf-8 -*- 
import numpy as np 
from revo import config 
import torch 
import torch.nn as nn  
from torchtext.vocab import FastText

from revo.nlu.featurizer.featurizer import Featurizer 


TORCH_FASTTEXT_PATH = config.TORCH_FASTTEXT_PRETRAIN_PATH
TOKENIZER = config.CHINESE_TOKENIZER
LANGUAGE = config.TORCH_FASTTEXT_LANGUAGE

class Embedding:
    def __init__(self, tokenizer, stoi, itos, vectors, unk_mark, unk_id):
        # self.tokenizer = tokenizer 
        self.stoi = stoi 
        self.itos = itos 
        self.vectors = vectors
        self.unk_id = unk_id
        self.unk_mark = unk_mark # 没有<UNK>

        
    @classmethod
    def load(cls, path):
        raise NotImplementedError

    def tokenize(self, sentence, tokenizer=TOKENIZER):
        return tokenizer(sentence)

    def serialize(self, x):
        return [self.stoi.get(i, self.unk_id) for i in self.tokenize(x)]
         
    

class Word2VectEmbedding:
    """
    """
    @classmethod
    def load(cls, tokenizer, pretrain_path, unk_mark=None, unk_mark_idx=None, max_vocab_size=None):
        """"gensim w2v pretrain model default, file suffix should be .vec
        """
        if unk_mark_idx and max_vocab_size:
            assert unk_mark_idx < max_vocab_size, "截取字典时， 字典大小不能小于UNK的索引"

        gensim_model = gensim.models.KeyedVectors.load_word2vec_format(pretrain_path, binary=False)
        w2v = gensim_model.vectors  # numpy.array
        itos = gensim_model.index2word
        embedding_dim = w2v.shape[1]

        if max_vocab_size:
            w2v = gensim_model.vectors[:max_vocab_size-1]
            itos = gensim_model.index2word[:max_vocab_size-1]      

        if unk_mark == None:
            unk_mark = "<UNK>"
            unk_vector = np.random.randn(embedding_dim)  # TODO 初始化未知词向量的方式
            itos.append(unk_mark)
            w2v = np.stack([w2v, unk_vector])

        vocab_size = len(itos)
        stoi = {item: idx for idx,item in  enumerate(itos)}


        return 
                            
        # vectors = torch.from_numpy(w2v)[:max_vocab_size]
        
        
        # self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # self.embedding.from_pretrained(vectors, freeze=True)


class FasttextEmbedding(Embedding):
    """torchtext Fasttext pretrain model default
    """
    def __init__(self, tokenizer, stoi, itos, vectors, unk_mark, unk_mark_idx):
        super(FasttextEmbedding, self).__init__(tokenizer, stoi, itos, vectors, unk_mark, unk_mark_idx)
        self.embedding = nn.Embedding.from_pretrained(torch.from_numpy(vectors))
        
    @classmethod
    def load(cls, tokenizer=TOKENIZER, pretrain_path=TORCH_FASTTEXT_PATH, unk_mark=None, unk_mark_idx=None, max_vocab_size=None):
        "torchtext Fasttext pretrain model default"
        torchFasttext = FastText(language=LANGUAGE, cache=pretrain_path)
        itos = torchFasttext.itos
        vectors = torchFasttext.vectors.numpy()
        embedding_dim = vectors.shape[1] 

        if max_vocab_size:
            itos = itos[:max_vocab_size]
            vectors = vectors[:max_vocab_size]   

        if unk_mark == None:
            unk_mark = "<UNK>"
            unk_vector = np.random.randn(1, embedding_dim)  # TODO 初始化未知词向量的方式
            itos.append(unk_mark)
            vectors = np.vstack([vectors, unk_vector])
            unk_mark_idx = unk_vector.shape[0]

        vocab_size = len(itos)
        stoi = {item: idx for idx,item in  enumerate(itos)}
        return cls(tokenizer, stoi, itos, vectors, unk_mark, unk_mark_idx)
        

class MergeEmbedding:
    pass 