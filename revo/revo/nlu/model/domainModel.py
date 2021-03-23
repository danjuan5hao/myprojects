# -*- coding: utf-8 -*- 
# from revo.config import NLU_DOMAIN_CLF_CONFIG

from revo.nlu.classifier.classifier import Classifier
from revo.nlu.featurizer.embeddingFeaturizer import Embedding
from revo.config import FeaturizerConfig, DomainConfig 

class DomainModel:
    def __init__(self, embedding, classifier):
        self.embedding = embedding
        self.classifier = classifier
        
    def process(self, message):
        return message

    @classmethod
    def load(cls):
        embedding_name = FeaturizerConfig.NAME 
        embedding= Embedding.load(embedding_name)
        embedding_dim = embedding.dim

        num_domains = DomainConfig.DOMAIN_NUMS
        classifier_name = DomainConfig.DOMAIN_CLF_NAME
        path = DomainConfig.DOMAIN_CLF_MODEL_DIR

        clf = Classifier.load(classifier_name, embedding_dim, num_domains, path)

        return cls(embedding, clf)
          

    