# -*- coding: utf-8 -*- 
from revo.config import DOMAIN_CONFIG

from revo.nlu.classifier.classifier import TextCnnClassifier
from revo.nlu.featurizer.embeddingFeaturizer import FasttextEmbedding

class DomainModel:
    def __init__(self, embedding, classifier, domains):
        self.embedding = embedding
        self.classifier = classifier
        
    def process(self, message):
        return message

    @classmethod
    def load(clf, config, ):
        embedding = 
        classifier = 
        return clf(embedding, classifier)
          

    