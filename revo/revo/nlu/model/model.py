# -*- coding: utf-8 -*- 
from revo.nlu.model.domainModel import DomainModel
from revo.nlu.model.intentAndSLotModel import IntentAndSlotWrapper
# from revo.nlu.featurizer.embeddingFeaturizer import FasttextEmbedding 
from revo.config import DomainConfig, FeaturizerConfig


class NLUModel:
    def __init__(self, domain_clf):
        self.domain_clf = domain_clf 
        # self.intent_slot_wrapper = intent_slot_wrapper


    def process(self, message):

        domain_name = self.domain_clf.process(message)
        intent_and_slots = self.intent_slot_wrapper.process(message, domain_name) 

        return message

    # def tokenize(self, x):
    #     rst = self.embedding.serialize(x)
    #     return rst
    
    @staticmethod
    def load_intent_slot_wrapper(config):
        pass 
    
    @staticmethod
    def load_domain_clf(config):
        pass 
         

    @classmethod
    def load(cls):
        domain_clf = DomainModel.load()
        # intent_slot_wrapper = 
        return cls(domain_clf)

     
    
        

    