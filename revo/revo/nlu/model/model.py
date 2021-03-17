# -*- coding: utf-8 -*- 
from revo.nlu.model.domainModel import DomainModel
from revo.nlu.model.intentAndSLotModel import IntentAndSlotWrapper
# from revo.nlu.featurizer.embeddingFeaturizer import FasttextEmbedding 



class NLUModel:
    def __init__(self, domain_clf, intent_slot_wrapper):
        super(NLUModel, self).__init__()
        self.domain_clf = domain_clf 
        self.intent_slot_wrapper = intent_slot_wrapper


    def process(self, message):

        domain_name = self.domain_clf.process(message)
        intent_and_slots = self.intent_slot_wrapper.process(message, domain_idx)
        return 

    # def tokenize(self, x):
    #     rst = self.embedding.serialize(x)
    #     return rst

    def forward(self, x):
        pass
    
    @staticmethod
    def load_intent_slot_wrapper(config, domain_names):
         


    @classmethod
    def load(cls, config): 
        intent_slot_wrapper = 

     
    
        

    