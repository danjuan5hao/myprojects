# -*- coding: utf-8 -*-
class Message:
    
    def __init__(self, utterance, history):
        self.utterance = utterance 
        self.histroy = history
        self.nlu_results = {}
         
    def set_features(self, name, value):
        pass 

    def get_features(self, name):
        pass


class NluRst:
    def __init__(self):
        self.domain = None 
        self.intent = None 
        self.slot = None 
        self.value = None 

