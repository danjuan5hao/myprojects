# -*- coding: utf-8 -*-
class IntentAndSlotModel:
    def __init__(self, ):
        self.name = 
        self.embedding = 
        self.model = 

    def process(self, x):
        return self.model(x) 

    @classmethod
    def load(clf, name, path, config):
        name = 
        embedding = 
        model = 
        return clf(name, embedding, model)


class IntentAndSlotWrapper:

    def process(utterance, domain_idx, ):
        pass

    @staticmethod
    def get_full_path(path, name):
        model_name = f"intent_slot_{name}.vec"
        return os.path.join(path, model_name)

    @classmethod
    def load(cls, path, domain_names):
        kwargs = {name: IntentAndSlotModel.load(cls.get_full_path(path, domain_names)) for name in domain_names}

    