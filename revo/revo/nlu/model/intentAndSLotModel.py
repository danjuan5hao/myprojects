# -*- coding: utf-8 -*-
class IntentAndSlotModel:
    def __init__(self, ):
        # self.name = 
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


class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.lstm = 
        self.attn = 
        self.intent_clf = 
        self.seq_tagger = 
        
    def forward(self, x):
        x = self.lstm(x)
        x = self.attn(x)
        clf_x = self.intent_clf(x)
        seq_tag_x =self.seq_tagger(x)
        return clf_x, seq_tag_x
    



# class IntentAndSlotWrapper:

#     def process(utterance, domain_idx, ):
#         pass

#     @staticmethod
#     def get_full_path(path, name):
#         model_name = f"intent_slot_{name}.vec"
#         return os.path.join(path, model_name)

#     @classmethod
#     def load(cls, path, domain_names):
#         kwargs = {name: IntentAndSlotModel.load(cls.get_full_path(path, domain_names)) for name in domain_names}

    