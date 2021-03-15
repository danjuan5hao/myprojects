    # -*- coding:utf-8 -*- 
from revo.core.policy import Policy 

class RulePolicy(Policy):
    def __init__(self, domains):
        pass 

    def predict_next_action(self, dial_state):
        self.find_next_need_to_know_slot(dial_state, domains)
        return self.request_next_domain_and_slot()


    def request_next_domain_and_slot(self, dial_state, domains):
        return 

