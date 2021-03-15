# -*- coding: utf-8 -*-
class ConvRecord:
    def __init__(self, ):
        self.reocrd_id = None 
        self.domain_name = None 
        self.slots = []

    def update_rocord(self, domain_name, slot, value):
        pass 

class DialState:
    """

    """
    def __init__(self):
        self.records = []
        pass

    def update(self, record_id, domain_name, intent, slot_and_value):
        pass

    
def initial_dial_state(domains=None):
    # TODO 目前不对domians做检查
    return DialState()