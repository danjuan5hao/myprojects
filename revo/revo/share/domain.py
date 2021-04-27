# -*- coding: utf-8 -*-


class Domain:
    """
    Domain define part of the universe what bot should know about and how to react.
    
    """
    def __init__(self, name, **kwargs):
        self.name = name 
        self.intents = kwargs.get('intents', [])
        self.slots = kwargs.get("slots", [])

    @classmethod
    def load_from_file(cls, f):
        pass 

    def parse_domain_file(self, f):
        pass 
    






    