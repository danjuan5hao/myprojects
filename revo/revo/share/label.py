# -*- coding: utf-8 -*-
import json

class Label:
    def __init__(self, name):
        self.name = name

class Intent(Label):
    def __init__(self, name):
        super(Intent, self).__init__(name)

    def __repr__(self) :
        return f"IntentLable: {self.name}"

class Slot(Label):
    def __init__(self, name, value=None):
        super(Slot, self).__init__(name)
        self.value = value

    def to_dict(self):
        return {self.name: self.value}
    
    def __repr__(self):
        return f"SlotLable: {json.dumps(self.to_dict())}"