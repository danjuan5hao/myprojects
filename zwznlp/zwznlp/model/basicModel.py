# -*- coding: utf-8 -*-

class BasicModel(object):
    def build_embedding(self):
        """We build input and embedding layer for torch model here"""
        raise NotImplementedError

    def build_model(self):
        """We build torch model here"""
        raise NotImplementedError
