# -*- coding: utf-8 -*-
import json


class Normalizer:
    def __init__(self, diseases_path):
        self.names = set(json.load(open(diseases_path, encoding="utf-8")))

    def normalize(self, mention):
        if mention in self.names:
            return mention
        return None
