# -*- coding: utf-8 -*-
import json


class Normalizer:
    def __init__(self, diseases_path):
        with open(diseases_path, encoding="utf-8") as f:
            self.names = set(json.load(f))

    def normalize(self, mention):
        if mention in self.names:
            return mention
        return None
