# -*- coding: utf-8 -*-
class SemanticCache:
    """精确匹配缓存;预留 embedding 相似(无 embedding 依赖时退化为精确)。"""
    def __init__(self, threshold=0.95):
        self.threshold = threshold
        self._store = {}

    def lookup(self, question):
        return self._store.get(question.strip())

    def save(self, question, answer):
        self._store[question.strip()] = answer
