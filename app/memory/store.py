# -*- coding: utf-8 -*-
import copy
import json

_DEFAULT = {"slots": {}, "history": [], "last_intent": None}


class MemoryStore:
    def __init__(self, settings):
        self._redis = None
        self._mem = {}
        if settings.redis_url:
            try:
                import redis
                self._redis = redis.Redis.from_url(settings.redis_url, decode_responses=True)
                self._redis.ping()
            except Exception:
                self._redis = None

    def get(self, session_id):
        if self._redis:
            raw = self._redis.get(f"kbqa:{session_id}")
            return json.loads(raw) if raw else {"slots": {}, "history": [], "last_intent": None}
        return self._mem.get(session_id, {"slots": {}, "history": [], "last_intent": None})

    def set(self, session_id, state):
        if self._redis:
            self._redis.set(f"kbqa:{session_id}", json.dumps(state, ensure_ascii=False))
        else:
            self._mem[session_id] = copy.deepcopy(state)
