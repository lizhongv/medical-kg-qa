# -*- coding: utf-8 -*-
import os
from dataclasses import dataclass


@dataclass
class Settings:
    accept_threshold: float = 0.8
    deny_threshold: float = 0.4
    neo4j_uri: str = "bolt://127.0.0.1:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "123456"
    redis_url: str | None = None
    llm_base_url: str | None = None
    llm_api_key: str | None = None
    llm_model: str = "gpt-4o-mini"


def load_settings(env: dict | None = None) -> Settings:
    e = os.environ if env is None else env

    def f(key, default):
        v = e.get(key)
        return float(v) if v is not None else default

    return Settings(
        accept_threshold=f("KBQA_ACCEPT_THRESHOLD", 0.8),
        deny_threshold=f("KBQA_DENY_THRESHOLD", 0.4),
        neo4j_uri=e.get("KBQA_NEO4J_URI", "bolt://127.0.0.1:7687"),
        neo4j_user=e.get("KBQA_NEO4J_USER", "neo4j"),
        neo4j_password=e.get("KBQA_NEO4J_PASSWORD", "123456"),
        redis_url=e.get("KBQA_REDIS_URL"),
        llm_base_url=e.get("KBQA_LLM_BASE_URL"),
        llm_api_key=e.get("KBQA_LLM_API_KEY"),
        llm_model=e.get("KBQA_LLM_MODEL", "gpt-4o-mini"),
    )
