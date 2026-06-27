# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from app.settings import load_settings
from app.kg.neo4j_client import KGClient
from app.llm.client import LLMClient

app = FastAPI(title="KBQA")
_settings = load_settings()
_kg = KGClient(_settings)
_llm = LLMClient(_settings)


class ChatIn(BaseModel):
    text: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok", "deps": {"neo4j": _kg.available, "llm": _llm.available}}


@app.post("/chat")
def chat(req: ChatIn):
    # 骨架阶段:占位回复,后续任务接入 controller
    return {"answer": "(骨架)收到:" + req.text, "path": "stub"}
