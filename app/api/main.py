# -*- coding: utf-8 -*-
from fastapi import FastAPI
from pydantic import BaseModel
from app.settings import load_settings
from app.kg.neo4j_client import KGClient
from app.llm.client import LLMClient
from app.nlu.chitchat import Chitchat
from app.nlu.intent import IntentModel
from app.nlu.slot import SlotFiller
from app.nlu.pipeline import NluPipeline
from app.orchestrator.controller import Controller
from app.memory.store import MemoryStore
import os

app = FastAPI(title="KBQA")
_settings = load_settings()
_kg = KGClient(_settings)
_llm = LLMClient(_settings)

_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_diseases = os.path.join(_root, "knowledge_extraction", "bilstm_crf", "checkpoint", "diseases.json")
_chit = Chitchat(os.path.join(_root, "nlu", "sklearn_Classification", "model_file"))
_intent = IntentModel(os.path.join(_root, "nlu", "bert_intent_recognition", "pytorch", "checkpoint"))
_slot = SlotFiller(_diseases) if os.path.exists(_diseases) else None
_nlu = NluPipeline(_chit, _intent, _slot) if _slot else None
_memory = MemoryStore(_settings)
_controller = Controller(_nlu, _kg, _settings, memory=_memory, llm=_llm) if _nlu else None


class ChatIn(BaseModel):
    text: str
    session_id: str = "default"


@app.get("/health")
def health():
    return {"status": "ok", "deps": {"neo4j": _kg.available, "llm": _llm.available}}


@app.post("/chat")
def chat(req: ChatIn):
    if _controller is None:
        return {"answer": "(NLU 资源缺失)", "path": "stub"}
    return _controller.handle(req.text, req.session_id)
