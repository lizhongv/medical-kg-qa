# -*- coding: utf-8 -*-
class NluPipeline:
    def __init__(self, chitchat, intent, slot):
        self.chitchat = chitchat
        self.intent = intent
        self.slot = slot

    def analyze(self, text):
        chat = self.chitchat.classify(text)
        if chat:
            return {"kind": "chitchat", "intent": chat, "confidence": 1.0,
                    "slots": {"Disease": None}, "matched": False}
        intent = self.intent.predict(text)
        diseases = self.slot.extract(text)
        disease = diseases[0] if diseases else None
        return {"kind": "diagnosis", "intent": intent["name"],
                "confidence": intent["confidence"],
                "slots": {"Disease": disease}, "matched": disease is not None}
