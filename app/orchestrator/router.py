# -*- coding: utf-8 -*-
def route(nlu, settings):
    if nlu.get("kind") == "chitchat":
        return "chitchat"
    conf = nlu.get("confidence", 0.0)
    if nlu.get("intent") and conf >= settings.accept_threshold and nlu.get("matched"):
        return "fast"
    return "slow"
