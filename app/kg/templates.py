# -*- coding: utf-8 -*-
import sys
import os

# 引入仓库根 config.semantic_slot
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from config import semantic_slot


def render(intent, slots):
    info = semantic_slot.get(intent)
    if not info or "cql_template" not in info:
        return []
    tpl = info["cql_template"]
    tpls = tpl if isinstance(tpl, list) else [tpl]
    return [t.format(**slots) for t in tpls]


def reply_prefix(intent, slots):
    info = semantic_slot.get(intent) or {}
    tpl = info.get("reply_template", "")
    try:
        return tpl.format(**slots)
    except Exception:
        return tpl
