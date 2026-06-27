# -*- coding: utf-8 -*-
import sys
import os

# Add repo root to sys.path so that namespace-package imports like
# "nlu.bert_intent_recognition.pytorch.model" work without __init__.py files.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", ".."))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
