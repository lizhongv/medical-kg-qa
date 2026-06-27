# -*- coding: utf-8 -*-
import json
import ahocorasick


class SlotFiller:
    def __init__(self, diseases_path: str):
        with open(diseases_path, encoding="utf-8") as f:
            words = json.load(f)
        self.actree = ahocorasick.Automaton()
        for i, w in enumerate(words):
            if w:
                self.actree.add_word(w, (i, w))
        self.actree.make_automaton()

    def extract(self, text: str) -> list[str]:
        spans = []
        for end, (i, w) in self.actree.iter(text):
            spans.append((end - len(w) + 1, end, w))
        final = []
        for s, e, w in spans:
            if not any(s2 <= s and e <= e2 and (s2, e2) != (s, e) for s2, e2, _ in spans):
                final.append(w)
        seen, out = set(), []
        for w in final:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out
