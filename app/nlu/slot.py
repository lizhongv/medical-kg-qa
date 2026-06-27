# -*- coding: utf-8 -*-
import json
import ahocorasick


class SlotFiller:
    def __init__(self, diseases_path):
        words = json.load(open(diseases_path, encoding="utf-8"))
        self.actree = ahocorasick.Automaton()
        for i, w in enumerate(words):
            if w:
                self.actree.add_word(w, (i, w))
        self.actree.make_automaton()

    def extract(self, text):
        hits = [v[1] for _, v in self.actree.iter(text)]
        # 去掉被更长匹配包含的短词
        final = [w for w in hits if not any(w != o and w in o for o in hits)]
        # 去重保序
        seen, out = set(), []
        for w in final:
            if w not in seen:
                seen.add(w)
                out.append(w)
        return out
