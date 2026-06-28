# -*- coding: utf-8 -*-
"""命令行测试 /chat(用 Python 发请求,正确 UTF-8,避开终端编码问题)。
用法: python test_chat.py   或   python test_chat.py "你的问题"
"""
import sys
import json
import urllib.request

URL = "http://localhost:8000/chat"


def ask(text, sid="u1"):
    data = json.dumps({"text": text, "session_id": sid}).encode("utf-8")
    req = urllib.request.Request(URL, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.load(r)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        qs = [" ".join(sys.argv[1:])]
    else:
        qs = ["你好", "高血压怎么治", "高血压有什么症状", "糖尿病不能吃什么", "感冒挂什么科"]
    for q in qs:
        try:
            res = ask(q)
            print("Q:", q)
            print("  path:", res.get("path"))
            print("  A:", res.get("answer"))
            print()
        except Exception as e:
            print("Q:", q, "=> 出错:", e)
