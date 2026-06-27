# -*- coding: utf-8 -*-
"""尝试把 hfl/rbt3 下到 ./base_model/。若网络受限,训练脚本会回退到 Hub 名称。"""
import os

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "base_model")


def main():
    os.makedirs(OUT, exist_ok=True)
    try:
        from transformers import AutoTokenizer, AutoModel
        tok = AutoTokenizer.from_pretrained("hfl/rbt3")
        mdl = AutoModel.from_pretrained("hfl/rbt3")
        tok.save_pretrained(OUT)
        mdl.save_pretrained(OUT)
        print("基座模型已保存到", OUT)
    except Exception as e:
        print("下载失败:", e)
        print("可在 GPU 机器联网后重跑本脚本,或让 train.py 直接用 'hfl/rbt3' 自动下载。")


if __name__ == "__main__":
    main()
