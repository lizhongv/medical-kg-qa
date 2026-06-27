# -*- coding: utf-8 -*-
"""下载 cMedQANER(ChineseBLUE)到 ./data/。GitHub raw 通道。"""
import os
import urllib.request

BASE = "https://raw.githubusercontent.com/chenxshuo/ChineseBLUE/master/data/cMedQANER"
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "data")


def main():
    os.makedirs(OUT, exist_ok=True)
    for name in ["train.txt", "dev.txt", "test.txt"]:
        dst = os.path.join(OUT, name)
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            print(f"已存在,跳过: {name}")
            continue
        url = f"{BASE}/{name}"
        print(f"下载 {url}")
        urllib.request.urlretrieve(url, dst)
        print(f"  -> {dst} ({os.path.getsize(dst)} bytes)")


if __name__ == "__main__":
    main()
