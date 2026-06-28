# -*- coding: utf-8 -*-
"""
意图训练数据构建(A 模板生成 + B LLM 改写 + 清洗)——单文件。

目标: 原始数据标签噪声大(同句多标、语义纠缠),用"疾病实体 × 意图模板"的方式
      生成**标签干净、无歧义**的合成数据,可选用 LLM 增加口语多样性。

依赖: pip install pandas  (B 改写另需 openai)
用法:
  python gen_data.py                          # A:纯模板生成 -> ../data/train_aug.csv
  python gen_data.py --per_intent 400         # 每个疾病意图生成约 N 条
  python gen_data.py --mix_real               # 额外混入"清洗后的真实训练数据"
  python gen_data.py --llm --llm_ratio 0.3    # B:对 30% 句子用 LLM 口语化改写(需 API key)
  python gen_data.py --clean_test             # 另产出 ../data/test_clean.csv(去重/去矛盾)

LLM(B)读环境变量: KBQA_LLM_API_KEY / KBQA_LLM_BASE_URL / KBQA_LLM_MODEL
产物: ../data/train_aug.csv(字段 text,label_class,label,与原数据一致)
"""
import os
import csv
import json
import random
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data")
DISEASES = os.path.join(HERE, "..", "..", "kg", "diseases.json")

# ---- 13 意图 × 中文模板({d}=疾病名);键名必须与数据里的 label_class 一致 ----
TEMPLATES = {
    "定义": [
        "{d}是什么病", "什么是{d}", "{d}是什么", "{d}是啥", "{d}是种什么病",
        "介绍一下{d}", "{d}指的是什么", "{d}是怎么回事", "{d}到底是什么病",
        "想了解下{d}", "解释一下{d}", "{d}是什么意思", "{d}属于什么病", "{d}这病是什么",
    ],
    "病因": [
        "{d}是什么原因引起的", "为什么会得{d}", "{d}的病因是什么", "{d}是怎么得的",
        "{d}的发病原因有哪些", "{d}是什么导致的", "得{d}的原因", "{d}咋得的",
        "引起{d}的原因是什么", "{d}是因为什么", "为啥会患{d}", "{d}的成因", "{d}由什么引起",
    ],
    "预防": [
        "{d}怎么预防", "如何预防{d}", "{d}的预防措施有哪些", "怎样避免得{d}",
        "{d}能预防吗", "预防{d}要注意什么", "怎么才能不得{d}", "{d}的防范方法",
        "{d}平时怎么预防", "防止{d}该怎么做",
    ],
    "临床表现(病症表现)": [
        "{d}有什么症状", "{d}的症状有哪些", "{d}会有哪些表现", "得了{d}会怎么样",
        "{d}的临床表现是什么", "{d}有什么不舒服", "{d}发病时什么感觉", "{d}的症状是啥",
        "{d}都有哪些症状", "{d}病人有什么表现", "{d}早期有什么症状", "怎么知道得了{d}",
    ],
    "相关病症": [
        "{d}会引起哪些并发症", "{d}有什么并发症", "{d}会伴随哪些疾病", "{d}容易并发什么病",
        "{d}的并发症有哪些", "得了{d}还会得什么病", "{d}会导致其他什么病", "{d}相关的疾病有哪些",
    ],
    "治疗方法": [
        "{d}怎么治", "{d}怎么治疗", "{d}如何治疗", "{d}吃什么药", "{d}用什么药",
        "{d}有什么治疗方法", "{d}怎么治好", "得了{d}怎么办", "{d}的治疗方案",
        "{d}怎么医", "治疗{d}的方法有哪些", "{d}该吃啥药", "{d}怎么调理",
    ],
    "所属科室": [
        "{d}挂什么科", "{d}看什么科室", "{d}应该挂哪个科", "得了{d}去哪个科",
        "{d}属于什么科", "看{d}挂什么号", "{d}去医院挂什么科", "{d}该看哪个科室",
    ],
    "传染性": [
        "{d}传染吗", "{d}会传染吗", "{d}有传染性吗", "{d}会不会传染给别人",
        "{d}传染人吗", "{d}是传染病吗", "{d}会传染给家人吗", "{d}通过什么传染",
        "和{d}病人接触会被传染吗",
    ],
    "治愈率": [
        "{d}治愈率高吗", "{d}能治好吗", "{d}的治愈率是多少", "{d}好治吗",
        "{d}治好的几率大吗", "{d}能根治吗", "{d}康复的可能性大吗", "{d}治愈的概率高不高",
    ],
    "禁忌": [
        "{d}不能吃什么", "{d}有什么忌口", "得了{d}不能吃啥", "{d}饮食要注意什么",
        "{d}忌吃什么食物", "{d}哪些东西不能吃", "{d}有什么饮食禁忌", "{d}患者不宜吃什么",
    ],
    "化验/体检方案": [
        "{d}要做什么检查", "{d}需要做哪些检查", "诊断{d}要做什么检查", "{d}怎么检查出来",
        "查{d}要做什么", "{d}做什么化验", "{d}要做哪些体检项目", "确诊{d}需要什么检查",
    ],
    "治疗时间": [
        "{d}要治多久", "{d}多长时间能好", "{d}的治疗周期是多久", "{d}治疗要多长时间",
        "{d}多久能治好", "{d}恢复要多久", "{d}一般治多久", "{d}疗程多长",
    ],
}

# 其他:无关/闲聊/非13类意图(不含疾病槽)
OTHER = [
    "你好", "你是谁", "你叫什么名字", "今天天气怎么样", "谢谢你", "你会做什么",
    "讲个笑话", "现在几点了", "再见", "你好厉害", "帮我推荐个餐厅", "1加1等于几",
    "你能干嘛", "唱首歌", "明天会下雨吗", "怎么去北京", "我心情不好", "你多大了",
    "在吗", "吃了吗", "你是机器人吗", "好无聊", "陪我聊聊天", "你喜欢什么",
    "股票怎么买", "今天星期几", "帮我写首诗", "附近有什么好玩的", "怎么减肥", "怎么赚钱",
    "你好呀", "哈喽", "拜拜", "早上好", "晚安", "你真棒", "我想睡觉", "电影推荐一下",
]


def load_label_map():
    """从现有 train.csv 读取 label_class -> label(id)映射,确保和原数据一致。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "train.csv"))
    name2id = {}
    for r in df.itertuples():
        name2id[str(r.label_class)] = int(r.label)
    return name2id


def load_diseases(min_len=2, max_len=8):
    ds = json.load(open(DISEASES, encoding="utf-8"))
    return [d for d in ds if min_len <= len(d) <= max_len]


def gen_templates(per_intent, name2id, seed=42):
    rng = random.Random(seed)
    diseases = load_diseases()
    rows = []  # (text, label_class, label)
    for name, tpls in TEMPLATES.items():
        if name not in name2id:
            continue
        seen = set()
        tries = 0
        while len([r for r in rows if r[1] == name]) < per_intent and tries < per_intent * 20:
            tries += 1
            t = rng.choice(tpls)
            d = rng.choice(diseases)
            text = t.format(d=d)
            if text in seen:
                continue
            seen.add(text)
            rows.append((text, name, name2id[name]))
    # 其他类
    if "其他" in name2id:
        others = list(dict.fromkeys(OTHER))
        for text in others:
            rows.append((text, "其他", name2id["其他"]))
    return rows


def clean_real_train(name2id):
    """去重 + 删除同句多标(矛盾)样本,返回干净的真实训练行。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "train.csv"))
    # 删同句多标
    conflict = df.groupby("text")["label_class"].nunique()
    bad = set(conflict[conflict > 1].index)
    df = df[~df["text"].isin(bad)].drop_duplicates(subset=["text"])
    return [(str(r.text), str(r.label_class), int(r.label)) for r in df.itertuples()]


def clean_test():
    """产出去重/去矛盾的干净测试集。"""
    import pandas as pd
    df = pd.read_csv(os.path.join(DATA, "test.csv"))
    conflict = df.groupby("text")["label_class"].nunique()
    bad = set(conflict[conflict > 1].index)
    df = df[~df["text"].isin(bad)].drop_duplicates(subset=["text"])
    out = os.path.join(DATA, "test_clean.csv")
    df[["text", "label_class", "label"]].to_csv(out, index=False, encoding="utf-8")
    print(f"清洗测试集 -> {out}  ({len(df)} 条)")


def llm_paraphrase(rows, ratio, n_variants=1):
    """B:对部分句子用 OpenAI 兼容 API 做口语化改写,保持意图不变。"""
    key = os.environ.get("KBQA_LLM_API_KEY")
    if not key:
        print("⚠️ 未设置 KBQA_LLM_API_KEY,跳过 LLM 改写(B)")
        return []
    from openai import OpenAI
    client = OpenAI(base_url=os.environ.get("KBQA_LLM_BASE_URL"), api_key=key)
    model = os.environ.get("KBQA_LLM_MODEL", "gpt-4o-mini")
    rng = random.Random(7)
    picked = rng.sample(rows, max(1, int(len(rows) * ratio)))
    extra = []
    for i, (text, name, lid) in enumerate(picked):
        prompt = (f"把下面这句医疗问句改写成 {n_variants} 个更口语、更自然的说法,"
                  f"意图保持不变(仍是问「{name}」),可加入常见口语/轻微错别字。"
                  f"每行一个,不要编号,不要解释。\n句子:{text}")
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}])
            for line in resp.choices[0].message.content.strip().splitlines():
                line = line.strip().lstrip("0123456789.、-） )").strip()
                if line and line != text:
                    extra.append((line, name, lid))
        except Exception as e:
            print("  LLM 改写失败一条:", e)
        if (i + 1) % 50 == 0:
            print(f"  LLM 改写进度 {i+1}/{len(picked)}")
    return extra


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per_intent", type=int, default=350, help="每个疾病意图生成约 N 条")
    ap.add_argument("--mix_real", action="store_true", help="混入清洗后的真实训练数据")
    ap.add_argument("--llm", action="store_true", help="B:用 LLM 口语化改写增多样性(需 API key)")
    ap.add_argument("--llm_ratio", type=float, default=0.3, help="改写句子占比")
    ap.add_argument("--clean_test", action="store_true", help="额外产出干净测试集 test_clean.csv")
    ap.add_argument("--out", type=str, default=os.path.join(DATA, "train_aug.csv"))
    args = ap.parse_args()

    name2id = load_label_map()
    rows = gen_templates(args.per_intent, name2id)
    print(f"A 模板生成: {len(rows)} 条")

    if args.llm:
        extra = llm_paraphrase(rows, args.llm_ratio)
        print(f"B LLM 改写新增: {len(extra)} 条")
        rows += extra

    if args.mix_real:
        real = clean_real_train(name2id)
        print(f"混入清洗后真实数据: {len(real)} 条")
        rows += real

    # 全局去重
    seen, final = set(), []
    for text, name, lid in rows:
        if text not in seen:
            seen.add(text)
            final.append((text, name, lid))

    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["text", "label_class", "label"])
        w.writerows(final)
    print(f"写出 {len(final)} 条 -> {args.out}")

    # 各类计数
    from collections import Counter
    c = Counter(n for _, n, _ in final)
    print("各类条数:")
    for k, v in sorted(c.items(), key=lambda x: -x[1]):
        print(f"  {k:18s} {v}")

    if args.clean_test:
        clean_test()


if __name__ == "__main__":
    main()
