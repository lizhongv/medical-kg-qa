# -*- coding: utf-8 -*-
import torch
from torch.utils.data import Dataset


def read_bio(path):
    sentences = []
    chars, tags = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.strip() == "":
                if chars:
                    sentences.append((chars, tags))
                    chars, tags = [], []
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            chars.append(parts[0])
            tags.append(parts[-1])
    if chars:
        sentences.append((chars, tags))
    return sentences


def build_vocab(sentences, min_freq=1):
    from collections import Counter
    c = Counter()
    for chars, _ in sentences:
        c.update(chars)
    vocab = {"<PAD>": 0, "<UNK>": 1}
    for ch, freq in sorted(c.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq >= min_freq:
            vocab[ch] = len(vocab)
    return vocab


def build_tag2id(sentences):
    tags = set()
    for _, t in sentences:
        tags.update(t)
    tag2id = {"<PAD>": 0}
    for t in sorted(tags):
        tag2id[t] = len(tag2id)
    return tag2id


class NerDataset(Dataset):
    def __init__(self, sentences, vocab, tag2id):
        self.sentences = sentences
        self.vocab = vocab
        self.tag2id = tag2id

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        chars, tags = self.sentences[idx]
        cid = [self.vocab.get(c, 1) for c in chars]
        tid = [self.tag2id[t] for t in tags]
        return (torch.tensor(cid, dtype=torch.long),
                torch.tensor(tid, dtype=torch.long),
                len(cid))


def collate_fn(batch):
    lengths = [b[2] for b in batch]
    maxlen = max(lengths)
    n = len(batch)
    chars = torch.zeros(n, maxlen, dtype=torch.long)
    tags = torch.zeros(n, maxlen, dtype=torch.long)
    mask = torch.zeros(n, maxlen, dtype=torch.bool)
    for i, (cid, tid, ln) in enumerate(batch):
        chars[i, :ln] = cid
        tags[i, :ln] = tid
        mask[i, :ln] = True
    return chars, tags, mask, torch.tensor(lengths, dtype=torch.long)
