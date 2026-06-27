import os
import torch
from knowledge_extraction.bilstm_crf.pytorch.data_loader import (
    read_bio, build_vocab, build_tag2id, NerDataset, collate_fn,
)

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = os.path.join(HERE, "..", "data", "train.txt")


def test_read_bio_basic():
    sents = read_bio(TRAIN)
    assert len(sents) > 1000
    chars, tags = sents[0]
    assert len(chars) == len(tags)
    assert all(t == "O" or t.startswith("B_") or t.startswith("I_") for t in tags)


def test_build_maps():
    sents = read_bio(TRAIN)
    vocab = build_vocab(sents)
    tag2id = build_tag2id(sents)
    assert vocab["<PAD>"] == 0 and vocab["<UNK>"] == 1
    assert tag2id["<PAD>"] == 0
    assert "B_disease" in tag2id


def test_dataset_and_collate():
    sents = read_bio(TRAIN)[:8]
    vocab = build_vocab(sents)
    tag2id = build_tag2id(sents)
    ds = NerDataset(sents, vocab, tag2id)
    batch = [ds[i] for i in range(4)]
    chars, tags, mask, lengths = collate_fn(batch)
    assert chars.shape == tags.shape == mask.shape
    assert chars.shape[0] == 4
    assert mask.dtype == torch.bool
