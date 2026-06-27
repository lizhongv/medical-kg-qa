import torch
from nlu.bert_intent_recognition.pytorch.model import TextCnnHead


def test_textcnn_head_shape():
    head = TextCnnHead(hidden=32, num_labels=13)
    cls = torch.randn(4, 32)
    tokens = torch.randn(4, 18, 32)  # [B, L-2, H]
    logits = head(cls, tokens)
    assert logits.shape == (4, 13)
