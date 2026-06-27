import torch
from knowledge_extraction.bilstm_crf.pytorch.model import BiLstmCrf


def test_forward_shapes():
    m = BiLstmCrf(vocab_size=50, num_tags=7)
    chars = torch.randint(0, 50, (4, 10))
    mask = torch.ones(4, 10, dtype=torch.bool)
    em = m(chars, mask)
    assert em.shape == (4, 10, 7)


def test_loss_and_decode():
    m = BiLstmCrf(vocab_size=50, num_tags=7)
    chars = torch.randint(0, 50, (4, 10))
    tags = torch.randint(0, 7, (4, 10))
    mask = torch.ones(4, 10, dtype=torch.bool)
    loss = m.loss(chars, tags, mask)
    assert loss.dim() == 0 and loss.item() > 0
    paths = m.decode(chars, mask)
    assert len(paths) == 4 and len(paths[0]) == 10
