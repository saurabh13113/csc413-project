import torch

from chorales.grammar import allowed_tokens
from models.transformer import ChoraleTransformer
from chorales.vocab import END, START


def test_causal_and_generate():
    torch.manual_seed(0)
    m = ChoraleTransformer(d_model=32, n_layers=2, n_heads=2, d_ff=64, dropout=0.0, max_len=64).eval()
    L = 20
    x = torch.randint(3, 51, (1, L))
    f = {"voice": torch.randint(0, 4, (1, L)), "beat_pos": torch.randint(0, 16, (1, L)),
         "fermata": torch.zeros(1, L, dtype=torch.long)}
    a = m(x, **f)
    x2 = x.clone()
    x2[0, 10:] = 5  # change the future
    b = m(x2, **f)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5)  # past logits unaffected
    assert not torch.allclose(a[0, 10:], b[0, 10:])

    # generation longer than max_len exercises the sliding context window
    ids = m.generate(max_steps=30, seed=0)
    assert ids[0] == START and ids[-1] == END and (len(ids) - 2) % 4 == 0
    for i in range(1, len(ids)):
        assert ids[i] in allowed_tokens(ids[:i])
