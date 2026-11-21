"""Decoder-only Transformer over the interleaved S-A-T-B token sequence, plus sampling.

Forward pass (per position i):
    h_i = E_tok[x_i] + E_pos[i] + E_voice[voice_i] + E_beat[beat_pos_i] + E_ferm[fermata_i]
    h   = N x (pre-LN causal self-attention + pre-LN feed-forward, each with a residual)
    logits_i = W_out LN(h_i)          (W_out tied to E_tok)
"""
import torch
from torch import nn

from chorales.dataset import generation_features
from chorales.grammar import allowed_tokens
from chorales.vocab import BEAT_POS_SPECIAL, END, N_VOICES, START, VOCAB_SIZE, VOICE_SPECIAL


class ChoraleTransformer(nn.Module):
    def __init__(self, d_model=256, n_layers=6, n_heads=4, d_ff=1024, dropout=0.1, max_len=1024):
        super().__init__()
        self.max_len = max_len
        self.tok = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.voice = nn.Embedding(VOICE_SPECIAL + 1, d_model)
        self.beat = nn.Embedding(BEAT_POS_SPECIAL + 1, d_model)
        self.fermata = nn.Embedding(2, d_model)
        self.drop = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight  # weight tying

    def forward(self, input_ids, voice, beat_pos, fermata, pad_mask=None):
        """All inputs (B, L) LongTensors; pad_mask True at padding. Returns logits (B, L, V)."""
        L = input_ids.shape[1]
        pos = torch.arange(L, device=input_ids.device)
        h = self.tok(input_ids) + self.pos(pos) + self.voice(voice) + self.beat(beat_pos) + self.fermata(fermata)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=input_ids.device)
        # Causal mask alone is enough: padding sits only at the end, so real positions never
        # attend to it, and padded positions' outputs are ignored by the loss (-100 targets).
        h = self.blocks(self.drop(h), mask=causal, is_causal=True)
        return self.head(self.norm(h))


def n_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def param_breakdown(model):
    """{top-level module: #params}, for the README parameter count."""
    return {name: sum(p.numel() for p in m.parameters()) for name, m in model.named_children()
            if name != "head"}  # head is tied to tok


@torch.no_grad()
def generate(model, time_signature="4/4", max_steps=256, temperature=1.0, prefix=None,
             fermata_template=None, start_beat_pos=0, mask_fn=allowed_tokens, seed=None):
    """Sample one chorale token by token. Returns the id list (starts with <START>).

    mask_fn(ids_so_far) -> set of allowed next ids. Default is the format grammar; the rule
    checker can wrap it to add music rules (constrained decoding).
    """
    # ponytail: no KV cache, re-runs the whole window every token (~1k tokens fine on GPU).
    model.eval()
    device = next(model.parameters()).device
    gen = torch.Generator(device=device)
    if seed is not None:
        gen.manual_seed(seed)
    ids = list(prefix) if prefix else [START]
    feats = [generation_features(p, time_signature, fermata_template, start_beat_pos) for p in range(len(ids))]
    while ids[-1] != END and (len(ids) - 1) // N_VOICES < max_steps:
        # context window: whole sequence, or the last <= max_len tokens starting at a step boundary
        s = 0 if len(ids) <= model.max_len else len(ids) - model.max_len + (-(len(ids) - model.max_len - 1)) % N_VOICES
        x = {k: torch.tensor([[f[k] for f in feats[s:]]], device=device) for k in ("voice", "beat_pos", "fermata")}
        logits = model(torch.tensor([ids[s:]], device=device), **x)[0, -1].float()
        allowed = mask_fn(ids)
        if not allowed:
            raise RuntimeError(f"no allowed token at position {len(ids)}")
        mask = torch.full_like(logits, float("-inf"))
        mask[list(allowed)] = 0
        probs = torch.softmax((logits + mask) / temperature, dim=-1)
        ids.append(int(torch.multinomial(probs, 1, generator=gen)))
        feats.append(generation_features(len(ids) - 1, time_signature, fermata_template, start_beat_pos))
    if ids[-1] != END:  # hit max_steps: close the piece at the next step boundary
        ids = ids[:1 + (len(ids) - 1) // N_VOICES * N_VOICES] + [END]
    return ids
