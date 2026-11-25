"""Decoder-only Transformer over the interleaved S-A-T-B token sequence, plus sampling.

Forward pass (per position i):
    h_i = E_tok[x_i] + E_pos[i] + E_voice[voice_i] + E_beat[beat_pos_i] + E_ferm[fermata_i]
    h   = N x (pre-LN causal self-attention + pre-LN feed-forward, each with a residual)
    logits_i = W_out LN(h_i)          (W_out tied to E_tok)
"""
import torch
from torch import nn

from models.neural_base import NeuralChoraleModel


class ChoraleTransformer(NeuralChoraleModel):
    def __init__(self, d_model=256, n_layers=6, n_heads=4, d_ff=1024, dropout=0.1, max_len=1024):

        super().__init__(name="Transformer", d_model=d_model, dropout=dropout, max_len=max_len)
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.pos = nn.Embedding(max_len, d_model)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_ff, dropout, activation="gelu",
                                                   batch_first=True, norm_first=True)

        self.blocks = nn.TransformerEncoder(layer, n_layers, enable_nested_tensor=False)

    def forward(self, input_ids, voice, beat_pos, fermata, pad_mask=None):
        """All inputs (B, L) LongTensors; pad_mask True at padding. Returns logits (B, L, V)."""
        L = input_ids.shape[1]
        pos = torch.arange(L, device=input_ids.device)
        h = self.embed_inputs(input_ids, voice, beat_pos, fermata) + self.pos(pos)
        causal = nn.Transformer.generate_square_subsequent_mask(L, device=input_ids.device)
        # Causal mask alone is enough: padding sits only at the end, so real positions never
        # attend to it, and padded positions' outputs are ignored by the loss (-100 targets).
        h = self.blocks(self.drop(h), mask=causal, is_causal=True)
        return self.head(self.norm(h))

    def get_config(self):
        return {
            **super().get_config(),
            "n_layers": self.n_layers,
            "n_heads": self.n_heads,
            "d_ff": self.d_ff,
            "dropout": self.drop.p,
        }
