
import torch
from torch import nn
from pathlib import Path

from models.base import ChoraleModel
from chorales.dataset import generation_features
from chorales.grammar import allowed_tokens
from chorales.vocab import (BEAT_POS_SPECIAL, END, N_VOICES, START, VOCAB_SIZE, VOICE_SPECIAL)


class NeuralChoraleModel(nn.Module, ChoraleModel):

    def __init__(self, name, d_model, dropout, max_len):

        nn.Module.__init__(self)
        ChoraleModel.__init__(self, name)

        self.max_len = max_len
        self.d_model = d_model
        self.tok = nn.Embedding(VOCAB_SIZE, d_model)
        self.voice = nn.Embedding(VOICE_SPECIAL + 1, d_model)
        self.beat = nn.Embedding(BEAT_POS_SPECIAL + 1, d_model)
        self.fermata = nn.Embedding(2, d_model)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, VOCAB_SIZE, bias=False)
        self.head.weight = self.tok.weight

    def embed_inputs(self, input_ids, voice, beat_pos, fermata):
        h = self.tok(input_ids) + self.voice(voice) + self.beat(beat_pos) + self.fermata(fermata)
        return self.drop(h)

    def forward(self, input_ids, voice, beat_pos, fermata, pad_mask=None, **kwargs):
        raise NotImplementedError("Subclasses must implement forward().")

    def n_params(self):
            return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def param_breakdown(self):
        """{top-level module: #params}, for the README parameter count."""
        return {name: sum(p.numel() for p in m.parameters()) for name, m in self.named_children()
                if name != "head"}  # head is tied to tok

    def get_device(self):
        return next(self.parameters()).device

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_name": self.name,
            "model_state": self.state_dict(),
            "model_config": self.get_config(),
        }, path)

    def get_config(self):
        return {
            "max_len": self.max_len,
            "d_model": self.d_model
        }

    @torch.no_grad()
    def generate(self, time_signature="4/4", max_steps=256, temperature=1.0, prefix=None,
                fermata_template=None, start_beat_pos=0, mask_fn=allowed_tokens, seed=None):
        """Sample one chorale token by token. Returns the id list (starts with <START>).

        mask_fn(ids_so_far) -> set of allowed next ids. Default is the format grammar; the rule
        checker can wrap it to add music rules (constrained decoding).
        """
        # ponytail: no KV cache, re-runs the whole window every token (~1k tokens fine on GPU).
        self.eval()
        device = next(self.parameters()).device
        gen = torch.Generator(device=device)
        if seed is not None:
            gen.manual_seed(seed)
        ids = list(prefix) if prefix else [START]
        feats = [generation_features(p, time_signature, fermata_template, start_beat_pos) for p in range(len(ids))]
        while ids[-1] != END and (len(ids) - 1) // N_VOICES < max_steps:
            # context window: whole sequence, or the last <= max_len tokens starting at a step boundary
            s = 0 if len(ids) <= self.max_len else len(ids) - self.max_len + (-(len(ids) - self.max_len - 1)) % N_VOICES
            x = {k: torch.tensor([[f[k] for f in feats[s:]]], device=device) for k in ("voice", "beat_pos", "fermata")}
            logits = self(torch.tensor([ids[s:]], device=device), **x)[0, -1].float()
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
