
"""LSTM language model for interleaved Bach chorale generation."""

import torch
from torch import nn

from models.neural_base import NeuralChoraleModel


class ChoraleLSTM(NeuralChoraleModel):
    def __init__(self, d_model=384, n_layers=2, dropout=0.2, max_len=1024):
        super().__init__(name="LSTM", d_model=d_model, dropout=dropout, max_len=max_len)
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=n_layers, dropout=dropout if n_layers > 1 else 0.0, batch_first=True)
        
    def forward(self, input_ids, voice, beat_pos, fermata, pad_mask=None, hidden=None):
        """
        Inputs: (B, L) LongTensors.
        Returns: logits (B, L, VOCAB_SIZE).

        pad_mask is accepted for compatibility with the Transformer
        interface. Padded targets are excluded by the loss function.
        """

        h = self.embed_inputs(input_ids, voice, beat_pos, fermata)
        h = self.drop(h)
        h, hidden = self.lstm(h, hidden)

        return self.head(self.norm(h))

    def initial_hidden(self, batch_size, device):
        """Create zero-initialized recurrent states."""

        shape = (self.n_layers, batch_size, self.d_model)

        return (torch.zeros(shape, device=device), torch.zeros(shape, device=device))


    def get_config(self):
        return {
            **super().get_config(),
            "n_layers": self.n_layers,
            "dropout": self.drop.p,
        }
