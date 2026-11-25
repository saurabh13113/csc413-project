
"""N-gram Markov model for Bach chorale generation."""

import math
import random
from collections import Counter, defaultdict
import torch
from pathlib import Path

from models.base import ChoraleModel
from chorales.grammar import allowed_tokens
from chorales.vocab import END, START, VOCAB_SIZE, N_VOICES


class ChoraleMarkov(ChoraleModel):

    def __init__(self, order=3, vocab_size=VOCAB_SIZE, alpha=0.1, backoff=True):

        super().__init__(name="Markov")

        self.order = order
        self.vocab_size = vocab_size
        self.alpha = alpha
        self.backoff = backoff

        # transitions[n][context] = Counter of next-token frequencies.
        self.transitions = [defaultdict(Counter) for _ in range(order + 1)]
        self.fitted = False

    def fit(self, sequences):
        for sequence in sequences:
            for i in range(1, len(sequence)):
                token = sequence[i]
                if not 0 <= token < self.vocab_size:
                    raise ValueError(f"Invalid token ID: {token}")
                for n in range(self.order + 1):
                    if i < n:
                        continue
                    context = tuple(sequence[i - n:i])
                    self.transitions[n][context][token] += 1
        self.fitted = True
        return self

    def predict(self, context):
        if not self.fitted:
            raise RuntimeError("Model must be fitted before prediction")
        
        max_order = min(len(context), self.order)
        orders = (range(max_order, -1, -1) if self.backoff else [max_order])

        for n in orders:
            current_context = tuple(context[-n:]) if n > 0 else ()
            counts = self.transitions[n].get(current_context)
            if counts:
                total = sum(counts.values())
                denominator = total + self.alpha * self.vocab_size
                return {
                    token: (counts.get(token, 0) + self.alpha) / denominator
                    for token in range(self.vocab_size)
                }

        # Should only occur if the model was fitted on no sequences.
        return {
            token: 1.0 / self.vocab_size
            for token in range(self.vocab_size)
        }

    def next_token_probability(self, context, target):
        """Probability of a specific next token."""

        return self.predict(context).get(target, 0.0)

    def save(self, path):

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        torch.save({
            "model_name": self.name,
            "order": self.order,
            "vocab_size": self.vocab_size,
            "alpha": self.alpha,
            "backoff": self.backoff,
            "transitions": self.transitions,
            "fitted": self.fitted,
        }, path)

        print(f"{self.name} saved to {path}")

    def generate(self, time_signature="4/4", max_steps=256, temperature=1.0, prefix=None, fermata_template=None, start_beat_pos=0, mask_fn=allowed_tokens, seed=None):
        """Generate a token sequence"""

        sequence = list(prefix) if prefix is not None else [START]
        rng = random.Random(seed)

        while (sequence[-1] != END and (len(sequence) - 1) // N_VOICES < max_steps):
            context = sequence[-self.order:]
            probabilities = self.predict(context)
            # Apply grammar constraints.
            if mask_fn is not None:
                allowed = mask_fn(sequence)
                probabilities = {
                    token: probability
                    for token, probability in probabilities.items()
                    if token in allowed
                }

            if not probabilities:
                raise RuntimeError(
                    f"No valid next token at position {len(sequence)}"
                )
            
            # Temperature sampling.
            tokens = list(probabilities.keys())
            weights = [probability ** (1.0 / temperature) for probability in probabilities.values()]
            next_token = rng.choices(tokens, weights=weights, k=1)[0]
            sequence.append(next_token)

        # If the generation limit is reached, close the chorale
        if sequence[-1] != END:
            complete_length = 1 + (len(sequence) - 1) // N_VOICES * N_VOICES
            sequence = sequence[:complete_length] + [END]

        return sequence

    def sequence_nll(self, sequence):
        """Calculate mean per-token negative log-likelihood."""

        total_loss = 0.0
        count = 0

        for i in range(1, len(sequence)):
            context = sequence[:i]
            target = sequence[i]
            probability = self.next_token_probability(context, target)
            total_loss -= math.log(max(probability, 1e-12))
            count += 1

        return total_loss / count if count else 0.0

    def n_params(self):
        """Number of observed transition-count entries."""
        return sum(len(counter) for contexts in self.transitions for counter in contexts.values())

    def param_breakdown(self):
        """Number of observed transition entries at each order."""

        return {
            f"order_{n}": sum(len(counter) for counter in contexts.values())
            for n, contexts in enumerate(self.transitions)
        }
