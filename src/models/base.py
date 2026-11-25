
from abc import ABC, abstractmethod

import torch
from torch import nn
from pathlib import Path


class ChoraleModel(ABC):
    """
    Abstract parent class for all chorale generation models.

    Provides a common interface for:
        - Training
        - Generation
        - Model saving and loading
        - Parameter counting
        - Device management
    """

    def __init__(self, name):
        self.name = name
        self.fitted = False

    @abstractmethod
    def generate(self, time_signature="4/4", max_steps=256, temperature=1.0, prefix=None, fermata_template=None, start_beat_pos=0, mask_fn=None, seed=None):
        """Generate a chorale sequence."""
        pass

    @abstractmethod
    def n_params(self):
        """Return the number of model parameters."""
        pass

    @abstractmethod
    def param_breakdown(self):
        """Return parameter counts by component."""
        pass

    def fit(self, train_data, **kwargs):
        """
        Optional training interface.

        Override this in subclasses that require training.
        """
        raise NotImplementedError(
            f"{self.name} does not implement fit()."
        )

    @abstractmethod
    def save(self, path):
        """Save model state."""
        pass

    @classmethod
    def load(cls, path, device="cpu"):
        """Load a saved chorale model from a checkpoint."""

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")

        checkpoint = torch.load(path, map_location=device, weights_only=False)

        # Neural model checkpoint.
        from models.factory import ModelFactory
        if "model_state" in checkpoint:
            model_name = checkpoint["model_name"]
            model_config = checkpoint["model_config"]
            model = ModelFactory.create(model_name.lower(), **model_config)
            model.load_state_dict(checkpoint["model_state"])

        # Markov model checkpoint.
        else:
            model = ModelFactory.create(
                "markov",
                order=checkpoint["order"],
                vocab_size=checkpoint["vocab_size"],
                alpha=checkpoint["alpha"],
                backoff=checkpoint["backoff"],
            )

            model.transitions = checkpoint["transitions"]
            model.fitted = checkpoint["fitted"]

        model.to_device(device)
        model.fitted = True

        return model

    def get_device(self):
        """Return the device used by the model."""

        if isinstance(self, nn.Module):
            return next(self.parameters()).device

        return torch.device("cpu")

    def to_device(self, device):
        """Move PyTorch models to a device."""

        if isinstance(self, nn.Module):
            self.to(device)

        return self

    def summary(self):
        """Print a summary of the model."""

        print(f"\nModel: {self.name}")
        print(f"Parameters: {self.n_params():,}")
        print("\nParameter breakdown:")

        for name, count in self.param_breakdown().items():
            print(f"  {name}: {count:,}")

        print(f"\nDevice: {self.get_device()}")
        print(f"Fitted: {self.fitted}")

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(name='{self.name}', fitted={self.fitted})"
        )
