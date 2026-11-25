
from models.transformer import ChoraleTransformer
from models.lstm import ChoraleLSTM
from models.markov import ChoraleMarkov


class ModelFactory:
    """Create chorale models using a common interface."""

    MODELS = {
        "transformer": ChoraleTransformer,
        "lstm": ChoraleLSTM,
        "markov": ChoraleMarkov,
    }

    @classmethod
    def create(cls, model_name, **kwargs):
        model_name = model_name.lower()

        if model_name not in cls.MODELS:
            raise ValueError(
                f"Unknown model: {model_name}. "
                f"Available models: {list(cls.MODELS.keys())}"
            )

        return cls.MODELS[model_name](**kwargs)

    @classmethod
    def available_models(cls):
        return list(cls.MODELS.keys())
