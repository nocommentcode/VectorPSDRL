import numpy as np
from ..agent.ensemble_bnn_model import BatchEnsembleBNNModel, EnsembleBNNModel
from ..agent.env_model import EnvModel
from ..agent.neural_linear_model import NeuralLinearModel

from .psdrl import PSDRL
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.logger import Logger


def agent_model_factory(config, actions, env_dim, device, random):
    bayes = config["algorithm"]["name"]
    if bayes == "NeuralLinear":
        env_model = NeuralLinearModel(config, env_dim, actions, device)
    elif bayes == "EGreedy":
        env_model = EnvModel(config, env_dim, actions, device)
    elif bayes == "BatchEnsemble":
        env_model = BatchEnsembleBNNModel(config, env_dim, actions, device, random)
    elif bayes == "Ensemble":
        env_model = EnsembleBNNModel(config, env_dim, actions, device, random)
    else:
        raise ValueError(f"Bayes type {bayes} is not supported")

    return env_model


def Agent(
    config: dict,
    actions: list,
    logger: "Logger",
    env_dim: int,
    seed: int = None,
):
    device = "cpu" if not config["gpu"] else "cuda:0"
    random_state = np.random.RandomState(seed)

    env_model = agent_model_factory(config, actions, env_dim, device, random_state)

    return PSDRL(
        config, actions, logger, env_dim, device, random_state, seed, env_model
    )
