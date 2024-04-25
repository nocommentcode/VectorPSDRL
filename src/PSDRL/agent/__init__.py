from ..agent.env_model import EnvModel
from ..agent.neural_linear_model import NeuralLinearModel

from .psdrl import PSDRL
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.logger import Logger


def Agent(
    config: dict,
    actions: list,
    logger: "Logger",
    env_dim: int,
    seed: int = None,
):
    bayes = config["algorithm"]["name"]
    if bayes == "NeuralLinear":
        env_model = NeuralLinearModel
    elif bayes == "EGreedy":
        env_model = EnvModel
    else:
        raise ValueError(f"Bayes type {bayes} is not supported")

    return PSDRL(config, actions, logger, env_dim, seed, env_model)
