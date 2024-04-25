from .psdrl import PSDRL
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..common.logger import Logger


def Agent(
    config: dict,
    actions: list,
    logger: "Logger",
    env_dim : int,
    seed: int = None,
):
    agent = PSDRL
    return agent(
        config,
        actions,
        logger,
        env_dim,
        seed,
    )
