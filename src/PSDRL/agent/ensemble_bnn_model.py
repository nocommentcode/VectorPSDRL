from ..bnn.ensemble_linear import EnsembleLinear
from ..bnn.batch_ensemble_layer import BatchEnsembleLinear
from ..agent.env_model import EnvModel
from ..networks.bnn_transition import BNNTransition
from ..training.transition import TransitionModelTrainer


from numpy.random import RandomState


class AEnsembleBNNModel(EnvModel):
    type = "none"

    def __init__(
        self,
        config,
        env_dim: int,
        actions,
        device: str,
        random: RandomState,
    ) -> None:
        super().__init__(config, env_dim, actions, device)

        # transition model
        self.transition_network = BNNTransition(
            env_dim,
            self.num_actions,
            config["transition"],
            self.device,
            config["algorithm"]["ensemble_size"],
            self.type,
            random,
        )

        self.transition_trainer = TransitionModelTrainer(
            config["transition"],
            self.transition_network,
            self.autoencoder,
            self.terminal_network,
            config["replay"]["batch_size"],
            self.num_actions,
            self.device,
        )


class EnsembleBNNModel(AEnsembleBNNModel):
    type = EnsembleLinear


class BatchEnsembleBNNModel(AEnsembleBNNModel):
    type = BatchEnsembleLinear
