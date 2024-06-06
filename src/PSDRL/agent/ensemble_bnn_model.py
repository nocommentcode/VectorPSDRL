import numpy as np
import torch

from ..common.utils import create_state_action_batch
from ..common.replay import Dataset
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
        self.random = random

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

        self.ensemble_size = config["algorithm"]["ensemble_size"]
        self.exploration_mode = config["algorithm"]["exploration"]

        self.transition_trainer = TransitionModelTrainer(
            config["transition"],
            self.transition_network,
            self.autoencoder,
            self.terminal_network,
            config["replay"]["batch_size"],
            self.num_actions,
            self.device,
        )
        self.reset_diversity_measure()
        self.sample()

    def reset_diversity_measure(self):
        self.diversity_std = []

    def record_diversity(self, output):
        self.diversity_std.append(output.std(0).sum().item())

    def flush_diversity(self):
        std = np.array(self.diversity_std).mean()
        self.reset_diversity_measure()
        return std

    def sample_ensemble(self, output, record_deversity=False):
        output = output.view((self.ensemble_size, -1, *output.shape[1:]))

        if record_deversity:
            self.record_diversity(output)

        return output[self.sample_index]

    def exploration_policy(self, obs: torch.tensor, hidden_state: torch.tensor = None):
        # set hidden state to current hidden if not specified
        if hidden_state is None:
            hidden_state = self.prev_state

        obs, h = create_state_action_batch(
            obs, self.actions, hidden_state, self.num_actions, self.device
        )

        # sample new model for each step if shallow
        if self.exploration_mode == "shallow":
            self.sample()

        # predict only for sampled ensemble
        prediction, h1 = self.transition_network.predict(
            obs, h, ensemble_index=self.sample_index
        )

        # calculate states, rewards and terminals
        states, rewards = prediction[:, :-1], prediction[:, -1]
        terminals = self.terminal_network.predict(states)
        return states, rewards.reshape(-1, 1), terminals, h1

    def exploitation_policy(self, obs: torch.tensor, hidden_state: torch.tensor = None):
        # set hidden state to current hidden if not specified
        if hidden_state is None:
            hidden_state = self.prev_state

        obs, h = create_state_action_batch(
            obs, self.actions, hidden_state, self.num_actions, self.device
        )

        # replicate obs and hidden for each ensemble
        obs = torch.concatenate([obs for _ in range(self.ensemble_size)], 0)
        h = torch.concatenate([h for _ in range(self.ensemble_size)], 0)

        # predict for all ensemble
        prediction, h1 = self.transition_network.predict(obs, h, ensemble_index=None)

        # record diversity
        self.record_diversity(prediction)

        # calculate states, rewards and terminals
        states, rewards = prediction[:, :-1], prediction[:, -1]
        terminals = self.terminal_network.predict(states)
        return states, rewards.reshape(-1, 1), terminals, h1

    def sample(self):
        self.sample_index = self.random.randint(0, self.ensemble_size)

    def train_(self, dataset: Dataset) -> None:
        super().train_(dataset)
        if len(self.diversity_std) > 0:
            dataset.logger.add_scalars("BNN/Ensemble STD", self.flush_diversity())
        self.sample()


class EnsembleBNNModel(AEnsembleBNNModel):
    type = EnsembleLinear


class BatchEnsembleBNNModel(AEnsembleBNNModel):
    type = BatchEnsembleLinear
