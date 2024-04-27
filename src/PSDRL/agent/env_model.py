from ..common.replay import Dataset
from ..common.utils import create_state_action_batch
from ..networks.representation import AutoEncoder
from ..networks.terminal import Network as TerminalNetwork
from ..networks.transition import Network as TransitionNetwork
from ..training.representation import RepresentationTrainer
from ..training.transition import TransitionModelTrainer


import torch


class EnvModel:
    def __init__(self, config, env_dim: int, actions, device: str) -> None:
        self.device = device
        self.env_dim = env_dim
        self.num_actions = len(actions)
        self.actions = torch.tensor(actions).to(self.device)

        # representation
        if config["visual"]:
            self.autoencoder = AutoEncoder(config["representation"], self.device)
            self.representation_trainer = RepresentationTrainer(
                config["representation"]["training_iterations"], self.autoencoder
            )
        else:
            self.autoencoder = None
            self.representation_trainer = None

        # transition model
        self.terminal_network = TerminalNetwork(
            env_dim, config["terminal"], self.device
        )
        self.transition_network = TransitionNetwork(
            env_dim,
            self.num_actions,
            config["transition"],
            self.device,
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

        self.reset_hidden_state()

    def train_(self, dataset: Dataset) -> None:
        if self.representation_trainer is not None:
            self.representation_trainer.train_(dataset)
        self.transition_trainer.train_(dataset)

    def reset_hidden_state(self) -> None:
        self.prev_state = torch.zeros(self.transition_network.gru_dim).to(self.device)

    def get_hidden_state(self) -> torch.tensor:
        return self.prev_state

    def set_hidden_state(self, state: torch.tensor) -> None:
        self.prev_state = state

    def embed_observation(self, obs: torch.tensor) -> torch.tensor:
        if self.autoencoder is None:
            return obs

        return self.autoencoder.embed(obs)

    def predict(self, obs: torch.tensor, hidden_state: torch.tensor = None):
        # set hidden state to current hidden if not specified
        if hidden_state is None:
            hidden_state = self.prev_state

        obs, h = create_state_action_batch(
            obs, self.actions, hidden_state, self.num_actions, self.device
        )

        prediction, h1 = self.transition_network.predict(obs, h)
        states, rewards = prediction[:, :-1], prediction[:, -1]
        terminals = self.terminal_network.predict(states)

        return states, rewards.reshape(-1, 1), terminals, h1
