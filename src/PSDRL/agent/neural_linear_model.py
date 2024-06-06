import numpy as np
import torch
from ..agent.env_model import EnvModel

from ..common.replay import Dataset
from ..common.utils import (
    create_state_action_batch,
    state_action_append,
    extract_episode_data,
)
from ..common.settings import BATCH_EMBEDDING_SIZE, BLR_COEFFICIENT, ONE_OVER_LAMBDA


class NeuralLinearModel(EnvModel):
    def __init__(self, config, env_dim: int, actions, device: str) -> None:
        super().__init__(config, env_dim, actions, device)

        #  Posterior distribution parameters
        self.noise_variance = ONE_OVER_LAMBDA
        config = config["algorithm"]
        self.transition_prior = config["transition_prior"]
        self.reward_prior = config["reward_prior"]

        self.eye = torch.eye(self.transition_network.latent_dim).to(self.device)
        self.mu = (
            torch.zeros(env_dim + 1, self.transition_network.latent_dim)
            .add(self.noise_variance)
            .to(self.device)
        )
        self.transition_cov = self.eye.add(self.transition_prior)
        self.reward_cov = self.eye.add(self.reward_prior)

        #  Sampling parameters
        self.N = config["n_samples"]
        self.model_samples = torch.zeros(self.N, *self.mu.shape, device=self.device)
        self.randsamp = torch.zeros(self.N, *self.mu.shape).to(self.device)
        self.sample_range = torch.arange(self.N * self.num_actions).to(self.device)
        self.sample_indices = torch.repeat_interleave(
            (torch.arange(self.N)), self.num_actions
        ).to(self.device)

        # K-Learning
        self.temperature = config["temp"] if config["name"] == "K_Learning" else None

        # Random sampling from the prior
        self.sample()

    def predict(self, obs: torch.tensor, hidden_state: torch.tensor = None):
        """
        Simulate one timestep using the current sampled model(s).
        """
        if hidden_state is None:
            hidden_state = self.prev_state

        obs, h = create_state_action_batch(
            obs, self.actions, hidden_state, self.num_actions, self.device
        )

        feature_map, h = self.transition_network.get_feature_maps(obs, h)
        matmul = self.model_samples.matmul(feature_map.T)
        prediction = matmul.squeeze().T
        states, rewards = prediction[:, :-1], prediction[:, -1]
        terminals = self.terminal_network.predict(states)
        return states, rewards.reshape(-1, 1), terminals, h

    def exploration_policy(self, obs: torch.tensor, hidden_state: torch.tensor = None):
        # explore -> use neural linear
        return self.predict(obs, hidden_state)

    def exploitation_policy(self, obs: torch.tensor, hidden_state: torch.tensor = None):
        # exploit -> use normal network
        return super().predict(obs, hidden_state)

    def encode_replay_buffer(self, dataset: Dataset):
        """
        Map all observations present in the replay buffer to latent states in batches of BATCH_EMBEDDING_SIZE.
        """
        num_episodes = len(dataset.episodes)
        s_a = torch.zeros(
            (num_episodes, dataset.max_ep_len, self.env_dim + self.num_actions),
            device=self.device,
        )
        y = torch.zeros(
            (num_episodes, dataset.max_ep_len, self.env_dim + 1), device=self.device
        )
        t = torch.zeros((num_episodes, dataset.max_ep_len, 1))

        for idx, ep in enumerate(dataset.episodes):
            ep_len = len(ep["states"])
            embed_iterations = int(np.floor((ep_len - 1) / BATCH_EMBEDDING_SIZE)) + 1
            embed_idx = BATCH_EMBEDDING_SIZE

            s = torch.zeros(ep_len, self.env_dim, device=self.device)
            s1 = torch.zeros(ep_len, self.env_dim, device=self.device)
            o, a, o1, r, done = extract_episode_data([ep])
            o, a, o1, r, done = (
                o.squeeze(),
                a.reshape(-1, 1),
                o1.squeeze(),
                r.reshape(-1, 1),
                done.reshape(-1, 1),
            )
            if self.autoencoder:
                for i in range(embed_iterations):
                    s[i * embed_idx : i * embed_idx + BATCH_EMBEDDING_SIZE] = (
                        self.autoencoder.embed(
                            o[i * embed_idx : i * embed_idx + BATCH_EMBEDDING_SIZE]
                        )
                    )
                    s1[i * embed_idx : i * embed_idx + BATCH_EMBEDDING_SIZE] = (
                        self.autoencoder.embed(
                            o1[i * embed_idx : i * embed_idx + BATCH_EMBEDDING_SIZE]
                        )
                    )
            else:
                s = ep["states"]
                s1 = ep["next_states"]

            s_a[idx, :ep_len] = state_action_append(s, a, self.num_actions, self.device)
            y[idx, :ep_len] = torch.cat((s1, r), dim=1)
            t[idx, :ep_len] = done

        return num_episodes, s_a, y, t

    def compute_feature_maps(self, dataset: Dataset):
        """
        Compute the feature maps for the entire replay buffer. This is done by first mapping all observations to
        latent states. Then, all episode trajectories are traversed in parallel while storing the feature maps and
        targets of each transition.
        """
        num_episodes, s_a, y, t = self.encode_replay_buffer(dataset)
        linear_representations = torch.zeros(
            (dataset.total_num_transitions, self.transition_network.latent_dim),
            device=self.device,
        )
        targets = torch.zeros(
            (dataset.total_num_transitions, self.env_dim + 1), device=self.device
        )
        h = torch.zeros(
            (num_episodes, self.transition_network.gru_dim), device=self.device
        )

        finished = torch.tensor([])
        eps = torch.arange(0, num_episodes, 1)
        unfinished_idx = eps
        add_idx = 0
        for idx in range(dataset.max_ep_len):
            prediction, h[unfinished_idx] = self.transition_network.get_feature_maps(
                s_a[unfinished_idx, idx], h[unfinished_idx]
            )
            linear_representations[add_idx : add_idx + len(prediction)] = prediction
            targets[add_idx : add_idx + len(prediction)] = y[unfinished_idx, idx]
            add_idx = add_idx + len(prediction)

            finished = torch.cat((finished, torch.where(t[:, idx] == 1)[0]))
            unfinished_idx = torch.tensor([x for x in eps if x not in finished])

        return linear_representations, targets

    def update_posteriors(self, dataset: Dataset):
        """
        Compute feature maps for all states in the replay buffer, then update posteriors with Bayesian Linear
        Regression. Cholesky decomposition for computing the inverse is used with additional checks to ensure
        positive-semi definite matrices.
        """

        x, y = self.compute_feature_maps(dataset)
        Phi_pre = x.T.matmul(x) * self.noise_variance
        coeff = BLR_COEFFICIENT

        while True:
            input_matrix = Phi_pre.double() + (self.eye.double() * coeff)
            try:
                chol = torch.linalg.cholesky(input_matrix)
                Phi = torch.cholesky_inverse(chol).float()
            except:
                coeff *= 10
                continue

            if Phi.isnan().any():
                coeff *= 10
                continue

            self.reward_cov = Phi * self.reward_prior
            self.transition_cov = Phi * self.transition_prior
            break

        for i in range(self.env_dim + 1):
            self.mu[i] = (self.noise_variance * Phi).matmul(x.T.matmul(y[:, i]))

    def sample(self):
        self.randsamp[:] = torch.randn(self.N, *self.mu.shape)
        self.model_samples[:, :-1, :] = self.mu[:-1] + (
            self.randsamp[:, :-1, :] @ self.transition_cov
        )
        self.model_samples[:, -1, :] = self.mu[-1] + (
            self.randsamp[:, -1, :] @ self.reward_cov
        )

    def train_(self, dataset: Dataset) -> None:
        super().train_(dataset)
        self.update_posteriors(dataset)
        self.sample()
