import enum
import torch
import numpy as np

from ..agent.env_model import EnvModel

from ..common.logger import Logger
from ..common.replay import Dataset
from ..common.utils import preprocess_image
from ..networks.value import Network as ValueNetwork
from ..training.policy_psdrl import PolicyTrainer
from ..common.settings import TP_THRESHOLD


class PolicyType(enum.Enum):
    EXPLOITATION = 0
    EXPLORATION = 1


class PSDRL:
    def __init__(
        self,
        config: dict,
        actions: list,
        logger: Logger,
        env_dim: int,
        device: str,
        random_state: np.random.RandomState,
        seed: int = None,
        env_model=EnvModel,
    ):
        self.device = device
        self.random_state = random_state

        self.num_actions = len(actions)
        self.actions = torch.tensor(actions).to(self.device)

        self.epsilon = config["algorithm"]["policy_noise"]
        self.epsilon_decay = config["algorithm"]["policy_noise_decay"]

        self.update_freq = config["algorithm"]["update_freq"]
        self.warmup_length = config["algorithm"]["warmup_length"]
        self.warmup_freq = config["algorithm"]["warmup_freq"]
        self.discount = config["value"]["discount"]

        self.dataset = Dataset(
            logger,
            config["replay"],
            config["experiment"]["time_limit"],
            self.device,
            config["visual"],
            env_dim,
            seed,
        )

        self.value_network = ValueNetwork(
            env_dim,
            config["value"],
            self.device,
            config["transition"]["gru_dim"],
        )

        self.model = env_model

        self.policy_trainer = PolicyTrainer(
            config["value"],
            config["transition"]["gru_dim"],
            self.value_network,
            self.device,
            config["replay"]["batch_size"],
            self.actions,
        )

        self.set_to_exploration()

    def set_to_exploration(self):
        self.policy = PolicyType.EXPLORATION

    def set_to_exploitation(self):
        self.policy = PolicyType.EXPLOITATION

    def select_action(self, obs: np.array, step: int):
        """
        Reset the hidden state at the start of a new episode. Return a random action with a probability of epsilon,
        otherwise follow the current policy and sampled model greedily.
        """
        if step == 0:
            self.model.reset_hidden_state()

        if self.random_state.random() < self.epsilon:
            return self.random_state.choice(self.num_actions)

        obs, is_image = preprocess_image(obs)
        obs = torch.from_numpy(obs).float().to(self.device)
        obs = self.model.embed_observation(obs)

        if self.policy == PolicyType.EXPLORATION:
            return self.explore(obs)
        else:
            return self.exploit(obs)

    def explore(self, obs: torch.tensor):
        """
        Return greedy action with respect to the current value network and all possible transitions predicted
        with the current sampled model (Equation 8).
        """
        states, rewards, terminals, h = self.model.exploration_policy(obs)

        v = self.discount * (
            self.value_network.predict(torch.cat((states, h), dim=1))
            * (terminals < TP_THRESHOLD)
        )
        values = (rewards + v).detach().cpu().numpy()
        action = self.random_state.choice(np.where(np.isclose(values, max(values)))[0])
        self.model.set_hidden_state(h[action])

        return self.actions[action]

    def exploit(self, obs: torch.tensor):
        """
        Return greedy action with respect to the current value network and all possible transitions predicted
        with the current sampled model (Equation 8).
        """
        states, rewards, terminals, h = self.model.exploitation_policy(obs)
        v = self.discount * (
            self.value_network.predict(torch.cat((states, h), dim=1))
            * (terminals < TP_THRESHOLD)
        )
        values = (rewards + v).detach().cpu().numpy()

        # reshape to num_predictions, actions, 1
        values = values.reshape((-1, self.num_actions, 1))

        # find max value action for each prediction
        candidate_actions = []
        for candidate in values:
            action = self.random_state.choice(
                np.where(np.isclose(candidate, max(candidate)))[0]
            )
            candidate_actions.append(action)

        # find consensus
        counts = np.bincount(np.array(candidate_actions))
        action = np.argmax(counts)

        self.model.set_hidden_state(h[action])
        return self.actions[action]

    def update(
        self,
        current_obs: np.array,
        action: int,
        rew: int,
        obs: np.array,
        done: bool,
        ep: int,
        timestep: int,
    ):
        """
        Add new transition to replay buffer and if it is time to update:
         - Update the representation model (Equation 1)
         - Update transition model (Equation 2) and terminal models (Equation 3).
         - Update posterior distributions model (Equation 4).
         - Sample new model from posteriors.
         - Update value network based on the new sampled model (Equation 5).
        """
        current_obs, obs = preprocess_image(current_obs)[0], preprocess_image(obs)[0]
        self.dataset.add_data(current_obs, action, obs, rew, done)
        update_freq = (
            self.update_freq if timestep > self.warmup_length else self.warmup_freq
        )

        self.epsilon = self.epsilon * self.epsilon_decay
        self.dataset.logger.add_scalars("General/epsilon", self.epsilon)

        if ep and timestep % update_freq == 0:
            self.model.train_(self.dataset)
            self.policy_trainer.train_(self.model, self.dataset)
