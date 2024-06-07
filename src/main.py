from io import BytesIO
import os
import argparse

from matplotlib import pyplot as plt
import numpy as np
from ruamel.yaml import YAML
import gym
import torch
from PSDRL.common.replay import Dataset
import wandb
from PSDRL.common.data_manager import DataManager
from PSDRL.common.utils import init_env, load, preprocess_image
from PSDRL.common.logger import Logger
from PSDRL.agent.psdrl import PSDRL
from PSDRL.agent import Agent

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_test_episode(env: gym.Env, agent: PSDRL, time_limit: int):
    agent.set_to_exploitation()
    current_observation = env.reset()
    episode_step = 0
    episode_reward = 0
    done = False
    while not done:
        action = agent.select_action(current_observation, episode_step)
        observation, reward, done, _ = env.step(action)
        episode_reward += reward
        current_observation = observation
        episode_step += 1
        done = done or episode_step == time_limit
    return episode_reward


import seaborn as sns
from PIL import Image


def plot_value(size: int, agent: PSDRL, logger: Logger, timestep: int):
    def build_all_states():
        states = []

        def build_states_for_time(time):
            states = []
            for pos in range(size):
                state = []
                for _ in range(time):
                    state.append([0 for _ in range(size)])

                state.append([0 if i != pos else 1 for i in range(size)])

                for _ in range(size - time - 1):
                    state.append([0 for _ in range(size)])

                states.append(np.array(state).flatten())

            return states

        for time in range(size):
            states += build_states_for_time(time)

        return torch.FloatTensor(states).to(agent.device)

    states = build_all_states()
    h = torch.zeros((len(states), agent.model.transition_network.gru_dim)).to(
        agent.model.device
    )
    v = agent.discount * (agent.value_network.predict(torch.cat((states, h), dim=1)))
    values = v.view((size, size)).detach().cpu().numpy()

    # remove impossible states
    for time in range(len(values)):
        for j in range(time + 1, size):
            values[time, j] = 0

    # plot heatmap and save to pil image
    plt.close()
    ax = sns.heatmap(values, linewidth=0.5)
    buf = BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img = Image.open(buf)

    # log
    logger.log_image(timestep, img, "Value")


def log_correct_path(env: gym.Env, agent: PSDRL):
    def get_right_action():
        row = env._row
        col = env._column
        return env._action_mapping[row, col]

    agent.model.reset_hidden_state()
    obs = env.reset()
    for time in range(env._size):
        right_a = get_right_action()
        print(right_a)

        obs, is_image = preprocess_image(obs)
        obs = torch.from_numpy(obs).float().to(agent.device)
        obs = agent.model.embed_observation(obs)
        states, rewards, terminals, h = agent.model.exploration_policy(obs)

        obs, reward, done, _ = env.step(right_a)

        pred_state = states[right_a]
        pred_state = pred_state.detach().cpu().numpy().reshape((env._size, env._size))
        pred_rew = rewards[right_a]
        pred_terminals = terminals[right_a]

        agent.model.set_hidden_state(h[right_a])

        print(f"Time {time}:")
        print(
            f"{reward},{done} {'   '*env._size}{pred_rew[0].detach().cpu().numpy()}, {pred_terminals[0].detach().cpu().numpy()}"
        )
        for act, pred in zip(obs, pred_state):
            print(act, pred)


def early_stop(early_stop_config, dataset: Dataset) -> bool:
    if not early_stop_config["enabled"]:
        return False

    n_episodes = early_stop_config["n_episodes"]
    last_n_episodes = dataset.episodes[-n_episodes:]

    episode_returns = [ep["cum_rew"] for ep in last_n_episodes]
    av_return = sum(episode_returns) / n_episodes

    return av_return >= early_stop_config["threshold"]


def run_experiment(
    env: gym.Env,
    agent: PSDRL,
    logger: Logger,
    test_env: gym.Env,
    steps: int,
    test: int,
    test_freq: int,
    time_limit: int,
    save: bool,
    save_freq: int,
    early_stop_config,
):
    ep = 0
    experiment_step = 0

    while experiment_step < steps:
        episode_step = 0
        episode_reward = 0

        current_observation = env.reset()
        done = False
        while not done:

            if test and experiment_step % test_freq == 0:
                test_reward = run_test_episode(test_env, agent, time_limit)
                logger.log_episode(
                    experiment_step, train_reward=np.nan, test_reward=test_reward
                )
                print(
                    f"Episode {ep}, Timestep {experiment_step}, Test Reward {test_reward}"
                )

                plot_value(env._size, agent, logger, experiment_step)
                log_correct_path(test_env, agent)

            agent.set_to_exploration()
            action = agent.select_action(current_observation, episode_step)
            observation, reward, done, _ = env.step(action)
            done = done or episode_step == time_limit
            agent.update(
                current_observation,
                action,
                reward,
                observation,
                done,
                ep,
                experiment_step,
            )

            episode_reward += reward
            current_observation = observation
            episode_step += 1
            experiment_step += 1

            if ep and save and experiment_step % save_freq == 0:
                logger.data_manager.save(agent, experiment_step)
        print(
            f"Episode {ep}, Timestep {experiment_step}, Train Reward {episode_reward}"
        )

        ep += 1
        logger.log_episode(
            experiment_step, train_reward=episode_reward, test_reward=np.nan
        )

        if early_stop(early_stop_config, agent.dataset):
            break


def main(config: dict):
    data_manager = DataManager(config)
    logger = Logger(data_manager)
    exp_config = config["experiment"]

    env, actions, test_env = init_env(
        exp_config["suite"], exp_config["env"], exp_config["test"], exp_config["seed"]
    )

    agent = Agent(
        config,
        actions,
        logger,
        (
            config["representation"]["embed_dim"]
            if config["visual"]
            else np.prod(env.observation_spec().shape)
        ),
        config["experiment"]["seed"],
    )
    if config["load"]:
        load(agent, config["load_dir"])

    run_experiment(
        env,
        agent,
        logger,
        test_env,
        exp_config["steps"],
        exp_config["test"],
        exp_config["test_freq"],
        exp_config["time_limit"],
        config["save"],
        config["save_freq"],
        config["early_stop"],
    )


def run_on_seed(args):
    with open(args.config, "r") as f:
        yaml = YAML(typ="rt")
        config = yaml.load(f)

        config["experiment"]["env"] = args.env
        config["experiment"]["seed"] = args.seed
        config["experiment"][
            "name"
        ] = f"DeepSea ({args.env}) - {config['algorithm']['name']} - {args.seed}"
        if config["experiment"]["suite"] == "bsuite":
            config["replay"]["sequence_length"] = int(args.env)
    main(config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=str, default="./configs/config_psdrl_vector.yaml"
    )
    parser.add_argument(
        "--env",
        type=str,
        default="3",
        help="Currently if you put an integer it makes DeepSea with the size of that integer.",
    )
    parser.add_argument("--seed", type=int, nargs="+", default=None)
    parser.add_argument("--experiment_name", type=str, default="")

    args = parser.parse_args()
    envs = args.env
    for seed in args.seed:
        args.seed = seed
        run_on_seed(args)
        wandb.finish()
