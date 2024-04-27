import os
import argparse

import numpy as np
from ruamel.yaml import YAML
import gym
import wandb
from PSDRL.common.data_manager import DataManager
from PSDRL.common.utils import init_env, load
from PSDRL.common.logger import Logger
from PSDRL.agent import Agent

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"


def run_test_episode(env: gym.Env, agent: Agent, time_limit: int):
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


def run_experiment(
    env: gym.Env,
    agent: Agent,
    logger: Logger,
    test_env: gym.Env,
    steps: int,
    test: int,
    test_freq: int,
    time_limit: int,
    save: bool,
    save_freq: int,
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
    )


def run_on_seed(args):
    with open(args.config, "r") as f:
        yaml = YAML(typ="rt")
        config = yaml.load(f)

        config["experiment"]["env"] = args.env
        config["experiment"]["seed"] = args.seed
        config["experiment"][
            "name"
        ] = f"DeepSea ({args.env}) - {config['experiment']['name']} - {args.seed}"
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
