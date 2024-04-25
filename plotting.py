import argparse
import json
import os
from itertools import zip_longest
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
import shutil

sns.set_theme()
import math
import time


def get_color(idx):
    if idx < 10:
        linestyle = '-.'
    elif idx < 20:
        linestyle = '--'
    elif idx < 30:
        linestyle = '-'
    else:
        linestyle = 'dotted'
    return linestyle


def initialize_plots(envs, bins, w, h, suite, directory='./logdir/', ncols=3):
    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    n_plots = len(envs)        
    col_quant = ncols
    n_cols = col_quant if n_plots > col_quant - 1 else n_plots
    n_rows = int(np.ceil(n_plots / col_quant))
    fig, axs = plt.subplots(n_rows, int(n_cols), figsize=(w, h), sharex=True)
    bin = int(bins)
    x_range = list(range(0, 1000001, bin))

    if not os.path.exists("./plots"):
        os.mkdir("./plots")
    return fig, axs, x_range, bin, n_cols, n_plots


def initialize_subplot(col_quant, n_plots, idx, axs, env, n_cols):
    subt = col_quant
    if idx > col_quant - 1:
        subt = col_quant
    if n_plots == 1:
        ax = axs
    else:
        ax = axs[int(idx / col_quant)][idx - (int(idx / col_quant) * subt)] if n_plots > col_quant else axs[idx]
    ax.set_title(env, fontsize=22)
    if idx > n_plots - n_cols - 1 or n_plots < col_quant:
        ax.set_xlabel("Environment Steps", fontsize=18, labelpad=10)
    return ax


def extract_data(path, run, alg):
    run_rewards = []
    run_timesteps = []
    returnz = 'Reward/Train_Reward'
    try:

        with open(path + run + '/metrics.jsonl', 'r') as f:
            metrics = [json.loads(l) for l in f.readlines()]
            for k in metrics:
                if not math.isnan(k[returnz]):
                    if int(k[returnz]) > -99999:
                        run_rewards.append(float(k[returnz]))
                        run_timesteps.append(float(k['Timestep']))
    except:
        return None, None
    return run_rewards, run_timesteps


def bin_rewards(run_rewards, bin, run_timesteps, x_range, env):
    binned_rews = [run_rewards[0]]
    indexer = 0
    for idxi, b in enumerate(x_range[:-1]):

        subset_idx = [t for t in run_timesteps if t > b and t < b + bin]

        if len(subset_idx) == 0:
            if idxi == 0:
                continue
            else:
                break

        subset = run_rewards[indexer:indexer + len(subset_idx)]
        binned_rews.append(sum(subset) / len(subset))
        indexer += len(subset)
    return binned_rews


def plot(lengths, rewards, coloridx, n_runs, x_range, ax, label_alg):
    if len(lengths) < 1:
        return
    break_idx = max(lengths)
    mean = np.nanmean(np.array(list(zip_longest(*rewards)), dtype=float), axis=1) if len(rewards) > 1 else rewards[0]
    ax.plot(x_range[:break_idx], mean, label="{}".format(label_alg + '(' + str(n_runs) + ')'),
            linestyle=get_color(coloridx) if 'PSDRL' not in label_alg else '-')
    if len(rewards) > 1:
        std = np.nanstd(np.array(list(zip_longest(*rewards)), dtype=float), axis=1)
        cf = 1.96 * std / np.sqrt(len(x_range[:break_idx]))
    else:
        std = cf = np.zeros(len(mean))
    ax.fill_between(x_range[:break_idx], mean - cf, mean + cf, alpha=.1)
    ax.tick_params(axis='both', which='major', labelsize=17)
    ax.xaxis.offsetText.set_fontsize(11)
    ax.set_autoscaley_on(True)


def run_plotting(envs, bins, w, h, suite, directory, ncols=3):
    fig, axs, x_range, bin, n_cols, n_plots = initialize_plots(envs, bins, w, h, ncols)
    latex_table = ""
    results = {}
    for idx, env in enumerate(envs):
        results[env] = {}
        coloridx = 0
        env = str(env)
        env = env.replace(" ", "")
        if not os.path.exists(directory + env):
            continue
        ax = initialize_subplot(n_cols, n_plots, idx, axs, env, n_cols)
        env_dir = directory + env
        folders = sorted(os.listdir(env_dir))

        for algorithm in folders:
            if '0.001' not in algorithm and 'PSDRL' not in algorithm:
              continue
            path = env_dir + '/' + algorithm + "/"
            runs = os.listdir(path)
            rewards = []
            lengths = []
            n_runs = 0

            for run in runs:
                run_rewards, run_timesteps = extract_data(path, run, algorithm)
                if not run_rewards:
                    continue

                binned_rews = bin_rewards(run_rewards, bin, run_timesteps, x_range, env)

                lengths.append(len(binned_rews))
                rewards.append(binned_rews)
                n_runs += 1

            if n_runs > 0:
                mean_rewards = np.mean(np.nanmean(np.array(list(zip_longest(*rewards)), dtype=float), axis=1))
                std_dev = np.std(np.nanstd(np.array(list(zip_longest(*rewards)), dtype=float), axis=1))
                stderr = std_dev / np.sqrt(n_runs)
                results[env][algorithm] = {'mean_rew' : mean_rewards, 'stderror' : stderr}
                
                
                





            plot(lengths, rewards, coloridx, n_runs, x_range, ax, algorithm)
            coloridx += 1

        if idx == 0:
            fig.legend(ncol=2, loc='upper center', fontsize=18)

    plt.tight_layout()
    fig.subplots_adjust(top=0.80)
    fig.savefig("./plots/{}-plots.pdf".format(envs))
    top_row = "\textbf{Game}" + " & "
    for alg in results[list(results.keys())[0]]:
        top_row += '\\textbf{' + (alg.split('temp')[1] if 'PSDRL' not in alg else 'PSDRL') + "} & "
    top_row += '\\\\'
    print(top_row)
    for env in results.keys():
      game_row = env + " & "
      for alg in results[env].keys():
          game_row += str(round(results[env][alg]['mean_rew'])) + " & "
      
      print(game_row + '\\\\')
      

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--plot_bin", default=29000, type=int)
    parser.add_argument("--suite", default="atari", type=str)
    parser.add_argument("--fig_size_w", default=20, type=int)
    parser.add_argument("--fig_size_h", default=16, type=int)
    parser.add_argument("--directory", default="/data/scratch/acw638/k-logdir/", type=str)
    parser.add_argument("--ncols", default=3, type=int)
    args = parser.parse_args()
    envs = ["Asterix", "Enduro", "Freeway", "Hero", "Qbert", "Seaquest", "Pong"]
    run_plotting(envs, args.plot_bin, args.fig_size_w, args.fig_size_h, args.suite, args.directory,
                 args.ncols)

