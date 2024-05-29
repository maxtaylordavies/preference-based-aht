from functools import partial
import time

import gym
import numpy as np
from tqdm import tqdm
from stable_baselines3 import PPO, DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env as _make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecEnv,
    VecMonitor,
    is_vecenv_wrapped,
    sync_envs_normalization,
)
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from preference_based_aht.environment import AHTLBFEnv

sns.set_theme(style="darkgrid")

SEED = int(time.time())
print(f"Seed: {SEED}")

hyperparams = {
    "normalize_advantage": True,
    "gae_lambda": 0.95,
    "gamma": 0.99,
    "n_steps": 512,
    "batch_size": 512,
    "n_epochs": 4,
    "ent_coef": 0.01,
    "learning_rate": 0.0003,
    "clip_range": 0.2,
}

env_id = "ahtlbf-8x8-2p-3f-v2"


def make_env():
    env = gym.make(env_id)
    env.seed(SEED)
    env.set_npc_agents(["closest_food"])
    return env


def make_vec_env(n_envs=1, render_mode=None):
    env = _make_vec_env(make_env, n_envs=n_envs, seed=SEED)
    env.render_mode = render_mode
    return env


# create train and eval environments
train_env = make_vec_env()

# create PPO model
model = PPO("MlpPolicy", train_env, verbose=1, **hyperparams)


def play_episode(env, model, render=False):
    obs, done, ret, length = env.reset(), False, 0, 0
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _ = env.step(action)
        ret, length = ret + reward, length + 1
        if render:
            env.render(mode="human")
            time.sleep(0.1)
    return ret, length


# evaluation function
def run_eval(model, n_episodes=1):
    env = make_env()
    returns, lengths = [], []

    for ep_idx in range(n_episodes):
        ret, length = play_episode(env, model)
        returns.append(ret)
        lengths.append(length)

    env.close()
    del env
    return returns, lengths


# visualisation function
def visualise_policy(model, n_episodes=1):
    env = make_env()
    for ep_idx in range(n_episodes):
        play_episode(env, model, render=True)
    env.close()
    del env


# training
t_done, max_t = 0, 1e6
eval_interval, last_eval, n_eval_eps = 20000, -1e9, 20
vis_interval, last_vis, n_vis_eps = eval_interval, -1e9, 1
eval_metrics = {"t": [], "returns": [], "lengths": []}
pbar = tqdm(
    total=max_t, initial=0, position=0, leave=True, desc="Training", unit="steps"
)

_, callback = model._setup_learn(max_t, None, True, "", False)

while t_done < max_t:
    if t_done - last_eval >= eval_interval:
        last_eval = t_done
        rets, lengths = run_eval(model, n_episodes=n_eval_eps)
        eval_metrics["t"].append(t_done)
        eval_metrics["returns"].append(rets)
        eval_metrics["lengths"].append(lengths)
        tqdm.write(f"Mean eval reward, ep length: {np.mean(rets)}, {np.mean(lengths)}")

    if t_done - last_vis >= vis_interval:
        last_vis = t_done
        visualise_policy(model, n_episodes=n_vis_eps)

    model.collect_rollouts(
        train_env,
        callback,
        model.rollout_buffer,
        n_rollout_steps=hyperparams["n_steps"],
    )
    model.train()
    t_done += hyperparams["n_steps"]
    pbar.update(hyperparams["n_steps"])

eval_metrics = {k: np.array(v).flatten() for k, v in eval_metrics.items()}
eval_metrics["t"] = np.repeat(eval_metrics["t"], n_eval_eps)
df = pd.DataFrame(eval_metrics)


fig, axs = plt.subplots(2, 1, sharex=True)
sns.lineplot(data=df, x="t", y="returns", ax=axs[0])
sns.lineplot(data=df, x="t", y="lengths", ax=axs[1])
plt.show()
