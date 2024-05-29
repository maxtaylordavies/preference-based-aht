import time

import gym
from tqdm import tqdm
from stable_baselines3 import PPO

# from preference_based_aht.agents import RandomAgent
from lbforaging.agents import RandomAgent
from preference_based_aht.buffer import *

SEED = 0

# create environment
env = gym.make("ahtlbf-8x8-2p-3f-v2")

# create agents
aht_agent = env.create_aht_agent("idle")
env.set_npc_agents(["closest_food"])

# training loop
total_steps, terminated, obs = 0, True, None
pbar = tqdm(total=1e5)
while total_steps < 1e5:
    if terminated:
        obs = env.reset()
        terminated = False

    env.render()
    time.sleep(0.1)

    # sample action from aht agent
    action = aht_agent.act(obs)

    # take step in environment
    next_obs, r, terminated, _ = env.step(action)

    # store transition in replay buffer

    # update agent(s)
    # aht_agent.update()

    obs = next_obs
    total_steps += 1
    pbar.update(1)
