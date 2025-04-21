"""
This file uses Stable Baselines 3 to set up and train a PPO agent on the LanderEnv environment.
Training progress and metrics are logged to Weights & Biases (wandb).
The environment LanderEnv is defined in the module 'lander_env'.
"""

import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import wandb
from src.env.lander_env import Lander4DOFEnv
import numpy as np
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.vec_env import VecVideoRecorder

# from lander_env import LanderEnv  # make sure your environment is in this module
from stable_baselines3.common.env_util import make_vec_env

# wandb.init(project="planetary_landing")

#as defined by furafo et al
#observation space is shape (11, )
#action space is shape (5, )

gym.envs.register(
        id='Lander4DOF-v0',
        entry_point=Lander4DOFEnv
    )

env = gym.make('Lander4DOF-v0')
env = TimeLimit(env, max_episode_steps=200000)

# #run 100 episode with random actions, calculate mean episodic reward
# def run_random_policy(env, num_episodes=100):
#     total_reward = 0
#     for episode in range(num_episodes):
#         obs, _ = env.reset()
#         terminated, truncated = False, False
#         mean_reward = 0
#         steps = 0
#         while not terminated and not truncated:
#             action = env.action_space.sample()  # Random action
#             obs, reward, terminated, truncated, _ = env.step(action)
#             mean_reward += reward
#             steps += 1
#         print(f"Episode {episode + 1}: mean reward per step: {mean_reward/steps}")
#     return mean_reward
# mean_reward = run_random_policy(env)
# print(f"Mean episodic reward over {100} episodes with random policy: {mean_reward}")

# raise
policy_kwargs = dict(activation_fn=th.nn.Tanh,
                     net_arch=dict(pi=[110, 74, 50], vf=[110, 74, 5]))

# gym.envs.register(
#         id='Lander4DOF-v0',
#         entry_point='lander_env:Lander4DOFEnv'
#     )

env = make_vec_env("Lander4DOF-v0", n_envs=4)
env = VecVideoRecorder(env, "videos", record_video_trigger=lambda x: x % 100000 == 0, video_length=50000, name_prefix="lander4dof")

model = PPO("MlpPolicy", env, policy_kwargs=policy_kwargs, verbose=1, gamma=0.99, gae_lambda=0.95, tensorboard_log='./tensorboard_log/', device='cpu')
model.learn(total_timesteps=1e6, progress_bar=True)
model.save("lander4dof_model")
# wandb.finish()s