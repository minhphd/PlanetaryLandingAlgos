import gymnasium as gym
import wandb
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
# from lander_env import LanderEnv  # make sure your environment is in this module
from stable_baselines3.common.env_util import make_vec_env

# wandb.init(project="planetary_landing")
gym.envs.register(
        id='Lander4DOF-v0',
        entry_point='lander_env:Lander4DOFEnv'
    )

env = gym.make("Lander4DOF-v0", render_mode="human")
model = PPO.load("lander4dof_model")

obs, _ = env.reset()
while True:
    action, _states = model.predict(obs)
    observation, reward, terminated, truncated, info = env.step(action)
    env.render()
