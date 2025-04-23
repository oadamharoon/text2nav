import stable_baselines3 as sb3
from stable_baselines3.common.env_checker import check_env
from robot_env import JetbotEnv
import numpy as np
import gymnasium as gym

env = JetbotEnv()
# print(env.action_space)
# print(env.observation_space)
# print(isinstance(env.action_space, gym.spaces.Box))
# print(isinstance(env.observation_space, gym.spaces.Box))
check_env(JetbotEnv(), warn=True)