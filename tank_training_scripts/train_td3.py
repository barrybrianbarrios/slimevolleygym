#!/usr/bin/env python3

# Train single CPU PPO1 on slimevolley.
# Should solve it (beat existing AI on average over 1000 trials) in 3 hours on single CPU, within 3M steps.

import gym
import numpy as np
# import slimevolleygym
# from slimevolleygym import SurvivalRewardEnv
import tankgym

from stable_baselines.ppo2 import PPO2
from stable_baselines import TD3
from stable_baselines.td3.policies import MlpPolicy
from stable_baselines.ddpg.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines import logger
from stable_baselines.common.callbacks import EvalCallback
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines.bench import Monitor

NUM_TIMESTEPS = int(2e7)
SEED = 721
EVAL_FREQ = 1e6
EVAL_EPISODES = 100
LOGDIR = "tank_td3" # moved to zoo afterwards.

logger.configure(folder=LOGDIR)

env = gym.make("TankGym-v0")
env.seed(SEED)
env.policy = tankgym.BaselineRand()

# The noise objects for TD3
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

model = TD3(MlpPolicy, env, action_noise=action_noise, verbose=1)

# eval_callback = EvalCallback(env, best_model_save_path=LOGDIR, log_path=LOGDIR, eval_freq=EVAL_FREQ, n_eval_episodes=EVAL_EPISODES)

model.learn(total_timesteps=NUM_TIMESTEPS, log_interval=10)

model.save(os.path.join(LOGDIR, "final_model")) # probably never get to this point.

env.close()
