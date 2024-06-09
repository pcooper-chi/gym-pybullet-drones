# Copied from:
# https://github.com/Ahmad-Jarrar/gym-pybullet-drones/blob/e1362616c2f3b0ada1547950f6d08679ae76febb/gym_pybullet_drones/examples/visualize.py

import os
import time
from datetime import datetime
import argparse
import gymnasium as gym
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.evaluation import evaluate_policy

from gym_pybullet_drones.utils.Logger import Logger
from gym_pybullet_drones.utils.utils import sync, str2bool
from gym_pybullet_drones.utils.enums import ObservationType, ActionType

from gym_pybullet_drones.envs.HoverAviary import HoverAviary

import pybullet as p

DEFAULT_GUI = True
DEFAULT_RECORD_VIDEO = True
DEFAULT_OUTPUT_FOLDER = 'results'
DEFAULT_COLAB = False

DEFAULT_OBS = ObservationType('rgb') # 'kin' or 'rgb'
DEFAULT_ACT = ActionType('one_d_rpm') # 'rpm' or 'pid' or 'vel' or 'one_d_rpm' or 'one_d_pid'

def run(output_folder=DEFAULT_OUTPUT_FOLDER, gui=DEFAULT_GUI, plot=True, colab=DEFAULT_COLAB, record_video=DEFAULT_RECORD_VIDEO, local=True):

    # p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
    env = HoverAviary(obs=DEFAULT_OBS, act=DEFAULT_ACT, ctrl_freq=24, gui=gui, record=record_video)


    obs = env.reset()
    start = time.time()
    for i in range((30)*env.CTRL_FREQ):
        # action = np.array([[0., 0.4, 0., 1.]])
        action = np.array([[-0.01]])
        obs, reward, terminated, truncated, info = env.step(action)
        print(reward)
        # env.render()
        sync(i, start, env.CTRL_TIMESTEP)
        if terminated:
            obs = env.reset()
    env.close()

if __name__ == '__main__':
    #### Define and parse (optional) arguments for the script ##
    parser = argparse.ArgumentParser(description='Single agent reinforcement learning example script')
    # parser.add_argument('--multiagent',         default=DEFAULT_MA,            type=str2bool,      help='Whether to use example LeaderFollower instead of Hover (default: False)', metavar='')
    parser.add_argument('--gui',                default=DEFAULT_GUI,           type=str2bool,      help='Whether to use PyBullet GUI (default: True)', metavar='')
    parser.add_argument('--record_video',       default=DEFAULT_RECORD_VIDEO,  type=str2bool,      help='Whether to record a video (default: False)', metavar='')
    parser.add_argument('--output_folder',      default=DEFAULT_OUTPUT_FOLDER, type=str,           help='Folder where to save logs (default: "results")', metavar='')
    ARGS = parser.parse_args()

    run(**vars(ARGS))