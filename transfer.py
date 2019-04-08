import os
from pathlib import Path
import gym
import numpy as np
from collections import ChainMap
from src.ddpg_round import DDPGRound
from src.averaging_round_manager import AveragingRoundManager
from src.comet_logger import CometLogger
from src.settings import EXPERIMENTS_PATH

import torch


def run(source_experiment, max_frames=5000, g=10, seed=3):
    params = DDPGRound.defaults()
    params['source_experiment'] = source_experiment
    params['project'] = 'transfer'
    params['algo'] = 'DDPG'
    params['env'] = 'GravityPendulum-v0'
    params['seed'] = seed
    params['max_frames'] = max_frames
    params['g'] = g
    experiment = CometLogger(project=params['project']).experiment()
    experiment.log_parameters(params)

    trailing_avg = (np.zeros(5) - 1500).tolist() # prime the list for averaging

    def round_done(id, frames, reward):
        experiment.log_metric('reward', reward, step=frames)
        trailing_avg.append(reward)
        experiment.log_metric('trailing_avg_5', np.mean(trailing_avg[-5]),step=frames)

    params['on_round_done'] = round_done

    value_state_dict = torch.load(EXPERIMENTS_PATH / source_experiment / 'value_net.mdl')
    policy_state_dict = torch.load(EXPERIMENTS_PATH / source_experiment / 'policy_net.mdl')

    single_learner = DDPGRound(params)
    single_learner.policy_net.load_state_dict(policy_state_dict)
    single_learner.value_net.load_state_dict(value_state_dict)
    single_learner.run()
