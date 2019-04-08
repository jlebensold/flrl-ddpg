import gym
import numpy as np

from src.comet_logger import CometLogger
from collections import ChainMap
from src.ddpg_round import DDPGRound


def run(max_frames=5000, g=10, seed=3):
    params = DDPGRound.defaults()
    params['algo'] = 'DDPG'
    params['env'] = 'GravityPendulum-v0'
    params['seed'] = seed
    params['max_frames'] = max_frames
    params['g'] = g
    params['id'] = 'baseline'

    experiment = CometLogger(project='ddpg-baselines').experiment()
    experiment.log_parameters(params)

    trailing_avg = (np.zeros(5) - 1500).tolist() # prime the list for averaging
    def round_done(id, frames, reward):
        experiment.log_metric('reward', reward, step=frames)
        trailing_avg.append(reward)
        experiment.log_metric('trailing_avg_5', np.mean(trailing_avg[-5]),step=frames)

    params['on_round_done'] = round_done


    print("Baselines - no FL")
    single_learner = DDPGRound(params)
    single_learner.run()
