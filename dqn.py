from collections import ChainMap
from src.dqn_round import DQNRound

import torch

from src.averaging_round_manager import AveragingRoundManager
from src.comet_logger import CometLogger
from src.settings import EXPERIMENTS_PATH

def run(max_episodes=1, env='GridworldEnv', fame_reg=False, distral=True,
        alpha=0.8, beta=5, num_rounds=200, seed=3,
        env_params=(5,4,6)):
    params = DQNRound.defaults()
    params['env'] = env
    params['seed'] = seed
    params['max_episodes'] = max_episodes
    params['env_params'] = env_params
    params['alpha'] = alpha
    params['beta'] = beta
    params['project'] = 'dqn-fl-kl'
    #params['device'] = 'cuda:0'
    params['device'] = 'cpu'
    params['distral'] = distral
    node_params = [ChainMap({'id':f'n{idx}', 'env_param': ep }, params) for idx, ep in enumerate(env_params) ]

    experiment = CometLogger(project=params['project']).experiment()
    experiment_params = {
        'algo': params['algo'],
        'env': params['env'],
        'experiment': experiment,
        'num_rounds': num_rounds,
        'device': torch.device(params['device']),
        'fame_regularize': fame_reg,
        'distral': distral,
        'experiment_path': EXPERIMENTS_PATH / experiment.id,
        'multiprocess': True,
        'alpha': float(alpha),
        'beta': float(beta),
    }
    experiment.log_parameters(params)
    experiment.log_parameters(experiment_params)

    round_manager = AveragingRoundManager(node_params, experiment_params)
    round_manager.run_rounds()
