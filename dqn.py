from collections import ChainMap
from src.dqn_round import DQNRound

import torch

from src.round_manager import RoundManager
from src.comet_logger import CometLogger
from src.settings import EXPERIMENTS_PATH

def run(num_episodes=1, env_name='GridworldEnv', device='cpu',
        mp=True,
        fame_reg=False,
        distral=True,
        alpha=0.5,
        beta=5,
        num_rounds=1,
        seed=1,
        env_params=(5,4,6)):
    params = DQNRound.defaults()
    params['env_name'] = env_name
    params['env_params'] = env_params
    params['seed'] = seed
    params['num_episodes'] = num_episodes
    params['alpha'] = alpha
    params['beta'] = beta
    params['project'] = f'DQN-{env_name}'
    params['device'] = device
    params['distral'] = distral


    if distral:
        tag = 'distral'
    elif fame_reg:
        tag = 'fame'
    else:
        tag = 'baseline'

    experiment = CometLogger(project=params['project']).experiment()
    experiment.add_tag(tag)
    node_params = [ChainMap({'id':f'n{idx}', 'experiment': experiment,  'env_param': ep }, params) for idx, ep in enumerate(env_params) ]
    experiment_params = {
        'algo': params['algo'],
        'env_name': env_name,
        'experiment': experiment,
        'num_rounds': num_rounds,
        'device': torch.device(params['device']),
        'fame_regularize': fame_reg,
        'distral': distral,
        'experiment_path': EXPERIMENTS_PATH / experiment.id,
        'multiprocess': mp,
        'alpha': float(alpha),
        'beta': float(beta),
    }
    experiment.log_parameters(params)
    experiment.log_parameters(experiment_params)

    round_manager = RoundManager(node_params, experiment_params)
    round_manager.run_rounds()
