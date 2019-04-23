import torch
from collections import ChainMap
from src.ddpg_round import DDPGRound
from src.averaging_round_manager import AveragingRoundManager
from src.comet_logger import CometLogger
from src.settings import EXPERIMENTS_PATH

def run(max_frames=500, num_rounds=10, seed=3, env_params=(8.,9.,10.,11.,12.,),
        env_name='GravityPendulum', fame_reg=True):
    distral=False
    params = DDPGRound.defaults()
    params['algo'] = 'DDPG'
    params['env'] = env_name
    params['seed'] = seed
    params['max_frames'] = max_frames
    params['env_params'] = env_params
    params['project'] = 'ddpg-fl-averaging'
    params['distral'] = distral
    params['device'] = 'cpu'
    params['fame_regularize'] = fame_reg

    node_params = [ChainMap({'id':f'n{idx}', 'env_param': g }, params) for idx,g in enumerate(env_params) ]

    experiment = CometLogger(project=params['project']).experiment()
    experiment_params = {
        'experiment': experiment,
        'num_rounds': num_rounds,
        'algo': params['algo'],
        'env': params['env'],
        'env_params': params['env_params'],
        'fame_regularize': fame_reg,
        'device': torch.device(params['device']),
        'distral': distral,
        'experiment_path': EXPERIMENTS_PATH / experiment.id,
        'multiprocess': True,
        'alpha': 0,
        'beta': 0,
    }
    experiment.log_parameters(params)
    experiment.log_parameters(experiment_params)

    round_manager = AveragingRoundManager(node_params, experiment_params)
    round_manager.run_rounds()
