import torch
from collections import ChainMap
from src.ddpg_round import DDPGRound
from src.averaging_round_manager import AveragingRoundManager
from src.comet_logger import CometLogger
from src.settings import EXPERIMENTS_PATH

def run(num_episodes=1,
        num_rounds=1,
        seed=1,
        mp=True,
        env_params=(8.,9.,10.,11.,12.,),
        env_name='GravityPendulum',
        fame_reg=True):

    distral=False
    params = DDPGRound.defaults()
    params['env'] = env_name
    params['seed'] = seed
    params['num_episodes'] = num_episodes
    params['env_params'] = env_params
    params['project'] = f'DDPG-{env_name}'
    params['distral'] = distral
    params['device'] = 'cpu'
    params['fame_regularize'] = fame_reg


    if fame_reg:
        tag = 'fame'
    else:
        tag = 'baseline'


    experiment = CometLogger(project=params['project']).experiment()
    experiment.add_tag(tag)
    node_params = [ChainMap({'id':f'n{idx}', 'experiment': experiment,  'env_param': ep }, params) for idx, ep in enumerate(env_params) ]
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
        'multiprocess': mp,
        'alpha': 0,
        'beta': 0,
    }
    experiment.log_parameters(params)
    experiment.log_parameters(experiment_params)

    round_manager = AveragingRoundManager(node_params, experiment_params)
    round_manager.run_rounds()
