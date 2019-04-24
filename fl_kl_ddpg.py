from collections import ChainMap
from src.ddpg_round import DDPGRound


from src.averaging_round_manager import AveragingRoundManager
from src.comet_logger import CometLogger
from src.settings import EXPERIMENTS_PATH

def run(max_frames=800, env='GravityPendulum', fame_reg=False, alpha=0.5, beta=0.5, num_rounds=12, seed=3, env_params=(10.)):
    params = DDPGRound.defaults()
    params['env'] = env
    params['seed'] = seed
    params['max_frames'] = max_frames
    params['env_params'] = env_params
    params['project'] = 'ddpg-fl-kl'
    node_params = [ChainMap({'id':f'n{idx}', 'env_param': env_param }, params)
            for idx, env_param in enumerate(env_params) ]

    experiment = CometLogger(project=params['project']).experiment()
    experiment_params = {
        'algo': params['algo'],
        'env': params['env'],
        'experiment': experiment,
        'num_rounds': num_rounds,
        'fame_regularize': fame_reg,
        'experiment_path': EXPERIMENTS_PATH / experiment.id,
        'shared_replay': False,
        'multiprocess': True,
        'alpha': float(alpha),
        'beta': float(beta),
    }
    experiment.log_parameters(params)
    experiment.log_parameters(experiment_params)

    round_manager = AveragingRoundManager(node_params, experiment_params)
    round_manager.run_rounds()
