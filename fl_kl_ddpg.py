from collections import ChainMap
from src.ddpg_round import DDPGRound


from src.averaging_round_manager import AveragingRoundManager
from src.comet_logger import CometLogger
from src.settings import EXPERIMENTS_PATH

def run(max_frames=800, alpha=0.5, beta=0.5, num_rounds=12, seed=3, gravities=(10.)):
    params = DDPGRound.defaults()
    params['algo'] = 'DDPG'
    params['env'] = 'GravityPendulum-v0'
    params['seed'] = seed
    params['max_frames'] = max_frames
    params['gravities'] = gravities
    params['project'] = 'ddpg-fl-kl'
    node_params = [ChainMap({'id':f'n{idx}', 'g': g }, params) for idx,g in enumerate(gravities) ]

    experiment = CometLogger(project=params['project']).experiment()
    experiment_params = {
        'experiment': experiment,
        'num_rounds': num_rounds,
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
