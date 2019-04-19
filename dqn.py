from collections import ChainMap
from src.dqn_round import DQNRound


from src.averaging_round_manager import AveragingRoundManager
from src.comet_logger import CometLogger
from src.settings import EXPERIMENTS_PATH

def run(max_frames=500, env='MountainCarDiscrete', num_rounds=10, seed=3, env_params=(10.,)):
    params = DQNRound.defaults()
    params['env'] = env
    params['seed'] = seed
    params['max_frames'] = max_frames
    params['env_params'] = env_params
    params['project'] = 'ddpg-fl-kl'
    node_params = [ChainMap({'id':f'n{idx}', 'env_param': ep }, params) for idx, ep in enumerate(env_params) ]
    dqn_round = DQNRound(node_params[0])
    dqn_round.run()

