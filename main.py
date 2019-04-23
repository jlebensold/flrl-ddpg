import fire

from baselines import run as baselines
from ddpg import run as ddpg
from fl_kl_ddpg import run as fl_kl_ddpg
from dqn import run as dqn
from transfer import run as transfer

if __name__ == '__main__':
    fire.Fire()
