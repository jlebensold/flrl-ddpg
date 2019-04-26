import fire
import comet_ml
from ddpg import run as ddpg
from dqn import run as dqn
from transfer import run as transfer

if __name__ == '__main__':
    fire.Fire()
