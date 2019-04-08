import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from collections import ChainMap

from src.replay_buffer import ReplayBuffer
from src.ou_noise import OUNoise
from src.networks import PolicyNetwork, ValueNetwork
from src.ddpg_round import DDPGRound
from gym.envs.registration import register as gym_register


import os
from pathlib import Path
import multiprocessing
from multiprocessing.pool import ThreadPool
import threading

class AveragingRoundManager:
    def __init__(self, node_params=[], experiment_params={}):
        self.nodes = [ DDPGRound(param) for param in node_params ]
        self.num_nodes = len(self.nodes)
        self.experiment = experiment_params['experiment']
        self.num_rounds = experiment_params['num_rounds']
        self.multiprocess= experiment_params['multiprocess']
        self.shared_replay = experiment_params['shared_replay']
        self.experiment_path = experiment_params['experiment_path']
        os.makedirs(self.experiment_path, exist_ok=True)


    def init_value_and_policy_params(self):
        policy_params = dict()
        value_params = dict()

        for name, param in self.nodes[0].value_net.named_parameters():
            value_params[name] = torch.zeros_like(param)

        for name, param in self.nodes[0].policy_net.named_parameters():
            policy_params[name] = torch.zeros_like(param)

        return value_params, policy_params

    def add_node_weights_to_network_params(self, node, value_params, policy_params):

        for name, tensor in node.value_net.named_parameters():
            value_params[name] += tensor

        for name, tensor in node.policy_net.named_parameters():
            policy_params[name] += tensor

        return value_params, policy_params


    def average_node_network_weights(self, value_params, policy_params):
        for node in self.nodes:
            value_params, policy_params = self.add_node_weights_to_network_params(node, value_params, policy_params)

        # average:
        for name, param in value_params.items():
            value_params[name] /= self.num_nodes

        for name, param in policy_params.items():
            policy_params[name] /= self.num_nodes

        return value_params, policy_params


    def run_rounds(self):

        trailing_avg = {}
        for node in self.nodes:
            trailing_avg[node.id] = (np.zeros(5) - 1500).tolist() # prime the list for averaging


        for idx in range(self.num_rounds):
            value_params, policy_params = self.init_value_and_policy_params()
            round_num = idx + 1
            print(f'Round {round_num}')
            if self.multiprocess:
                pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)
                results =  pool.starmap(job, [(node,) for node in self.nodes])
                round_reward = []
                for result in results:
                    node_idx = next(idx for idx, node in enumerate(self.nodes) if node.id == result['id'])
                    id = result['id']
                    self.nodes[node_idx].value_net.load_state_dict(result['value_net'])
                    self.nodes[node_idx].policy_net.load_state_dict(result['policy_net'])
                    self.nodes[node_idx].total_frames = result['total_frames']
                    round_reward.append(result['rewards'])
                    trailing_avg[id].append(result['rewards'][-1])


                    self.experiment.log_metric(f'reward.{id}', result['rewards'][-1], step=result['total_frames'])
                    self.experiment.log_metric(f'trailing_avg_5.{id}', np.mean(trailing_avg[id][-5]),step=result['total_frames'])

            else:
                for node in self.nodes:
                    node.run()

            # all replay memory
            if self.shared_replay:
                for result in results:
                    node_idx = next(idx for idx, node in enumerate(self.nodes) if node.id == result['id'])
                    for idx, node in enumerate(self.nodes):
                        for frame in result['replay_buffer_buffer']:
                            self.nodes[idx].replay_buffer.push(*frame)
            else:
                # just my replay memory
                for result in results:
                    node_idx = next(idx for idx, node in enumerate(self.nodes) if node.id == result['id'])
                    self.nodes[node_idx].replay_buffer.buffer = result['replay_buffer_buffer']
                    self.nodes[node_idx].replay_buffer.position = result['replay_buffer_position']


            print("Avg reward: ", np.mean(round_reward), "var: ", np.std(round_reward) )
            self.experiment.log_metric(f'round_reward_avg', np.mean(round_reward), step=round_num)
            self.experiment.log_metric(f'round_reward_std', np.std(round_reward), step=round_num)
            value_params, policy_params = self.average_node_network_weights(value_params, policy_params)

            for node in self.nodes:
                node.policy_net.load_state_dict(policy_params)
                node.value_net.load_state_dict(value_params)

        # At this point all networks are the same, let's pick one and save it:
        self.save_model(self.nodes[0].value_net, self.experiment_path / 'value_net.mdl')
        self.save_model(self.nodes[0].policy_net, self.experiment_path / 'policy_net.mdl')

        return value_params, policy_params

    def save_model(self, network, path):
        torch.save(network.state_dict(), path)
        self.experiment.log_asset(path, overwrite=True)

def job(node_round):
    rewards, total_frames = node_round.run()
    return {
            'id': node_round.id,
            'rewards': rewards,
            'total_frames': total_frames,
            'value_net': node_round.value_net.state_dict(),
            'policy_net': node_round.policy_net.state_dict(),
            'replay_buffer_buffer': node_round.replay_buffer.buffer,
            'replay_buffer_position': node_round.replay_buffer.position,
            }

