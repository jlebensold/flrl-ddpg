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
from src.ddpg_round import DDPGRound
from src.dqn_round import DQNRound
from gym.envs.registration import register as gym_register


import os
from pathlib import Path
import multiprocessing
from multiprocessing.pool import ThreadPool
import threading

class AveragingRoundManager:
    def __init__(self, node_params=[], experiment_params={}):

        if experiment_params['algo'] == 'DDPG':
            self.nodes = [ DDPGRound(param) for param in node_params ]
        if experiment_params['algo'] == 'DQN':
            self.nodes = [ DQNRound(param) for param in node_params ]

        self.num_nodes = len(self.nodes)
        self.experiment = experiment_params['experiment']

        self.num_rounds = experiment_params['num_rounds']
        self.multiprocess= experiment_params['multiprocess']
        self.shared_replay = experiment_params['shared_replay']
        self.experiment_path = experiment_params['experiment_path']
        self.alpha = experiment_params['alpha']
        self.beta = experiment_params['beta']
        os.makedirs(self.experiment_path, exist_ok=True)
        self.multiprocess = False

    def primary_node(self):
        return self.nodes[0]

    def primary_id(self):
        return self.primary_node().id

    def primary_policy_net(self):
        return self.primary_node().policy_net

#    def primary_value_net(self):
#        return self.primary_node().value_net

    def params_from_node(self, node):
        policy_params = dict()
#        value_params = dict()
#        for name, tensor in node.value_net.named_parameters():
#            value_params[name] = tensor

        for name, tensor in node.policy_net.named_parameters():
            policy_params[name] = tensor

        return policy_params

    def init_value_and_policy_params(self):
        policy_params = dict()
        for name, param in self.primary_policy_net().named_parameters():
            policy_params[name] = torch.zeros_like(param)

        return policy_params

    def add_node_weights_to_network_params(self, node, policy_params):
        for name, tensor in node.policy_net.named_parameters():
            policy_params[name] += tensor

        return policy_params


    def average_node_network_weights(self, policy_params):
        for node in self.nodes:
            policy_params = self.add_node_weights_to_network_params(node, policy_params)

        print(policy_params.keys())
        # average:
        for name, param in policy_params.items():
            policy_params[name] /= self.num_nodes

        return policy_params

    def run_rounds(self):

        trailing_avg = {}
        for node in self.nodes:
            trailing_avg[node.id] = (np.zeros(5) - 1500).tolist() # prime the list for averaging


        for idx in range(self.num_rounds):
            round_num = idx + 1
            print(f'Round {round_num}')

            if self.multiprocess == True:
                pool = multiprocessing.Pool(len(self.nodes))
                results =  pool.starmap(job, [(node,) for node in self.nodes])
            else:
                results = [job(node) for node in self.nodes]
            policy_params = self.init_value_and_policy_params()
            round_reward = []
            for result in results:
                node_idx = next(idx for idx, node in enumerate(self.nodes) if node.id == result['id'])
                id = result['id']
#                self.nodes[node_idx].value_net.load_state_dict(result['value_net'])
                self.nodes[node_idx].policy_net.load_state_dict(result['policy_net'])
                self.nodes[node_idx].total_frames = result['total_frames']
                round_reward.append(result['rewards'])
                # Take the last reward from the round and assume this is
                # avereage performance:
                trailing_avg[id].append(np.mean(result['rewards']))
                for (step, eps_reward) in result['episode_rewards']:
                    self.experiment.log_metric(f'reward.{id}', eps_reward, step=step)


                self.experiment.log_metric(f'round_avg.{id}', np.mean(result['rewards']), step=result['total_frames'])
                self.experiment.log_metric(f'trailing_avg_5.{id}', np.mean(trailing_avg[id][-5]),step=result['total_frames'])

            # all replay memory
            if self.shared_replay:
                for result in results:
                    node_idx = next(idx for idx, node in enumerate(self.nodes) if node.id == result['id'])
                    for idx, node in enumerate(self.nodes):
                        for frame in result['replay_buffer_buffer']:
                            # TODO: double check that this is doing what I think it's
                            # doing
                            self.nodes[idx].replay_buffer.push(*frame)
            else:
                # just my replay memory
                for result in results:
                    node_idx = next(idx for idx, node in enumerate(self.nodes) if node.id == result['id'])
                    self.nodes[node_idx].replay_buffer.buffer = result['replay_buffer_buffer']
                    self.nodes[node_idx].replay_buffer.position = result['replay_buffer_position']

            avg_round_reward = np.mean([np.mean(r) for r in round_reward])
            self.experiment.log_metric(f'round_reward_avg', avg_round_reward, step=round_num)
            policy_params = self.average_node_network_weights(policy_params)

            # first we load the primary policy
            self.primary_policy_net().load_state_dict(policy_params, strict=False)
#            self.primary_value_net().load_state_dict(value_params)
            primary_policy_params = policy_params
#            primary_value_params = value_params

            for node in self.nodes:
                if node.id == self.primary_id():
                    continue

                # what is the distance in hparams?
                #differential = np.abs(self.primary_node().env_param - node.env_param)
                node_policy = self.params_from_node(node)

                for key, param in node_policy.items():
                    if self.beta == 0. or self.alpha == 0.:
                        reg_a = 0
                        reg_b = 0
                    else:
                        reg_a = ( self.alpha / self.beta ) * np.log(torch.clamp(primary_policy_params[key].detach(), 0.01, 5_000_000).numpy())
                        reg_b = ( 1 / self.beta ) * np.log(torch.clamp(node_policy[key].detach(), 0.01, 5_000_000).numpy())

                    node_policy[key] = node_policy[key].detach() + torch.tensor(reg_a) - torch.tensor(reg_b)

                node.policy_net.load_state_dict(policy_params, strict=False)
        # At this point all networks are the same, let's pick one and save it:
#        self.save_model(self.primary_node().value_net, self.experiment_path / 'value_net.mdl')
        self.save_model(self.primary_node().policy_net, self.experiment_path / 'policy_net.mdl')

        return value_params, policy_params

    def save_model(self, network, path):
        torch.save(network.state_dict(), path)
        self.experiment.log_asset(path, overwrite=True)

def job(node_round):
    episode_rewards = list()
    def on_episode_done(id, step, reward):
        episode_rewards.append((step,reward))

    node_round.on_episode_done = on_episode_done
    rewards, total_frames = node_round.run()
    return {
            'id': node_round.id,
            'rewards': rewards,
            'episode_rewards': episode_rewards,
            'total_frames': total_frames,
#            'value_net': node_round.value_net.state_dict(),
            'policy_net': node_round.policy_net.state_dict(),
            'replay_buffer_buffer': node_round.replay_buffer.buffer,
            'replay_buffer_position': node_round.replay_buffer.position,
            }

