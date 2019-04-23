import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.distributions import Normal

from collections import ChainMap

from src.replay_buffer import ReplayBuffer
from src.ou_noise import OUNoise
from src.ddpg_round import DDPGRound
from src.dqn_round import DQNRound
from src.dqn import Transition, TransitionDistral, PolicyNetwork, PolicyNetworkGridworld
from gym.envs.registration import register as gym_register


import os
from pathlib import Path
import multiprocessing
from multiprocessing.pool import ThreadPool
import threading

class AveragingRoundManager:
    def __init__(self, node_params=[], experiment_params={}):
        if experiment_params['algo'] == 'DDPG':
            RoundClass = DDPGRound
        if experiment_params['algo'] == 'DQN':
            RoundClass = DQNRound

        self.nodes = [ RoundClass(param) for param in node_params ]
        self.num_nodes = len(self.nodes)

        self.experiment = experiment_params['experiment']

        self.num_rounds = experiment_params['num_rounds']
        self.multiprocess= experiment_params['multiprocess']
        self.experiment_path = experiment_params['experiment_path']
        self.fame_regularize = experiment_params['fame_regularize']
        self.device = experiment_params['device']
        self.distral = experiment_params['distral']


        if self.distral:
            self.pi0_net = PolicyNetworkGridworld(self.nodes[0].input_size, self.nodes[0].num_actions)
            self.pi0_net_optimizer = optim.Adam(self.pi0_net.parameters(), lr=0.001)
            for node in self.nodes:
                node.pi0_net = self.pi0_net



        if self.distral and self.fame_regularize:
            raise("Only distral or FAME can be enabled for one experiment")
        self.alpha = experiment_params['alpha']
        self.beta = experiment_params['beta']
        os.makedirs(self.experiment_path, exist_ok=True)

    def params_from_node(self, node):
        policy_params = dict()
#        value_params = dict()
#        for name, tensor in node.value_net.named_parameters():
#            value_params[name] = tensor

        for name, tensor in node.policy_net.named_parameters():
            policy_params[name] = tensor

        return policy_params

    def init_policy_params(self):
        policy_params = dict()
        for name, param in self.nodes[0].policy_net.named_parameters():
            policy_params[name] = torch.zeros_like(param).cpu()

        return policy_params

    def add_node_weights_to_network_params(self, node, policy_params):
        for name, tensor in node.policy_net.named_parameters():
            policy_params[name] += tensor.cpu()

        return policy_params


    def average_node_network_weights(self):
        policy_params = self.init_policy_params()

        for node in self.nodes:
            policy_params = self.add_node_weights_to_network_params(node, policy_params)

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
            round_reward = []
            avg_reward = [np.mean(r['rewards']) for r in results]
            print(f"Episode Rewards avg: {avg_reward} | {np.mean(avg_reward)}")
            for result in results:
                node_idx = next(idx for idx, node in enumerate(self.nodes) if node.id == result['id'])
                id = result['id']

                # Copy back my network weights:
                if self.nodes[node_idx].value_net is not None:
                    self.nodes[node_idx].value_net.load_state_dict(result['value_net'])

                self.nodes[node_idx].policy_net.load_state_dict(result['policy_net'])
                self.nodes[node_idx].total_frames = result['total_frames']

                round_reward.append(result['rewards'])
                # Take the last reward from the round and assume this is
                # avereage performance:
                trailing_avg[id].append(np.mean(result['rewards']))
                for (step, eps_reward) in result['episode_rewards']:
                    self.experiment.log_metric(f'reward.{id}', eps_reward, step=step + idx)


                self.experiment.log_metric(f'round_avg.{id}', np.mean(result['rewards']), step=result['total_frames'])
                self.experiment.log_metric(f'trailing_avg_5.{id}', np.mean(trailing_avg[id][-5]),step=result['total_frames'])

            # copy back node replay memory
            for result in results:
                node_idx = next(idx for idx, node in enumerate(self.nodes) if node.id == result['id'])
                self.nodes[node_idx].replay_buffer.buffer = result['replay_buffer_buffer']
                self.nodes[node_idx].replay_buffer.position = result['replay_buffer_position']
                self.nodes[node_idx].replay_buffer.policy_buffer = result['replay_buffer_policy_buffer']
                self.nodes[node_idx].replay_buffer.policy_position = result['replay_buffer_policy_position']

            # log round averaged reward for each episode in round across all
            # nodes:
            avg_round_reward = np.mean([np.mean(r) for r in round_reward])
            self.experiment.log_metric(f'round_reward_avg', avg_round_reward, step=round_num)


            if (self.distral):
                self.perform_distral_distillation()

            if (self.fame_regularize):
                self.perform_federated_averaging()
        # Now save the distilled policy network:
        self.save_model(self.pi0_net, self.experiment_path / 'policy_net.mdl')

    def perform_federated_averaging(self):
        # 1. Average policy params across all nodes
        averaged_policy_params = self.average_node_network_weights()

        # 1.1 update the pi0 network on all nodes:
        for node in self.nodes:
            node.policy_net.load_state_dict(averaged_policy_params, strict=False)


    def perform_distral_distillation(self):
        loss = 0
        gamma = 0.999

        # 1. Perform distillation by accessing policy memories
        for node in self.nodes:
            size_to_sample = np.minimum(node.batch_size, len(node.replay_buffer.buffer))
            transitions = node.replay_buffer.policy_sample(size_to_sample)
            batch = TransitionDistral(*zip(*transitions))

            state_batch = torch.cat(batch.state).to(self.device).detach()
            time_batch = torch.cat(batch.time).clone().detach().type(torch.float).to(self.device)
            actions = np.array([action.cpu().numpy()[0][0] for action in batch.action])
            cur_loss = (torch.pow(torch.tensor([gamma], dtype=torch.float, device=self.device), time_batch) *
                torch.log(self.pi0_net(state_batch)[:, actions])).sum()
            loss -= cur_loss

        self.pi0_net_optimizer.zero_grad()
        loss.backward()

        for param in self.pi0_net.parameters():
            param.grad.data.clamp_(-500, 500)
            # print("policy:", param.grad.data)
        self.pi0_net_optimizer.step()

        # 2. now update each nodes' pi0 network:
        for node in self.nodes:
            node.pi0_net.load_state_dict(self.pi0_net.state_dict())

    def save_model(self, network, path):
        torch.save(network.state_dict(), path)
        self.experiment.log_asset(path, overwrite=True)

def job(node_round):
    episode_rewards = list()
    def on_episode_done(id, step, reward):
        episode_rewards.append((step,reward))

    node_round.on_episode_done = on_episode_done
    rewards, total_frames = node_round.run()
    value_net = None
    if node_round.value_net is not None:
        value_net = node_round.value_net.state_dict()
    return {
            'id': node_round.id,
            'rewards': rewards,
            'episode_rewards': episode_rewards,
            'total_frames': total_frames,
            'value_net': value_net,
            'policy_net': node_round.policy_net.state_dict(),
            'replay_buffer_buffer': node_round.replay_buffer.buffer,
            'replay_buffer_position': node_round.replay_buffer.position,
            'replay_buffer_policy_buffer': node_round.replay_buffer.policy_buffer,
            'replay_buffer_policy_position': node_round.replay_buffer.policy_position,
            }

