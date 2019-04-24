import random
import gym
from gym.envs.registration import register as gym_register
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from .replay_buffer import ReplayBuffer
from .ou_noise import OUNoise
from .networks import PolicyNetwork, ValueNetwork
from .normalized_actions import NormalizedActions
from .gravity_pendulum import GravityPendulum
from .force_mountain_car_continuous import PositionContinuous_MountainCar

envs = {
    'MountainCarContinuous' : {
        'env': PositionContinuous_MountainCar,
        'state_dim': 2,
        'max_steps': 300,
        'value_lr': 1e-3,
        'policy_lr': 1e-4,
        'tau': 1.,
        'batch_size':64,
    },
    'GravityPendulum' : {
        'env': GravityPendulum,
        'state_dim': 3,
        'max_steps': 200,
        'value_lr': 1e-3,
        'policy_lr': 1e-4,
        'tau': 1e-2,
        'batch_size':128,
    }
}

ACTION_DIM = 1
HIDDEN_DIM = 256
class DDPGRound():

    @classmethod
    def defaults(cls):

        return dict(
                seed=1,
                device=torch.device("cpu"),
                num_episodes=1,
                max_frames=200,
                algo='DDPG',
                id="ID"
                )
    @classmethod
    def build_policy_network(cls, env_name):
        state_dim = envs[env_name]['state_dim']
        return PolicyNetwork(state_dim, ACTION_DIM, HIDDEN_DIM)

    def __init__(self, params):
        params = dict(params)
        self.env_param = params['env_param']
        self.device = params['device']
        self.max_frames  = params['max_frames']
        self.num_episodes = params['num_episodes']
        self.id = params['id']
        self.total_frames = 0
        self.seed = params['seed']
        self.env_name = params['env']

        # normally taken from the env, but since we're adjusting
        # environments after construction, it's easier to hard-code for now
        self.max_steps = envs[self.env_name]['max_steps']
        self.batch_size  = envs[self.env_name]['batch_size']

        state_dim = envs[self.env_name]['state_dim']
        self.value_net  = ValueNetwork(state_dim, ACTION_DIM, HIDDEN_DIM).to(self.device)
        self.policy_net = PolicyNetwork(state_dim, ACTION_DIM, HIDDEN_DIM).to(self.device)

        self.target_value_net  = ValueNetwork(state_dim, ACTION_DIM, HIDDEN_DIM).to(self.device)
        self.target_policy_net = PolicyNetwork(state_dim, ACTION_DIM, HIDDEN_DIM).to(self.device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)

        value_lr  = envs[self.env_name]['value_lr']
        policy_lr = envs[self.env_name]['policy_lr']

        self.value_optimizer  = optim.Adam(self.value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.MSELoss()

        replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)



    def env_factory(self):
        return envs[self.env_name]['env'](self.env_param)

    def setup(self):
        env_name = f'{self.env_name}-{self.id}-v0'
        gym_register(id=env_name,entry_point=self.env_factory, max_episode_steps=200,)
        env = gym.make(env_name)
        self.env = NormalizedActions(env)
        self.set_seed(self.seed)
        self.ou_noise = OUNoise(self.env.action_space)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def ddpg_update(self, batch_size,
            gamma = 0.99,
            min_value=-np.inf,
            max_value=np.inf):
        soft_tau = envs[self.env_name]['tau']
        state, action, reward, next_state, done = self.replay_buffer.sample(batch_size)

        state      = torch.FloatTensor(state).to(self.device)
        next_state = torch.FloatTensor(next_state).to(self.device)
        action     = torch.FloatTensor(action).to(self.device)
        reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
        done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)

        self.policy_loss = self.value_net(state, self.policy_net(state))
        self.policy_loss = -self.policy_loss.mean()

        next_action    = self.target_policy_net(next_state)
        target_value   = self.target_value_net(next_state, next_action.detach())
        expected_value = reward + (1.0 - done) * gamma * target_value
        expected_value = torch.clamp(expected_value, min_value, max_value)

        value = self.value_net(state, action)
        self.value_loss = self.value_criterion(value, expected_value.detach())


        self.policy_optimizer.zero_grad()
        self.policy_loss.backward()
        self.policy_optimizer.step()

        self.value_optimizer.zero_grad()
        self.value_loss.backward()
        self.value_optimizer.step()

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
                target_param.data.copy_(
                    target_param.data * (1.0 - soft_tau) + param.data * soft_tau
                )

    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy_net.forward(state)
        return action.detach().cpu().numpy()[0, 0]


    def run(self):
        self.setup()
        rewards = []
        frame_idx= 0
        cur_episode = 0
        for i_episode in range(self.num_episodes):
            cur_episode += 1
            state = self.env.reset()
            self.ou_noise.reset()
            episode_reward = 0

            for step in range(self.max_steps):
                action = self.get_action(state)
                action = self.ou_noise.get_action(action, step)
                next_state, reward, done, _ = self.env.step(action)
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.ddpg_update(self.batch_size)

                state = next_state
                episode_reward += reward
                frame_idx += 1
                self.total_frames += 1

                if done:
                    print(f'[{self.id}] - Episode {i_episode}: {episode_reward}')
                    break
            rewards.append(episode_reward)
        return rewards, self.total_frames
