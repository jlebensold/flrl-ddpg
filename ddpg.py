import math
import random

import gym
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from collections import ChainMap

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)



class NormalizedActions(gym.ActionWrapper):

    def _action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = low_bound + (action + 1.0) * 0.5 * (upper_bound - low_bound)
        action = np.clip(action, low_bound, upper_bound)

        return action

    def _reverse_action(self, action):
        low_bound   = self.action_space.low
        upper_bound = self.action_space.high

        action = 2 * (action - low_bound) / (upper_bound - low_bound) - 1
        action = np.clip(action, low_bound, upper_bound)

        return actions

class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3, min_sigma=0.3, decay_period=100000):
        self.mu           = mu
        self.theta        = theta
        self.sigma        = max_sigma
        self.max_sigma    = max_sigma
        self.min_sigma    = min_sigma
        self.decay_period = decay_period
        self.action_dim   = action_space.shape[0]
        self.low          = action_space.low
        self.high         = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x  = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)

#https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py

class ValueNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs + num_actions, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, 1)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x


class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, init_w=3e-3):
        super(PolicyNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, num_actions)

        self.linear3.weight.data.uniform_(-init_w, init_w)
        self.linear3.bias.data.uniform_(-init_w, init_w)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = F.tanh(self.linear3(x))
        return x

    def get_action(self, state):
        state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.forward(state)
        return action.detach().cpu().numpy()[0, 0]

from gym.envs.classic_control.pendulum import PendulumEnv, angle_normalize
from gym import Wrapper
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class GravityPendulum(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_speed=8
        self.max_torque=2.
        self.dt=.05
        self.viewer = None

        high = np.array([1., 1., self.max_speed])
        self.action_space = spaces.Box(low=-self.max_torque, high=self.max_torque, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.g = 10.
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        th, thdot = self.state # th := theta
        g = self.g
        m = 1.
        l = 1.
        dt = self.dt

        u = np.clip(u, -self.max_torque, self.max_torque)[0]
        self.last_u = u # for rendering
        costs = angle_normalize(th)**2 + .1*thdot**2 + .001*(u**2)

        newthdot = thdot + (-3*g/(2*l) * np.sin(th + np.pi) + 3./(m*l**2)*u) * dt
        newth = th + newthdot*dt
        newthdot = np.clip(newthdot, -self.max_speed, self.max_speed) #pylint: disable=E1111

        self.state = np.array([newth, newthdot])
        return self._get_obs(), -costs, False, {}

    def reset(self):
        high = np.array([np.pi, 1])
        self.state = self.np_random.uniform(low=-high, high=high)
        self.last_u = None
        return self._get_obs()

    def _get_obs(self):
        theta, thetadot = self.state
        return np.array([np.cos(theta), np.sin(theta), thetadot])

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-2.2,2.2,-2.2,2.2)
            rod = rendering.make_capsule(1, .2)
            rod.set_color(.8, .3, .3)
            self.pole_transform = rendering.Transform()
            rod.add_attr(self.pole_transform)
            self.viewer.add_geom(rod)
            axle = rendering.make_circle(.05)
            axle.set_color(0,0,0)
            self.viewer.add_geom(axle)
            fname = path.join(path.dirname(__file__), "assets/clockwise.png")
            self.img = rendering.Image(fname, 1., 1.)
            self.imgtrans = rendering.Transform()
            self.img.add_attr(self.imgtrans)

        self.viewer.add_onetime(self.img)
        self.pole_transform.set_rotation(self.state[0] + np.pi/2)
        if self.last_u:
            self.imgtrans.scale = (-self.last_u/2, np.abs(self.last_u)/2)

        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

def angle_normalize(x):
    return (((x+np.pi) % (2*np.pi)) - np.pi)


from gym.envs.registration import register as gym_register

gym_register(
    id='GravityPendulum-v0',
    entry_point=GravityPendulum,
    max_episode_steps=200,
)

class Round():

    @classmethod
    def defaults(cls):
        return dict(
                seed=1,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                env=NormalizedActions(gym.make("Pendulum-v0")),
                max_frames=12000,
                id="ID"
                )

    def __init__(self, params):
        params = dict(params)
        env = gym.make("GravityPendulum-v0")
        env.g = params['g']

        self.env = NormalizedActions(env)


        self.device = params['device']
        self.max_frames  = params['max_frames']
        self.max_steps   = 500
        self.batch_size  = 128
        self.id = params['id']
        self.total_frames = 0


        self.set_seed(params['seed'])
        self.ou_noise = OUNoise(self.env.action_space)

        state_dim  = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.shape[0]
        hidden_dim = 256

        self.value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        self.target_value_net  = ValueNetwork(state_dim, action_dim, hidden_dim).to(self.device)
        self.target_policy_net = PolicyNetwork(state_dim, action_dim, hidden_dim).to(self.device)

        for target_param, param in zip(self.target_value_net.parameters(), self.value_net.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.target_policy_net.parameters(), self.policy_net.parameters()):
            target_param.data.copy_(param.data)



        value_lr  = 1e-3
        policy_lr = 1e-4
        self.value_optimizer  = optim.Adam(self.value_net.parameters(),  lr=value_lr)
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=policy_lr)

        self.value_criterion = nn.MSELoss()

        replay_buffer_size = 1000000
        self.replay_buffer = ReplayBuffer(replay_buffer_size)


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
            max_value=np.inf,
            soft_tau=1e-2):

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
        rewards = []
        frame_idx= 0
        while frame_idx < self.max_frames:
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

#                if frame_idx % 100 == 0:
#                    print(f'[{self.id}] {frame_idx}: {np.mean(rewards)}')

                if done:
                    print(f'[{self.id}.{self.total_frames}] Episode reward {episode_reward}')
                    break


            rewards.append(episode_reward)

def main():
    params = Round.defaults()
    params['max_frames'] = 600

    nodes = [
            Round(ChainMap({'id':'bob', 'g': 10}, params)),
            Round(ChainMap({'id':'sally', 'g': 10}, params)),
            Round(ChainMap({'id':'jane', 'g': 10}, params)),
            Round(ChainMap({'id':'jim', 'g': 10}, params)),
    ]

    round_policy_params = dict()
    round_value_params = dict()

    for name, param in nodes[0].value_net.named_parameters():
        round_value_params[name] = torch.zeros_like(param)

    for name, param in nodes[0].policy_net.named_parameters():
        round_policy_params[name] = torch.zeros_like(param)

    for idx in range(5):

        print(f'Round {idx}')
        for node in nodes:
            node.run()

        # zero out the round:
        for name, param in round_value_params.items():
            round_value_params[name] = torch.zeros_like(param)
        for name, param in round_policy_params.items():
            round_policy_params[name] = torch.zeros_like(param)

        for node in nodes:
            for name, tensor in node.policy_net.named_parameters():
                round_policy_params[name] += tensor

            for name, tensor in node.value_net.named_parameters():
                round_value_params[name] += tensor

        # average:
        for name, param in round_value_params.items():
            round_value_params[name] /= len(nodes)

        for name, param in round_policy_params.items():
            round_policy_params[name] /= len(nodes)

        for node in nodes:
            node.policy_net.load_state_dict(round_policy_params)
            node.value_net.load_state_dict(round_value_params)

    print("Comparison")
    params = Round.defaults()
    single_learner = Round(ChainMap({'max_frames': nodes[0].total_frames, 'g': 10, 'id': 'fl.10'}, params))
    single_learner.policy_net.load_state_dict(round_policy_params)
    single_learner.value_net.load_state_dict(round_value_params)

    single_learner.run()
    single_learner = Round(ChainMap({'max_frames': nodes[0].total_frames, 'g': 8, 'id': 'fl.8'}, params))
    single_learner.policy_net.load_state_dict(round_policy_params)
    single_learner.value_net.load_state_dict(round_value_params)
    single_learner.run()

if __name__ == '__main__':
    main()
