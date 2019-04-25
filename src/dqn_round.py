import random
import gym
from gym.envs.registration import register as gym_register
import numpy as np

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.distributions import Categorical

from .replay_memory import ReplayMemory
from .dqn import DQN, DQNGridworld, Transition, TransitionDistral
from .gravity_cartpole import GravityCartPole
from .envs.gridworld_env import GridworldEnv

import io
import PIL.Image
import base64

import gym
import math
import random
from itertools import count
from PIL import Image
from scipy.special import softmax

envs = {
    'GravityCartPole' : {
        'env': GravityCartPole,
        'state_dim': 2,
        'max_steps': 300,
        'batch_size':128,
        'num_actions': 2
    },
    'GridworldEnv' : {
        'env': GridworldEnv,
        'state_dim': 2,
        'max_steps': 1000,
        'batch_size':128,
        'num_actions': 5
    },
}


EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 5
TARGET_UPDATE = 5
TRAIL_AVG = 10

class DQNRound():

    @classmethod
    def defaults(cls):

        return dict(
                seed=1,
                device="cpu",
                algo='DQN',
                on_episode_done=None,
                id="ID"
                )



    def __init__(self, params):

        params = dict(params)
        self.env_param = params['env_param']
        self.device = torch.device(params['device'])
        self.num_episodes = params['num_episodes']
        self.on_episode_done = params['on_episode_done']
        self.id = params['id']
        self.total_frames = 0
        self.seed = params['seed']
        self.env_name = params['env_name']
        self.distral = params['distral']
        self.pi0_net = None
        self.alpha = params['alpha']
        self.beta = params['beta']

        # TODO: this is here so that the averager doesn't explode. come up with
        # something better like returning the networks and moving model saving
        self.value_net = None

        # normally taken from the env, but since we're adjusting
        # environments after construction, it's easier to hard-code for now
        self.batch_size  = envs[self.env_name]['batch_size']
        state_dim = envs[self.env_name]['state_dim']
        self.num_actions = envs[self.env_name]['num_actions']
        self.max_steps = envs[self.env_name]['max_steps']
        #self.env = gym.make('CartPole-v1').unwrapped

        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a clamped and down-scaled render buffer in get_screen()
        policy_buffer_size = 0
        if self.distral:
            policy_buffer_size = 1000
            self.replay_buffer = ReplayMemory(TransitionDistral, 10_000, policy_buffer_size)
        else:
            self.replay_buffer = ReplayMemory(Transition, 10_000, policy_buffer_size)
        self.steps_done = 0
        self.episode_durations = []
        self.episode_rewards = []
        self.gamma = 0.999

        self.build_models()
        self.setup()

    def env_factory(self):
        return envs[self.env_name]['env'](self.env_param)

    def setup(self):
        if self.env_name == 'GridworldEnv':
            env = GridworldEnv(self.env_param)
        else:
            env_name = f'{self.env_name}-{self.id}-v0'
            gym_register(id=env_name,entry_point=self.env_factory, max_episode_steps=200,)
            env = gym.make(env_name).unwrapped
        self.env = env
        self.set_seed(self.seed)

    def set_seed(self, seed):
        torch.manual_seed(seed)
        np.random.seed(seed)
        self.env.seed(seed)
        torch.cuda.manual_seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True


    def build_models(self):
        #self.env.reset()
        #init_screen = self.get_screen()
        #_, _, screen_height, screen_width = init_screen.shape

        if self.env_name == 'GridworldEnv':
            env = GridworldEnv(self.env_param)
            print(env.current_grid_map)
            self.input_size  = env.observation_space.shape[0]
            self.policy_net = DQNGridworld(self.input_size, self.num_actions).to(self.device)
            self.target_net = DQNGridworld(self.input_size, self.num_actions).to(self.device)
        else:
            screen_height = 90
            screen_width = 40
            self.policy_net = DQN(screen_height, screen_width).to(self.device)
            self.target_net = DQN(screen_height, screen_width).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()


    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def showarray(self, a, fmt='png'):
        a = np.uint8(a)
        f = io.BytesIO()
        ima = PIL.Image.fromarray(a).save(f, fmt)
        return base64.b64encode(f.getvalue()).decode('ascii')

    def get_screen(self):
        if self.env_name == 'GridworldEnv':
            return self.get_screen_gridworld()
        return self.get_screen_cartpole()

    def get_screen_gridworld(self):
        screen = self.env.current_grid_map
        screen = np.ascontiguousarray(screen, dtype=np.float32)
        screen = torch.from_numpy(screen)
        return screen.unsqueeze(0).unsqueeze(0).type(torch.Tensor).to(self.device)


    def get_screen_cartpole(self):
        # Returned screen requested by gym is 400x600x3, but is sometimes larger
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))

        # Cart is in the lower half, so strip off the top and bottom of the screen
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # Strip off the edges, so that we have a square image centered on a cart
        screen = screen[:, :, slice_range]
        # Convert to float, rescale, convert to torch tensor
        # (this doesn't require a copy)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)

        resize = T.Compose([T.ToPILImage(),
                            T.Resize(40, interpolation=Image.CUBIC),
                            T.ToTensor()])

        # Resize, and add a batch dimension (BCHW)
        return resize(screen).unsqueeze(0).to(self.device)


    def select_action(self, state):
        sample = random.random()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(self.num_actions)]], device=self.device, dtype=torch.long)



    def distral_select_action(self, state):
        """
        Selects whether the next action is choosen by our model or randomly
        """
        with torch.no_grad():
            Q = self.policy_net(Variable(state))
            pi0 = self.pi0_net(Variable(state))
            V = torch.log((torch.pow(pi0, self.alpha) * torch.exp(self.beta * Q)).sum(1)) / self.beta
            # added
            V[torch.isnan(V)] = 1e-15

        pi_i = torch.pow(pi0, self.alpha) * torch.exp(self.beta * (Q - V))
        # added
        pi_i[torch.isnan(pi_i)] = 1e-15
        #print(pi_i)
        #pi_i = pi_i.detach().cpu()
        #if (pi_i.data.numpy()[0][0] < 0 or pi_i.data.numpy()[0][1] < 0):
        #    print("Warning!!!: pi_i has negative values: pi_i", pi_i.data.numpy()[0])
        pi_i = torch.max(torch.zeros_like(pi_i) + 1e-15, pi_i)
        #print(pi_i)
        #probabilities = softmax(pi_i.cpu().numpy()[0])
        # print("pi_i = ", pi_i)

        #num_actions = self.env.action_space.n
        #action = np.random.choice(np.arange(0, num_actions), p=probabilities)
        m = Categorical(pi_i)
        action = m.sample().data.view(1, 1)
        return action.to(self.device)
        #return torch.tensor(action, device=self.device)



    def optimize_model(self):
        """
            Run a training step
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        transitions = [[torch.tensor(t) for t in sample] for sample in transitions]
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-500, 500)
        self.optimizer.step()


    def distral_optimize_model(self):
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        transitions = [[torch.tensor(t) for t in sample] for sample in transitions]
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        batch = TransitionDistral(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        with torch.no_grad():
            # We don't want to backprop through the expected action values and volatile
            # will save us on temporarily changing the model parameters'
            # requires_grad to False!
            non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                        if s is not None]),)

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)


        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.batch_size).to(self.device)
        next_state_values[non_final_mask] = torch.log(
            (torch.pow(self.pi0_net(non_final_next_states), self.alpha)
            * torch.exp(self.beta * self.target_net(non_final_next_states))).sum(1)) / self.beta
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        loss = F.mse_loss(state_action_values + 1e-16, expected_state_action_values)
        # print("loss:", loss)
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-500, 500)
            # print("model:", param.grad.data)
        self.optimizer.step()

    def run(self):
        #self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.001)
        self.env.reset()
        self.episode_rewards = []
        self.current_time = 0
        if self.distral:
            self.pi0_net = self.pi0_net.to(self.device)
        for i_episode in range(self.num_episodes):
            # Initialize the environment and state
            if self.env_name == 'GridworldEnv':
                state = torch.tensor( self.env.reset(), dtype=torch.float).view(-1,self.input_size).to(self.device)
            else:
                self.env.reset()
                last_screen = self.get_screen()
                current_screen = self.get_screen()
                state = current_screen - last_screen
            rewards = []

            for step in range(self.max_steps):
                # Select and perform an action
                if self.distral:
                    action = self.distral_select_action(state)
                else:
                    action = self.select_action(state)


                if self.env_name == 'GridworldEnv':
                    next_state_tmp, reward, done, _ = self.env.step(action[0,0])
                    self.current_time += 1

                    # Observe new state
                    next_state = torch.tensor( next_state_tmp, dtype=torch.float).view(-1,self.input_size).to(self.device)
                    if done:
                        next_state = None
                        break

                else:
                    _, reward, done, _ = self.env.step(action.item())
                    # Observe new state
                    last_screen = current_screen
                    current_screen = self.get_screen()
                    self.total_frames += 1
                    self.current_time += 1

                    if not done:
                        next_state = current_screen - last_screen
                    else:
                        next_state = None
                        break


                reward = torch.tensor([reward], device=self.device)
                rewards.append(reward.cpu().numpy())

                if self.distral:
                    # Store the transition in memory
                    time = torch.tensor([self.current_time])
                    # Store the transition in replay_buffer
                    self.replay_buffer.push(state.numpy(), action.numpy(),
                            next_state.numpy(), reward.numpy(), time.numpy())
                else:
                    self.replay_buffer.push(state.numpy(), action.numpy(),
                            next_state.numpy(), reward.numpy())


                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                if self.distral:
                    self.distral_optimize_model()
                else:
                    self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    self.current_time = 0
                    break

            self.episode_rewards.append(np.sum(rewards))
            print(f'[{self.id}] - Episode {i_episode}: {np.sum(rewards)}')


            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        return self.episode_rewards, self.total_frames


    def set_models(self, params):
        self.target_net.load_state_dict(params, strict=False)
        self.policy_net.load_state_dict(params, strict=False)
