import random
import gym
from gym.envs.registration import register as gym_register
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


from .replay_memory import ReplayMemory
from .dqn import DQN
from .gravity_cartpole import GravityCartPole


import io
import PIL.Image
import base64

import gym
import math
import random
from collections import namedtuple
from itertools import count
from PIL import Image

envs = {
    'GravityCartPole' : {
        'env': GravityCartPole,
        'state_dim': 2,
        'max_steps': 300,
        'value_lr': 1e-3,
        'policy_lr': 1e-4,
        'tau': 1.,
        'batch_size':64,
    },
}
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class DQNRound():

    @classmethod
    def defaults(cls):

        return dict(
                seed=1,
#                device=torch.device("cuda:0"),
                device=torch.device("cpu"),
                max_frames=12000,
                algo='DQN',
                on_episode_done=None,
                id="ID"
                )



    def __init__(self, params):

        params = dict(params)
        self.env_param = params['env_param']
        self.device = params['device']
        self.num_episodes = params['max_frames']
        self.on_episode_done = params['on_episode_done']
        self.id = params['id']
        self.total_frames = 0
        self.seed = params['seed']
        self.env_name = params['env']

        # normally taken from the env, but since we're adjusting
        # environments after construction, it's easier to hard-code for now
        self.batch_size  = envs[self.env_name]['batch_size']
        state_dim = envs[self.env_name]['state_dim']
        self.env = gym.make('CartPole-v1').unwrapped



        # Get screen size so that we can initialize layers correctly based on shape
        # returned from AI gym. Typical dimensions at this point are close to 3x40x90
        # which is the result of a clamped and down-scaled render buffer in get_screen()


        self.cur_run = 0

        self.replay_buffer = ReplayMemory(Transition, 10000)
        self.steps_done = 0
        self.episode_durations = []
        self.episode_rewards = []
        self.running_avg = 100
        self.gamma = 0.999

        self.build_models()
        self.setup()

    def env_factory(self):
        return envs[self.env_name]['env'](self.env_param)

    def setup(self):
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
        self.env.reset()
        init_screen = self.get_screen()
        _, _, screen_height, screen_width = init_screen.shape
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
        EPS_START = 0.9
        EPS_END = 0.005
        EPS_DECAY = 200

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
            return torch.tensor([[random.randrange(2)]], device=self.device, dtype=torch.long)


    def optimize_model(self):
        """
            Run a training step
        """
        if len(self.replay_buffer) < self.batch_size:
            return
        transitions = self.replay_buffer.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.uint8)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
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
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()


    def run(self):
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.cur_run += 1
        self.env.reset()
        self.episode_rewards = []
        for i_episode in range(self.num_episodes):
            # Initialize the environment and state
            self.env.reset()
            last_screen = self.get_screen()
            current_screen = self.get_screen()
            state = current_screen - last_screen
            rewards = []
            for t in count():
                # Select and perform an action
                action = self.select_action(state)
                _, reward, done, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                rewards.append(reward.cpu().numpy())

                # Observe new state
                last_screen = current_screen
                current_screen = self.get_screen()

                if not done:
                    next_state = current_screen - last_screen
                else:
                    next_state = None

                # Store the transition in replay_buffer
                self.replay_buffer.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                self.optimize_model()
                if done:
                    self.episode_durations.append(t + 1)
                    break

            self.episode_rewards.append(np.sum(rewards))

            avg = None
            if len(self.episode_rewards) > self.running_avg:
                avg = np.mean(self.episode_rewards[-self.running_avg:])

            print(f'[{self.id}-{self.cur_run}] - Episode {i_episode}: {np.sum(rewards)}, avg: {avg}')
            self.episode_rewards.append(np.sum(rewards))

            TARGET_UPDATE = 10


            # Update the target network, copying all weights and biases in DQN
            if i_episode % TARGET_UPDATE == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

        print(f'[{self.id}-{self.cur_run}] - Complete - {np.mean(self.episode_rewards)}')
#        self.env.render()

        return rewards, self.total_frames


    def set_models(self, params):
        self.target_net.load_state_dict(params, strict=False)
        self.policy_net.load_state_dict(params, strict=False)

#agent_models = []
#env = gym.make('CartPole-v1').unwrapped
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def average_models(models):
    models = list(models)
    new_params = dict(models[0])
    candidate = models[0]
    for name, param in candidate:
        vals = [ model[name].data for model in models]
        new_params[name].data.copy_(np.mean(vals))
    return new_params

