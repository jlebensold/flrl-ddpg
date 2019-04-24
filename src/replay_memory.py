import random
from .dqn import Transition
from .dqn import TransitionDistral
class ReplayMemory(object):

    def __init__(self, transition, capacity, policy_capacity=0):
        self.capacity = capacity
        self.buffer = []
        self.position = 0
        self.transition = transition

        self.policy_capacity = policy_capacity
        self.policy_buffer = []
        self.policy_position = 0

    def push(self, state, action, next_state, reward, time=None):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        #self.buffer[self.position] = self.transition(*args)
        if time is None:
            self.buffer[self.position] = (state, action, next_state, reward)
        else:
            self.buffer[self.position] = (state, action, next_state, reward, time)
        self.position = (self.position + 1) % self.capacity

        if self.policy_capacity == 0:
            return

        if len(self.policy_buffer) < self.policy_capacity:
            self.policy_buffer.append(None)
        #self.policy_buffer[self.policy_position] = TransitionDistral(*args)
        if time is None:
            self.policy_buffer[self.policy_position] = (state, action, next_state, reward)
        else:
            self.policy_buffer[self.policy_position] = (state, action, next_state, reward, time)
        self.policy_position = (self.policy_position + 1) % self.policy_capacity


    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def policy_sample(self, batch_size):
        return random.sample(self.policy_buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
