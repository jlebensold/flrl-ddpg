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

    def push(self, *args):
        """Saves a transition."""
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = self.transition(*args)
        self.position = (self.position + 1) % self.capacity

        if self.policy_capacity == 0:
            return

        if len(self.policy_buffer) < self.policy_capacity:
            self.policy_buffer.append(None)
        self.policy_buffer[self.policy_position] = TransitionDistral(*args)
        self.policy_position = (self.policy_position + 1) % self.policy_capacity


    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def policy_sample(self, batch_size):
        return random.sample(self.policy_buffer, batch_size)

    def __len__(self):
        return len(self.buffer)
