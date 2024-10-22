import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
TransitionDistral = namedtuple('TransitionDistral', ('state', 'action', 'next_state', 'reward', 'time'))
class DQN(nn.Module):

    def __init__(self, h, w):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, 2) # 448 or 512

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class DQNGridworld(nn.Module):
    """
    Deep neural network with represents an agent.
    """
    def __init__(self, input_size, num_actions):
        super(DQNGridworld, self).__init__()
        self.linear1 = nn.Linear(input_size, 50)
        self.linear2 = nn.Linear(50, 50)
        self.head = nn.Linear(50, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        return self.head(x)


class PolicyNetworkGridworld(nn.Module):
    """
    Deep neural network which represents policy network.
    """
    def __init__(self, input_size, num_actions):
        super(PolicyNetworkGridworld, self).__init__()
        self.linear1 = nn.Linear(input_size, 50)
        self.linear2 = nn.Linear(50, 50)
        self.head = nn.Linear(50, num_actions)

    def forward(self, x):
        x = F.leaky_relu(self.linear1(x))
        x = F.leaky_relu(self.linear2(x))
        return F.softmax(self.head(x))


class PolicyNetwork(nn.Module):
    """
    Deep neural network which represents policy network.
    """
    def __init__(self, num_actions):
        super(PolicyNetwork, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 10, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(10)
        self.conv3 = nn.Conv2d(10, 10, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(10)
        self.head = nn.Linear(160, num_actions)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = F.leaky_relu(self.bn3(self.conv3(x)))
        x = F.leaky_relu(self.head(x.view(x.size(0), -1)))
        return self.softmax(x)
