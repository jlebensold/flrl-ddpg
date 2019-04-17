import gym
import numpy as np

from src.comet_logger import CometLogger
from collections import ChainMap
from src.ddpg_distral import DDPGDistral
from src.distral_round_manager import DistralRoundManager

from src.settings import EXPERIMENTS_PATH

from src.networks import ValueNetwork, PolicyNetwork

from gym.envs.registration import register as gym_register
from src.replay_buffer import ReplayBuffer
from src.gravity_pendulum import GravityPendulum
from src.normalized_actions import NormalizedActions
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# normally taken from the env, but since we're adjusting
# environments after construction, it's easier to hard-code for now
STATE_DIM = 3
ACTION_DIM = 1
HIDDEN_DIM = 256

def train(node_params, experiment_params):
    list_of_envs = []
    for idx, params in enumerate(node_params):
        env_name = f'GravityPendulum-{idx}-v0'
        def env_factory():
            g = params['g']
            return GravityPendulum(g)
        gym_register(id=env_name,entry_point=env_factory, max_episode_steps=200,)
        env = gym.make(env_name)
        list_of_envs.append(NormalizedActions(env))

    learning_rate = 0.001
    num_episodes = 10

    num_envs = len(node_params)
    policy = PolicyNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM)
    models = [PolicyNetwork(STATE_DIM, ACTION_DIM, HIDDEN_DIM) for _ in range(0, num_envs)]   ### Add torch.nn.ModuleList (?)
    replay_buffer_size = 1000000
    memories = [ReplayBuffer(replay_buffer_size) for _ in range(0, num_envs)]

    optimizers = [optim.Adam(model.parameters(), lr=learning_rate)
                    for model in models]
    policy_optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
    # optimizer = optim.RMSprop(model.parameters(), )

    episode_durations = [[] for _ in range(num_envs)]
    episode_rewards = [[] for _ in range(num_envs)]

    steps_done = np.zeros(num_envs)
    episodes_done = np.zeros(num_envs)
    current_time = np.zeros(num_envs)

    device = torch.device("cpu")
    batch_size = 128

    # Initialize environments
    for env in list_of_envs:
        env.reset()

    while np.min(episodes_done) < num_episodes:
        # TODO: add max_num_steps_per_episode

        # Optimization is given by alterating minimization scheme:
        #   1. do the step for each env
        #   2. do one optimization step for each env using "soft-q-learning".
        #   3. do one optimization step for the policy

        for i_env, env in enumerate(list_of_envs):
            # print("Cur episode:", i_episode, "steps done:", steps_done,
            #         "exploration factor:", eps_end + (eps_start - eps_end) * \
            #         math.exp(-1. * steps_done / eps_decay))




            state, action, reward, next_state, done = memories[i_env].sample(batch_size)

            state      = torch.FloatTensor(state).to(self.device)
            next_state = torch.FloatTensor(next_state).to(self.device)
            action     = torch.FloatTensor(action).to(self.device)
            reward     = torch.FloatTensor(reward).unsqueeze(1).to(self.device)
            done       = torch.FloatTensor(np.float32(done)).unsqueeze(1).to(self.device)



            # last_screen = env.current_grid_map
            current_screen = get_screen(env)
            state = current_screen # - last_screen
            # Select and perform an action
            action = select_action(state, policy, models[i_env], num_actions,
                                    eps_start, eps_end, eps_decay,
                                    episodes_done[i_env], alpha, beta)
            steps_done[i_env] += 1
            current_time[i_env] += 1
            _, reward, done, _ = env.step(action[0, 0])
            reward = Tensor([reward])

            # Observe new state
            last_screen = current_screen
            current_screen = get_screen(env)
            if not done:
                next_state = current_screen # - last_screen
            else:
                next_state = None

            # Store the transition in memory
            time = Tensor([current_time[i_env]])
            memories[i_env].push(state, action, next_state, reward, time)

            # Perform one step of the optimization (on the target network)
            optimize_model(policy, models[i_env], optimizers[i_env],
                            memories[i_env], batch_size, alpha, beta, gamma)
            if done:
                print("ENV:", i_env, "iter:", episodes_done[i_env],
                    "\treward:", env.episode_total_reward,
                    "\tit:", current_time[i_env], "\texp_factor:", eps_end +
                    (eps_start - eps_end) * math.exp(-1. * episodes_done[i_env] / eps_decay))
                env.reset()
                episodes_done[i_env] += 1
                episode_durations[i_env].append(current_time[i_env])
                current_time[i_env] = 0
                episode_rewards[i_env].append(env.episode_total_reward)
                if is_plot:
                    plot_rewards(episode_rewards, i_env)


        optimize_policy(policy, policy_optimizer, memories, batch_size,
                    num_envs, gamma)



def run(max_frames=500, num_rounds=5, seed=3, gravities=(10., 10.)):
    alpha = 0.5
    beta = 0.003
    params = DDPGDistral.defaults()
    params['algo'] = 'DDPG'
    params['env'] = 'GravityPendulum-v0'
    params['max_frames'] = max_frames
    params['alpha'] = alpha
    params['beta'] = beta

    node_params = [ChainMap({'id':f'n{idx}', 'g': g }, params) for idx,g in enumerate(gravities) ]
    experiment_params = {
        'experiment': None,
        'num_rounds': num_rounds,
        'experiment_path': None, #EXPERIMENTS_PATH / experiment.id,
        'multiprocess': True,
        'alpha': alpha,
        'beta': beta
    }



    train(node_params, experiment_params)





def select_action(state, policy, model, num_actions,  alpha, beta):
    """
    Selects whether the next action is choosen by our model or randomly
    """

    state  = torch.FloatTensor(state).unsqueeze(0).to(self.device)
    action = policy.forward(state)
    return action.detach().cpu().numpy()[0, 0]



def optimize_policy(policy, optimizer, memories, batch_size,
                    num_envs, gamma):
    loss = 0
    for i_env in range(num_envs):
        size_to_sample = np.minimum(batch_size, len(memories[i_env]))
        transitions = memories[i_env].policy_sample(size_to_sample)
        batch = Transition(*zip(*transitions))

        state_batch = Variable(torch.cat(batch.state))
        # print(batch.action)
        time_batch = Variable(torch.cat(batch.time))
        actions = np.array([action.numpy()[0][0] for action in batch.action])

        cur_loss = (torch.pow(Variable(Tensor([gamma])), time_batch) *
            torch.log(policy(state_batch)[:, actions])).sum()
        loss -= cur_loss
        # loss = cur_loss if i_env == 0 else loss + cur_loss

    optimizer.zero_grad()
    loss.backward()

    for param in policy.parameters():
        param.grad.data.clamp_(-500, 500)
        # print("policy:", param.grad.data)
    optimizer.step()

def optimize_model(policy, model, optimizer, memory, batch_size,
                    alpha, beta, gamma):
    if len(memory) < batch_size:
        return
    transitions = memory.sample(batch_size)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))
    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)

    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(batch_size).type(Tensor))
    next_state_values[non_final_mask] = torch.log(
        (torch.pow(policy(non_final_next_states), alpha)
        * torch.exp(beta * model(non_final_next_states))).sum(1)) / beta
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * gamma) + reward_batch

    # Compute Huber loss
    loss = F.mse_loss(state_action_values + 1e-16, expected_state_action_values)
    # print("loss:", loss)
    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-500, 500)
        # print("model:", param.grad.data)
    optimizer.step()
