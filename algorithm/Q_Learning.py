# coding=utf-8
import os
import sys
import random
import math
from collections import namedtuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import count
import torch
import torch.optim as optim
import torch.autograd
import torch.nn.functional as F
import torchvision.transforms as T

sys.path.append(os.getcwd())
from environment.MarketEnv import MarketEnv
from baselines.policy_network import LSTM


os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def df_preprocess(path):
    dataframe = pd.read_csv(path, index_col=0, header=0)
    dataframe['trade_date'] = dataframe['trade_date'].astype('datetime64')
    dataframe = dataframe[dataframe['trade_date'] <= pd.datetime.strptime('20190809', '%Y%m%d')]
    dataframe['trade_date'] = dataframe['trade_date'].dt.date
    dataframe = dataframe.set_index('trade_date').fillna(method='ffill', axis=0)
    # 剔除 399016
    colnames = dataframe.columns
    colnames = colnames[[col[:6] != '399016' for col in colnames]]
    dataframe = dataframe[colnames]
    dataframe = dataframe.dropna(axis=0, how='any')
    # 筛选出price列名及其对应的 dataframe
    price_columns = colnames[[col[-5:] == 'close' for col in colnames]]
    return dataframe, price_columns.to_list()

df, price_columns = df_preprocess('./data/create_feature.csv')
windows = 250
env = MarketEnv(df=df, price_cols=price_columns, windows=windows,
                initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)
n_actions = env.action_space.shape[0]
policy_net = LSTM(input_size=df.shape[1], hidden_size=128, output_size=n_actions).to(device)
target_net = LSTM(input_size=df.shape[1], hidden_size=128, output_size=n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()
optimizer = optim.RMSprop(policy_net.parameters())


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# 定义记忆长度为10000的replaymemory
memory = ReplayMemory(10000)

BATCH_SIZE = 128
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 200
TARGET_UPDATE = 10


steps_done = 0
def select_action(state1, state2, hold_rate, train=True):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state1, state2)
    else:
        return hold_rate.to(device)


episode_durations = []
# 优化模型参数
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # 对next_state进行拼接
    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    # tensor 拼接
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken.
    # These are the actions which would've been taken for each batch state according to policy_net
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))
    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    # Initialize the environment and state
    state1, state2 = env.reset()
    # env.reset()
    # last_screen = get_screen()
    # current_screen = get_screen()
    # state = current_screen - last_screen
    for t in count():
        state1 = torch.from_numpy(state1).unsqueeze(0)
        state2 = torch.from_numpy(state2).unsqueeze(0)
        hold_rate = torch.from_numpy(env.next_rate.astype(np.float32))
        # Select and perform an action
        action = select_action(state1, state2, hold_rate)
        next_state, reward, done, _ = env.step(action.squeeze().detach().numpy())
        # _, reward, done, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        # Observe new state
        #last_screen = current_screen
        #current_screen = get_screen()
        #if done:
        #    next_state = None
        # Store the transition in memory
        memory.push((state1, state2), action, next_state, reward)
        # Move to the next state
        state1, state2 = next_state
        if len(memory) > BATCH_SIZE:
            break
        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            episode_durations.append(t + 1)
            #plot_durations()
            break
    # Update the target network, copying all weights and biases in DQN
    if i_episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print('Complete')
env.render()
