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
sys.path.append(os.getcwd())
from environment.MarketEnv import MarketEnv
from baselines.policy_network import LSTM_A2C

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def df_preprocess(path):
    df = pd.read_csv(path, index_col=0, header=0)
    df['trade_date'] = df['trade_date'].astype('datetime64')
    df = df[df['trade_date'] <= pd.datetime.strptime('20190809', '%Y%m%d')]
    df['trade_date'] = df['trade_date'].dt.date
    df = df.set_index('trade_date').fillna(method='ffill', axis=0)
    # 剔除 399016
    colnames = df.columns
    colnames = colnames[[col[:6] != '399016' for col in colnames]]
    df = df[colnames]
    df = df.dropna(axis=0, how='any')
    # 筛选出price列名及其对应的 df
    price_columns = colnames[[col[-5:] == 'close' for col in colnames]]
    return df, price_columns.to_list()


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


memory = ReplayMemory(10000)

BATCH_SIZE = 256
GAMMA = 0.999
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 30000

steps_done = 0


def select_action(state1, state2, hold_rate):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state1, state2)[0]
    else:
        return hold_rate.to(device)


# 优化模型参数
def optimize_model(memory):
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))
    # 对next_state进行拼接
    env_next_state_batch = torch.tensor([env_next_state for env_next_state, _ in batch.next_state])
    act_next_state_batch = torch.tensor([act_next_state for _, act_next_state in batch.next_state])
    # 对state进行拼接
    env_state_batch = torch.cat([env_state for env_state, _ in batch.state])
    act_state_batch = torch.cat([act_state for _, act_state in batch.state])
    # 对action 和 reward 拼接
    # action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)

    state_action_batch, state_action_values = policy_net(env_state_batch, act_state_batch)
    next_state_action_batch, next_state_values = target_net(env_next_state_batch, act_next_state_batch)
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    q_value_loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)
    loss = q_value_loss - state_action_values.mean()/1000
    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-10, 10)
    optimizer.step()


def test_interact(env):
    net_list = []
    state1, state2 = env.reset()
    while True:
        state1 = torch.from_numpy(state1).unsqueeze(0)
        state2 = torch.from_numpy(state2).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(state1, state2)[0]
        next_state, reward, done, _ = env.step(action.squeeze().detach().numpy())
        net_list.append(env.next_net)
        state1, state2 = next_state
        if done:
            print('test process: \n')
            env.render()
            break
    return np.array(net_list)/env.initial_account_balance-1


def interact(env, num_episodes=50, target_update=10):
    ret_df = pd.DataFrame(index=df.index[250:], dtype=np.float64)
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state1, state2 = env.reset()
        for t in count():
            state1 = torch.from_numpy(state1).unsqueeze(0)
            state2 = torch.from_numpy(state2).unsqueeze(0)
            hold_rate = torch.from_numpy(env.next_rate.astype(np.float32)).unsqueeze(0)
            # Select and perform an action
            action = select_action(state1, state2, hold_rate)
            next_state, reward, done, _ = env.step(action.squeeze().detach().numpy())
            reward = torch.tensor([reward], device=device)
            memory.push((state1, state2), action, next_state, reward)
            # Move to the next state
            state1, state2 = next_state
            if len(memory) > BATCH_SIZE:
                optimize_model(memory)
            if done:
                env.render()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode+1 % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
            torch.save(policy_net.state_dict(), "./model/pathwise_derivative_model_%epoch.pt" % i_episode)
            ret_df['%sepoch'%i_episode] = test_interact(env)

    return ret_df



if __name__ == '__main__':
    df, price_columns = df_preprocess('./data/create_feature.csv')
    windows = 250
    env = MarketEnv(df=df, price_cols=price_columns, windows=windows,
                    initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)
    n_actions = env.action_space.shape[0]
    policy_net = LSTM_A2C(input_size=df.shape[1], hidden_size=128, output_size=n_actions).to(device)
    target_net = LSTM_A2C(input_size=df.shape[1], hidden_size=128, output_size=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.RMSprop(policy_net.parameters())

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

    num_episodes = 5
    TARGET_UPDATE = 1
    ret_df = interact(env, num_episodes=num_episodes, target_update=TARGET_UPDATE)
    ret_df.plot(title='Returns Curve')
    plt.savefig('./image/ret/pathwise_policy_gradient.jpg')
    plt.close()