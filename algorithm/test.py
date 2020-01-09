import os
import numpy as np
import pandas as pd
from collections import namedtuple
from environment.MarketEnv import MarketEnv
from algorithm.ReplayMemory import ReplayMemory
from algorithm.network import LSTM
import torch
import torch.optim as optim
# import torch.nn as nn
# import torch.nn.functional as F
import torch.autograd
# from torch.autograd import Variable
# import random
# import torch.nn.init as init

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

env = MarketEnv(df=df, price_cols=price_columns, windows=250,
                initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)

policy_net = LSTM(input_size=102, hidden_size=128, output_size=8).to(device)
optimizer = optim.RMSprop(policy_net.parameters())

state = env.reset()
state = torch.from_numpy(state).unsqueeze(0).to(device)

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward'))


def discount_reward(rewards, gamma=0.04/250):
    rewards_list = rewards.detach().numpy().tolist()
    discounted_ep_rs = np.zeros_like(rewards_list)
    running_add = 0
    for t in reversed(range(0, len(rewards_list))):
        running_add = running_add * gamma + rewards_list[t]
        discounted_ep_rs[t] = running_add
    return torch.from_numpy(discounted_ep_rs).to(device)


def optimize_model():
    transitions = memory.sample(len(memory))
    batch = Transition(*zip(*transitions))
    # state_batch = torch.cat(batch.state)
    # action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)
    log_prob_batch = torch.cat(batch.log_prob)

    discounted_rewards = discount_reward(reward_batch)
    loss = torch.mean(log_prob_batch * discounted_rewards)
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 50
for i_episode in range(num_episodes):
    memory = ReplayMemory(3000)
    while True:
        mu, sigma_matrix, sigma_vector = policy_net(state)
        sigma = sigma_matrix * torch.diagflat(sigma_vector + 1e-4) * torch.transpose(sigma_matrix, 0, 1)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        action = dist.sample()
        action = action/torch.sum(action)

        state, reward, done, _ = env.step(action.detach().numpy())
        reward = torch.tensor([reward], device=device)
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        log_prob = dist.log_prob(action)
        memory.push(state, action, log_prob, reward)
        if done:
            break
    optimize_model()
    env.render()


