import os
import sys
sys.path.append('D:/Project/Assert_Portfolio_Policy_Torch')
# sys.path.append('/home/python/work_direction/project/Assert_Portfolio_Policy_Torch')
import numpy as np
import pandas as pd
from collections import namedtuple
from environment.MarketEnv import MarketEnv
from baselines.ReplayMemory import ReplayMemory
from baselines.policy_network import LSTM
import torch
import torch.optim as optim
import torch.autograd


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
env = MarketEnv(df=df, price_cols=price_columns, windows=windows, initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)
policy_net = LSTM(input_size=df.shape[1], hidden_size=128, output_size=8).to(device)
optimizer = optim.RMSprop(policy_net.parameters())

Transition = namedtuple('Transition', ('state', 'action', 'log_prob', 'reward'))


def select_action(state):
    mu, sigma_matrix, sigma_vector = policy_net(state)
    sigma = sigma_matrix.squeeze() * torch.diagflat(sigma_vector + 1e-2) * torch.transpose(sigma_matrix.squeeze(), 0, 1)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    action = dist.sample()
    while torch.all(action < 0):
        action = dist.sample()
    action = torch.clamp(action, min=0, max=1)
    action = action / torch.sum(action)
    return action, dist


def interactivate(env):
    net_list = []
    state = env.reset()
    state = torch.from_numpy(state).unsqueeze(0).to(device)
    memory = ReplayMemory(3000)
    while True:
        action, dist = select_action(state)
        state, reward, done, next_rets = env.step(action.detach().numpy()[0])
        net_list.append(env.next_net)
        reward = torch.tensor([reward], device=device)
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        log_prob = dist.log_prob(action)
        memory.push(state, action, log_prob, reward)
        if done:
            break
    env.render()
    return memory, np.array(net_list)/env.initial_account_balance-1


def discount_reward(rewards, gamma=0.04/250):
    rewards_list = rewards.detach().numpy().tolist()
    discounted_ep_rs = np.zeros_like(rewards_list)
    running_add = 0
    for t in reversed(range(0, len(rewards_list))):
        running_add = running_add * gamma + rewards_list[t]
        discounted_ep_rs[t] = running_add
    return torch.from_numpy(discounted_ep_rs).to(device)


def optimize_model(memory, batch_size):
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    reward_batch = torch.cat(batch.reward)
    log_prob_batch = torch.cat(batch.log_prob)

    discounted_rewards = discount_reward(reward_batch)
    loss = -torch.mean(log_prob_batch * discounted_rewards)
    optimizer.zero_grad()
    loss.backward(retain_graph=True)
    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()


num_episodes = 500
for i_episode in range(num_episodes):
    memory, net_list = interactivate(env)
    optimize_model(memory)
    env.render()


