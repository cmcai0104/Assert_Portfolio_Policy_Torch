import os
import random
import numpy as np
import pandas as pd
from collections import namedtuple
from environment.MarketEnv import MarketEnv
from algorithm.ReplayMemory import ReplayMemory
from algorithm.network import LSTM
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


Transition = namedtuple('Transition', ('mu', 'target_combine'))


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


loss_fun = torch.nn.MSELoss()


def generate_training_data(env):
    state = env.reset()
    state = state.unsqueeze(0).to(device)
    memory = ReplayMemory(3000)
    while True:
        mu, sigma_matrix, sigma_vector = policy_net(state)
        sigma = sigma_matrix * torch.diagflat(sigma_vector + 1e-2) * torch.transpose(sigma_matrix, 0, 1)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        action = dist.sample()
        while torch.all(action < 0):
            action = dist.sample()
        action = torch.clamp(action, min=0, max=1)
        action = action / torch.sum(action)

        state, reward, done, next_rets = env.step(action.detach().numpy()[0])
        target_combine = np.zeros(env.action_dim, dtype=np.float32)
        max_id = np.append(next_rets, 0).argmax()
        target_combine[max_id] = 1
        target_combine = torch.from_numpy(target_combine).unsqueeze(0).to(device)

        state = torch.from_numpy(state).unsqueeze(0).to(device)

        memory.push(mu, target_combine)
        if done:
            break
    env.render()
    return memory

def optimize_model(memory, batch_size=64):
    transitions = memory.sample(batch_size)
    batch = Transition(*zip(*transitions))
    mu_batch = torch.cat(batch.mu)
    target_combine_batch = torch.cat(batch.target_combine)

    optimizer.zero_grad()
    loss = loss_fun(mu_batch, target_combine_batch)
    # loss.backward()
    loss.backward(retain_graph=True)
    # for param in policy_net.parameters():
    #     param.grad.data.clamp_(-1, 1)
    optimizer.step()
    return loss.item()


if __name__ == '__main__':

    df, price_columns = df_preprocess('../data/create_feature.csv')

    env = MarketEnv(df=df, price_cols=price_columns, windows=250,
                    initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)

    policy_net = LSTM(input_size=102, hidden_size=128, output_size=8).to(device)
    optimizer = optim.RMSprop(policy_net.parameters())
    batch_size = 128
    memory = generate_training_data(env)
    num_episodes = 1000
    for i_episode in range(num_episodes):
        transitions = memory.sample(batch_size)
        batch = Transition(*zip(*transitions))
        mu_batch = torch.cat(batch.mu)
        target_combine_batch = torch.cat(batch.target_combine)
        optimizer.zero_grad()
        loss = loss_fun(mu_batch, target_combine_batch)
        loss.backward(retain_graph=True)
        optimizer.step()
        print('Training：{}/{} | Loss：{} '.format(i_episode, num_episodes, loss))
        if (i_episode % 100 == 0) and (i_episode > 0):
            generate_training_data(env)


