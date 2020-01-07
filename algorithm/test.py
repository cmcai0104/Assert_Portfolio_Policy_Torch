import numpy as np
import pandas as pd
from collections import namedtuple
from environment.MarketEnv import MarketEnv
from algorithm.ReplayMemory import ReplayMemory
from algorithm.network import LSTM
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
import random
import torch.nn.init as init

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

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
def optimize_model():
    transitions = memory.sample(len(memory))
    batch = Transition(*zip(*transitions))
    pass


memory = ReplayMemory(10000)
while True:
    action = policy_net(state)
    action = action/torch.sum(action)
    state, reward, done, _ = env.step(action.detach().numpy())
    reward = torch.tensor([reward], device=device)
    state = torch.from_numpy(state).unsqueeze(0).to(device)
    memory.push(state, action, state, reward)
    if done:
        break
optimize_model()
env.render()


