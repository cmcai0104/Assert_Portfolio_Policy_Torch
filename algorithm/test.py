import numpy as np
import pandas as pd
from environment.MarketEnv import MarketEnv


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







import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd
from torch.autograd import Variable
import random
import torch.nn.init as init

class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hiddenfc1 = nn.Linear(hidden_size, 64)
        self.hiddenfc2 = nn.Linear(64, 32)
        self.hiddenfc3 = nn.Linear(32, 16)
        self.hiddenfc4 = nn.Linear(16, output_size)
        self.hidden_size = hidden_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.randn(2, 1, self.hidden_size)),
                Variable(torch.randn(2, 1, self.hidden_size)))

    def forward(self, wordvecs):
        lstm_out, self.hidden = self.lstm(wordvecs, self.hidden)
        tag_space = F.relu(self.hiddenfc1(lstm_out[:, -1, :]))
        tag_space = F.relu(self.hiddenfc2(tag_space))
        tag_space = F.relu(self.hiddenfc3(tag_space))
        tag_space = F.sigmoid(self.hiddenfc4(tag_space))
        self.hidden = self.init_hidden()
        return tag_space


policy = LSTM(input_size=102, hidden_size=128, output_size=8)
obs = env.reset()
obs = torch.tensor(np.array([obs])).device()
policy()
while True:
    action = policy(np.array([obs]))
    break
    action = action/np.sum(action)
    obs, reward, done, _ = env.step(action)
    print('observer:{}, \nreward:{}'.format(obs, reward))
    if done:
        break
env.render()