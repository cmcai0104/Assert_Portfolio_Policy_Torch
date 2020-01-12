import os
import numpy as np
import pandas as pd
from environment.MarketEnv import MarketEnv
from algorithm.network import LSTM
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

df, price_columns = df_preprocess('../data/create_feature.csv')

env = MarketEnv(df=df, price_cols=price_columns, windows=250,
                initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)

policy_net = LSTM(input_size=102, hidden_size=128, output_size=8).to(device)

state = env.reset()
state = state.unsqueeze(0).to(device)

mu, sigma_matrix, sigma_vector = policy_net(state)

sigma = sigma_matrix * torch.diagflat(sigma_vector + 1e-2) * torch.transpose(sigma_matrix, 0, 1)
dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=sigma)
action = dist.sample()
action = action / torch.sum(action)

state, reward, done, _ = env.step(action.detach().numpy()[0])
env.render()



