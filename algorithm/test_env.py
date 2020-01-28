import os
import numpy as np
import pandas as pd
from environment.MarketEnv import MarketEnv
from baselines.policy_network import LSTM_NN
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


def week_target(x):
    x = x.values
    y = np.zeros_like(x)
    y[x.argmax()] = 1.
    return pd.Series(y)


def get_pretrain_target(df, price_columns):
    price_df = df[price_columns]
    price_df_sub = price_df.copy().reset_index()
    price_df_sub['month'] = price_df_sub['trade_date'].astype('datetime64').dt.month
    # price_df_sub['month'] = [np.floor((month-1)/3)+1 for month in price_df_sub['month']]
    price_df_sub['year'] = price_df_sub['trade_date'].astype('datetime64').dt.year
    price_df_sub = price_df_sub.drop_duplicates(subset=['year', 'month'], keep='last').drop(
        columns=['year', 'month']).set_index('trade_date')
    rets_df_sub = price_df_sub.apply(lambda x: np.diff(np.log(x)), axis=0)
    rets_df_sub['cash'] = 0

    maxret_df_sub = rets_df_sub.apply(lambda x: week_target(x), axis=1)
    maxret_df_sub.index = price_df_sub.index.to_list()[:-1]
    maxret_df = pd.DataFrame(index=list(set(price_df.index.to_list())-set(price_df_sub.index.to_list()[:-1])))
    maxret_df = pd.concat([maxret_df, maxret_df_sub])
    maxret_df = maxret_df.sort_index(inplace=False)
    maxret_df = maxret_df.fillna(method='ffill')
    return maxret_df

df, price_columns = df_preprocess('./data/create_feature.csv')
windows = 250
batch_size = 128
env = MarketEnv(df=df, price_cols=price_columns, windows=windows,
                initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)

policy_net = LSTM_NN(input_size=102, action_size=8, hidden_size=128, output_size=8).to(device)


def select_action(state1, state2):
    mu, sigma_matrix, sigma_vector = policy_net(state1, state2)
    sigma = sigma_matrix.squeeze() * torch.diagflat(sigma_vector + 1e-2) * torch.transpose(sigma_matrix.squeeze(), 0, 1)
    dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=sigma)
    action = dist.sample()
    while torch.all(action < 0):
        action = dist.sample()
    action = torch.clamp(action, min=0, max=1)
    action = action / torch.sum(action)
    return action, dist


def interactivate(env):
    state1, state2 = env.reset()
    while True:
        state1 = torch.from_numpy(state1).unsqueeze(0)
        state2 = torch.from_numpy(state2).unsqueeze(0)
        action, dist = select_action(state1, state2)
        log_prob = dist.log_prob(action)
        state, reward, done, info = env.step(action.squeeze().detach().numpy())
        state1, state2 = state
        if done:
            break
    env.render()


interactivate(env)