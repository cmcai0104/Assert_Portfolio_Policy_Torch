import os
import sys
sys.path.append('/Users/CMCai/MyProject/Assert_Portfolio_Policy_Torch')
import time
import numpy as np
import pandas as pd
from environment.MarketEnv import MarketEnv
from algorithm.network import LSTM
import torch
from torch.utils.data import DataLoader, TensorDataset
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

def week_target(x):
    x = x.values
    y = np.zeros_like(x)
    y[x.argmax()] = 1.
    return pd.Series(y)

def get_pretrain_target(df, price_columns):
    price_df = df[price_columns]
    price_df_sub = price_df.copy().reset_index()
    price_df_sub['week'] = price_df_sub['trade_date'].astype('datetime64').dt.week
    price_df_sub['year'] = price_df_sub['trade_date'].astype('datetime64').dt.year
    price_df_sub = price_df_sub.drop_duplicates(subset=['year', 'week'], keep='last').drop(
        columns=['year', 'week']).set_index('trade_date')
    rets_df_sub = price_df_sub.apply(lambda x: np.diff(np.log(x)), axis=0)
    rets_df_sub['cash'] = 0

    maxret_df_sub = rets_df_sub.apply(lambda x:week_target(x), axis=1)
    maxret_df_sub.index = price_df_sub.index.to_list()[:-1]
    maxret_df = pd.DataFrame(index=list(set(price_df.index.to_list())-set(price_df_sub.index.to_list()[:-1])))
    maxret_df = pd.concat([maxret_df, maxret_df_sub])
    maxret_df = maxret_df.sort_index(inplace=False)
    maxret_df = maxret_df.fillna(method='ffill')
    return maxret_df

pretrain_targets = get_pretrain_target(df, price_columns).astype(np.float32)


env = MarketEnv(df=df, price_cols=price_columns, windows=250,
                    initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)
policy_net = LSTM(input_size=102, hidden_size=128, output_size=8).to(device)
optimizer = optim.RMSprop(policy_net.parameters())
criterion = torch.nn.MSELoss()


def generate_training_data(env):
    state_list = []
    state = env.reset()
    state_list.append(state)
    state = torch.from_numpy(state).unsqueeze(0).to(device)
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
        state_list.append(state)
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        if done:
            break
    env.render()
    return torch.from_numpy(np.array(state_list[:-1]))

state_tensor = generate_training_data(env)
pretrain_targets = torch.from_numpy(pretrain_targets.values[250:])
traindataset = TensorDataset(state_tensor, pretrain_targets)
trainloader = DataLoader(traindataset, batch_size=1, shuffle=True)
# dataiter = iter(trainloader)
# state, action = dataiter.next()
# mu = policy_net(state)[0]
# loss = criterion(mu, action)

if __name__ == '__main__':
    epochs = 100
    for epoch in range(epochs):
        time1 = time.time()
        for xb, yb in trainloader:
            pred = policy_net(xb)[0]
            loss = criterion(pred, yb)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        policy_net.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(policy_net(xb)[0], yb) for xb, yb in trainloader)
        print(epoch, valid_loss / len(trainloader))
        print('Time:', time.time() - time1)









