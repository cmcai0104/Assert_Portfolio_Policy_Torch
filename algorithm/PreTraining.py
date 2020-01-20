import os
import sys
sys.path.append('D:/Project/Assert_Portfolio_Policy_Torch')
# sys.path.append('/home/python/work_direction/project/Assert_Portfolio_Policy_Torch')
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from environment.MarketEnv import MarketEnv
from baselines.policy_network import LSTM
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


def generate_training_data(env):
    state_list = []
    net_list = []
    state = env.reset()
    state_list.append(state)
    state = torch.from_numpy(state).unsqueeze(0).to(device)
    while True:
        mu, sigma_matrix, sigma_vector = policy_net(state)
        sigma = sigma_matrix.squeeze() * torch.diagflat(sigma_vector + 1e-2) * torch.transpose(sigma_matrix.squeeze(), 0, 1)
        dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mu, covariance_matrix=sigma)
        action = dist.sample()
        while torch.all(action < 0):
            action = dist.sample()
        action = torch.clamp(action, min=0, max=1)
        action = action / torch.sum(action)

        state, reward, done, next_rets = env.step(action.detach().numpy()[0])
        net_list.append(env.next_net)
        state_list.append(state)
        state = torch.from_numpy(state).unsqueeze(0).to(device)
        if done:
            break
    env.render()
    return torch.from_numpy(np.array(state_list[:-1])), np.array(net_list)/env.initial_account_balance-1


df, price_columns = df_preprocess('./data/create_feature.csv')
windows = 250
batch_size = 128
pretrain_targets = get_pretrain_target(df, price_columns).astype(np.float32)
env = MarketEnv(df=df, price_cols=price_columns, windows=windows, initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)
policy_net = LSTM(input_size=df.shape[1], hidden_size=128, output_size=8).to(device)
optimizer = optim.SGD(policy_net.parameters(), lr=0.01, momentum=0.9)
criterion = torch.nn.CrossEntropyLoss()


state_tensor, net_rets_init = generate_training_data(env)
pretrain_targets = torch.from_numpy(pretrain_targets.values[windows:]).argmax(dim=1)
traindataset = TensorDataset(state_tensor, pretrain_targets)
trainloader = DataLoader(traindataset, batch_size=batch_size, shuffle=False)


if __name__ == '__main__':
    epochs = 5
    train_loss = []
    eval_loss = []
    train_time = []
    time1 = time.time()
    for epoch in range(epochs):
        loss = []
        for xb, yb in trainloader:
            pred = policy_net(xb)[0]
            batch_loss = criterion(pred.squeeze(), yb.squeeze())
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()
            loss.append(batch_loss.data.item())


        policy_net.eval()
        with torch.no_grad():
            valid_loss = sum(criterion(policy_net(xb)[0], yb) for xb, yb in trainloader)
        train_loss.append(np.mean(loss))
        eval_loss.append(valid_loss.data.item())
        print("{}/{} | batch_loss:{} | valid_loss:{}".format(epoch, epochs, round(np.mean(loss),4),
                                                             round(valid_loss.data.item() / len(trainloader),4)))
    print("Average training time:",round((time.time()-time1)/epochs,2))
    torch.save(policy_net, './model/pretraining_model.pt')

    loss_df= pd.DataFrame({'train loss':train_loss, 'eval loss':eval_loss}, index = range(epochs))
    loss_df.plot(title='Loss Curve')
    plt.savefig('./image/loss/loss_pretraining.png')
    plt.close()

    state_tensor, net_rets = generate_training_data(env)
    ret_df = pd.DataFrame({'Initialization policy':net_rets_init, 'after training policy':net_rets}, index = df.index[250:])
    ret_df.plot(title='Returns Curve')
    plt.savefig('./image/ret/ret_pretraining.png')
    plt.close()

