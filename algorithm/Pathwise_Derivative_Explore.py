# coding=utf-8
import os
import sys
import random
import math
from collections import namedtuple
from datetime import datetime
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
from baselines.policy_network import ACTOR_QVALUE

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def df_preprocess(path):
    df = pd.read_csv(path, index_col=0, header=0)
    df['trade_date'] = df['trade_date'].astype('datetime64')
    df = df[df['trade_date'] <= datetime(2019, 8, 9)]
    df['trade_date'] = df['trade_date'].dt.date
    df = df.set_index('trade_date')
    colnames = df.columns.to_list()
    colnames = list(set(colnames) - set(['000001.SH_pe_y', '000300.SH_pe_y', '000905.SH_pe_y', '399006.SZ_pe_y']))
    colnames = [col for col in colnames if (col[:6] != '399016')]

    for col in colnames:
        # add moving average
        df[col + '_ma7'] = df[col].rolling(window=7, min_periods=1).mean()
        df[col + '_ma21'] = df[col].rolling(window=21, min_periods=1).mean()
        # add macd
        df[col + '_ema26'] = df[col].ewm(span=26).mean()
        df[col + '_ema12'] = df[col].ewm(span=12).mean()
        df[col + '_MACD'] = df[col + '_ema12'] - df[col + '_ema26']
        # add bollinger bands
        df[col + '_sd20'] = df[col].rolling(window=20).std()
        df[col + 'upper_band'] = df[col + '_ma21'] + (df[col + '_sd20'] * 2)
        df[col + 'lower_band'] = df[col + '_ma21'] - (df[col + '_sd20'] * 2)
        # add ema
        df[col + '_ema'] = df[col].ewm(com=0.5).mean()
        # fourier transforms
        fft_list = np.fft.fft(np.asarray(df[col].tolist()))
        for num_ in [3, 6, 9, 100]:
            fft_list_m = np.copy(fft_list)
            fft_list_m[num_:-num_] = 0
            df[col + '_fft%s' % num_] = np.fft.ifft(fft_list_m)
    # add quantile
    for num_ in [1, 2, 3, 5]:
        df[[col + '_qu' + str(250 * num_) for col in colnames]] = df[colnames].rolling(window=(250 * num_),
                                                                                       min_periods=50).apply(
            lambda x: len(x[x <= x[-1]]) / len(x), raw=True)

    df = df[colnames].dropna(axis=0, how='all').fillna(method='ffill', axis=0).dropna(axis=0, how='any')
    price_columns = [col for col in colnames if (col[-5:] == 'close')]
    return df, price_columns


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


BATCH_SIZE = 256
GAMMA = 0.999
EPS_START_LOW = 0.45
EPS_START_HIG = 0.55
EPS_END_LOW = 0.1
EPS_END_HIG = 0.9
EPS_DECAY = 100000
steps_done = 0


def select_action(state1, state2, hold_rate):
    global steps_done
    sample = random.random()
    eps_threshold_low = EPS_END_LOW + (EPS_START_LOW - EPS_END_LOW) * math.exp(-1. * steps_done / EPS_DECAY)
    eps_threshold_hig = EPS_END_HIG - (EPS_END_HIG - EPS_START_HIG) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if eps_threshold_low < sample < eps_threshold_hig:
        with torch.no_grad():
            return policy_net(env_state=state1, action_state=state2)
    elif sample > eps_threshold_hig:
        return hold_rate.to(device)
    else:
        i = np.random.randint(low=0, high=n_actions)
        ratio = np.zeros(shape=(1, n_actions), dtype=np.float32)
        ratio[0, i] = 1.
        ratio = torch.from_numpy(ratio)
        return ratio.to(device)


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
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward).unsqueeze(1)

    excepted_state_values = target_net(env_state=env_next_state_batch, action_state=act_next_state_batch,
                                       action=target_net(env_state=env_next_state_batch,
                                                         action_state=act_next_state_batch),
                                       type='qvalue') + reward_batch
    estimate_state_values = policy_net(env_state=env_state_batch, action_state=act_state_batch, action=action_batch,
                                       type='qvalue')

    q_regression_loss = F.smooth_l1_loss(estimate_state_values, excepted_state_values * GAMMA)
    q_values_loss = - torch.mean(
        policy_net(env_state=env_state_batch, action_state=act_state_batch,
                   action=policy_net(env_state=env_state_batch, action_state=act_state_batch), type='qvalue'))
    loss = q_regression_loss * 10000 + q_values_loss
    # 优化模型
    optimizer.zero_grad()
    loss.backward()
    for param in policy_net.parameters():
        param.grad.data.clamp_(-10, 10)
    optimizer.step()
    return loss


def test_interact(env):
    net_list = []
    state1, state2 = env.reset()
    while True:
        state1 = torch.from_numpy(state1).unsqueeze(0)
        state2 = torch.from_numpy(state2).unsqueeze(0)
        with torch.no_grad():
            action = policy_net(env_state=state1, action_state=state2)
        next_state, reward, done, _ = env.step(action.squeeze().detach().numpy())
        net_list.append(env.next_net)
        state1, state2 = next_state
        if done:
            print('test process: ', end=' ')
            env.render()
            break
    return np.array(net_list) / env.initial_account_balance - 1


if __name__ == '__main__':
    df, price_columns = df_preprocess('./data/create_feature.csv')
    windows = 250
    train_env = MarketEnv(df=df.iloc[:1500, :], price_cols=price_columns, windows=windows,
                          initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)
    test_env = MarketEnv(df=df, price_cols=price_columns, windows=windows,
                         initial_account_balance=10000., buy_fee=0.015, sell_fee=0.)
    n_actions = train_env.action_space.shape[0]
    policy_net = ACTOR_QVALUE(input_size=df.shape[1], hidden_size=128, action_size=n_actions).to(device)
    target_net = ACTOR_QVALUE(input_size=df.shape[1], hidden_size=128, action_size=n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters())
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=10)

    Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))
    memory = ReplayMemory(3000)
    num_episodes = 50
    TARGET_UPDATE = 3

    ret_df = pd.DataFrame(index=df.index[250:], dtype=np.float64)
    loss_list = []
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        state1, state2 = train_env.reset()
        loss = 0
        for t in count():
            state1 = torch.from_numpy(state1).unsqueeze(0)
            state2 = torch.from_numpy(state2).unsqueeze(0)
            hold_rate = torch.from_numpy(train_env.next_rate.astype(np.float32)).unsqueeze(0)
            # Select and perform an action
            action = select_action(state1, state2, hold_rate)
            next_state, reward, done, _ = train_env.step(action.squeeze().detach().numpy())
            reward = torch.tensor([reward], device=device)
            memory.push((state1, state2), action, next_state, reward)
            # Move to the next state
            state1, state2 = next_state
            if len(memory) >= (BATCH_SIZE * 5) and (t % 2 == 0):
                lo = optimize_model(memory)
                loss += lo.detach().numpy()
                # scheduler.step(loss)
            if len(memory) >= (BATCH_SIZE * 5) and (t % 100 == 0):
                target_net.load_state_dict(policy_net.state_dict())
            if done:
                print('%s,  ' % i_episode, end=' ')
                train_env.render()
                break
        loss_list.append(loss / t)
        # Update the target network, copying all weights and biases in DQN
        if (i_episode + 1) % TARGET_UPDATE == 0:
            torch.save(policy_net.state_dict(), "./model/pathwise_derivative_explore%s epoch.pt" % (i_episode + 1))
            ret_df['%s epoch' % (i_episode + 1)] = test_interact(test_env)
            ret_df.plot(title='Returns Curve')
            plt.savefig('./image/ret/pathwise_derivative_explore.jpg')
            plt.close()

            plt.plot(loss_list)
            plt.title('Training Loss - Pathwise_Derivative')
            plt.savefig('./image/loss/pathwise_derivative_explore.jpg')
            plt.close()
