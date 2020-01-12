import numpy as np
import pandas as pd
import gym
import torch


class MarketEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, price_cols: list, windows=250,
                 initial_account_balance=10000., sell_fee=0., buy_fee=0.015):
        super(MarketEnv, self).__init__()
        self.df = df
        self.price_cols = price_cols
        self.action_dim = len(self.price_cols)+1
        self.windows = windows
        self.buy_fee = buy_fee  # 购买费率
        self.sell_fee = sell_fee  # 赎回费率
        self.initial_account_balance = initial_account_balance  # 初始资金
        self.reward_range = (0, np.inf)
        self.Max_Steps = len(self.df) - 1
        self.seed()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float16)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(windows, self.df.shape[1]), dtype=np.float16)

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]

    # 下一步观测
    def _next_observation(self):
        obs = np.array(self.df.iloc[(self.current_step + 1 - self.windows): (self.current_step + 1), :].values /
                       self.Max_Share_Price, dtype=np.float32)
        return obs

    # 重置环境状态至初始状态
    def reset(self, start_step=None):
        self.net_worth = self.initial_account_balance
        self.max_net_worth = self.initial_account_balance
        self.shares_held = np.append(np.zeros(shape=self.action_dim - 1), self.initial_account_balance)
        if start_step == None:
            self.current_step = self.windows - 1
        elif start_step == 'random':
            self.current_step = np.random.randint(self.windows - 1, self.Max_Steps - 100)
        self.Max_Share_Price = self.df.iloc[:(self.current_step + 1), ].max(axis=0).values
        self.start_date = self.df.index[self.current_step]
        return torch.from_numpy(self._next_observation())

    # 进行交易
    def _take_action(self, target_rate: np.array):
        self.current_price = np.array(self.df.iloc[self.current_step, ][self.price_cols])
        # self.shares_before = self.shares_held
        self.net_worth = np.sum(np.append(self.current_price, 1)*self.shares_held)
        self.net_before = self.net_worth
        hold_rate = (np.append(self.current_price, 1) * self.shares_held / self.net_worth)
        # if np.sqrt(np.sum((hold_rate-target_rate)**2)) >= np.sqrt(2)*np.random.rand()*0.2:
        para = np.zeros(len(hold_rate)-1)
        sell_index = np.where(hold_rate[:-1] > target_rate[:-1])
        buy_index = np.where(hold_rate[:-1] < target_rate[:-1])
        para[sell_index] = 1-self.sell_fee
        para[buy_index] = 1/(1-self.buy_fee)
        self.net_worth = ((hold_rate[:-1]*para).sum()+hold_rate[-1]) / \
                         ((target_rate[:-1]*para).sum()+target_rate[-1]) * self.net_worth
        self.shares_held = self.net_worth * target_rate / np.append(self.current_price, 1)

    # 在环境中执行一步
    def step(self, action: np.array):
        obs = self._next_observation()
        self._take_action(action)
        self.current_step += 1
        self.next_price = np.array(self.df.iloc[self.current_step, ][self.price_cols])
        reward = np.sum(np.append(self.next_price, 1)*self.shares_held)
        done = self.current_step >= self.Max_Steps
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        return obs, reward/self.net_before-1, done, self.next_price/self.current_price-1

    # 打印出环境
    def render(self, mode='human'):
        ret = self.net_worth / self.initial_account_balance * 100 - 100
        yea_ret = (self.net_worth/self.initial_account_balance)**(365/(self.df.index[self.current_step] - self.start_date).days)*100-100
        print('总市值：{}/{}  |  累计收益率：{}%  |  累计年化收益率：{}% '.format(
            round(self.net_worth,2), round(self.max_net_worth,2), round(ret,2), round(yea_ret,2)))

