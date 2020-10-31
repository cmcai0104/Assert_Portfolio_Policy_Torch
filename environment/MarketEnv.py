import numpy as np
import pandas as pd
import gym


class MarketEnv(gym.Env):
    def __init__(self, df: pd.DataFrame, price_cols: list, windows=250,
                 initial_account_balance=10000., sell_fee=0., buy_fee=0.015):
        super(MarketEnv, self).__init__()
        self.df = df
        self.price_cols = price_cols
        self.action_dim = len(self.price_cols) + 1
        self.windows = windows
        self.buy_fee = buy_fee  # 购买费率
        self.sell_fee = sell_fee  # 赎回费率
        self.initial_account_balance = initial_account_balance  # 初始资金
        self.reward_range = (0, np.inf)
        self.Max_Steps = len(self.df) - 1
        self.seed()
        self.action_space = gym.spaces.Box(low=0, high=1, shape=(self.action_dim,), dtype=np.float16)
        self.observation_space = gym.spaces.Box(low=0, high=1, shape=(windows, self.df.shape[1]), dtype=np.float16)

    # def seed(self, seed=None):
    #     self.np_random, seed = gym.utils.seeding.np_random(seed)
    #     return [seed]

    # 下一步观测
    def __next_observation(self):
        obs = np.array(self.df.iloc[(self.current_step + 1 - self.windows): (self.current_step + 1), :].values /
                       self.Max_Share_Price, dtype=np.float32)
        return obs

    # 重置环境状态至初始状态
    def reset(self, start_step=None):
        assert (start_step == None or start_step == 'random')
        self.net_worth = self.initial_account_balance
        self.max_net_worth = self.initial_account_balance
        self.shares_held = np.append(np.zeros(shape=self.action_dim - 1), self.initial_account_balance)
        if start_step == None:
            self.current_step = self.windows - 1
        elif start_step == 'random':
            self.current_step = np.random.randint(self.windows - 1, self.Max_Steps - 100)
        self.Max_Share_Price = self.df.iloc[:(self.current_step + 1), ].max(axis=0).values
        self.start_date = self.df.index[self.current_step]
        self.next_price = np.array(self.df.iloc[self.current_step,][self.price_cols])
        self.next_net = np.sum(np.append(self.next_price, 1) * self.shares_held)
        self.next_rate = (np.append(self.next_price, 1) * self.shares_held) / self.next_net
        return self.__next_observation(), (self.shares_held / self.initial_account_balance).astype(np.float32)

    # 进行交易
    def __take_action(self, target_rate: np.array):
        self.current_price = self.next_price
        self.net_worth = self.next_net
        self.net_before = self.next_net
        hold_rate = self.next_rate
        # 减少交易的探索
        # if np.any(target_rate != hold_rate):
        if not all(target_rate == hold_rate):
            para = np.zeros(len(hold_rate) - 1)
            sell_index = np.where(hold_rate[:-1] > target_rate[:-1])
            buy_index = np.where(hold_rate[:-1] < target_rate[:-1])
            para[sell_index] = 1 - self.sell_fee
            para[buy_index] = 1 / (1 - self.buy_fee)
            self.net_worth = ((hold_rate[:-1] * para).sum() + hold_rate[-1]) / \
                             ((target_rate[:-1] * para).sum() + target_rate[-1]) * self.net_worth
            self.shares_held = self.net_worth * target_rate / np.append(self.current_price, 1)

    # 在环境中执行一步
    def step(self, action: np.array):
        self.__take_action(action)
        self.current_step += 1
        self.next_price = np.array(self.df.iloc[self.current_step,][self.price_cols])
        self.next_net = np.sum(np.append(self.next_price, 1) * self.shares_held)
        self.next_rate = (np.append(self.next_price, 1) * self.shares_held) / self.next_net
        reward = np.log(self.next_net / self.net_before)
        done = self.current_step >= self.Max_Steps
        if self.net_worth > self.max_net_worth:
            self.max_net_worth = self.net_worth
        self.Max_Share_Price = self.df.iloc[:(self.current_step + 1), ].max(axis=0).values
        state = self.__next_observation(), self.next_rate.astype(np.float32)
        return state, reward, done, None

    # 打印出环境
    def render(self, mode='human'):
        ret = self.net_worth / self.initial_account_balance * 100 - 100
        yea_ret = (self.net_worth / self.initial_account_balance) ** (
                365 / (self.df.index[self.current_step] - self.start_date).days) * 100 - 100
        print('总市值：{}/{}  |  累计收益率：{}%  |  累计年化收益率：{}% '.format(
            round(self.net_worth, 2), round(self.max_net_worth, 2), round(ret, 2), round(yea_ret, 2)))
