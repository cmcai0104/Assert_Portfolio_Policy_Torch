import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class LSTM_Dist(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, output_size):
        super(LSTM_Dist, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_mu1 = nn.Linear(hidden_size + action_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, output_size)

        self.hidden_sigma_m1 = nn.Linear(hidden_size + action_size, 64)
        self.hidden_sigma_m2 = nn.Linear(64, int(output_size * output_size))

        self.hidden_sigma_v1 = nn.Linear(hidden_size + action_size, 64)
        self.hidden_sigma_v2 = nn.Linear(64, 32)
        self.hidden_sigma_v3 = nn.Linear(32, 16)
        self.hidden_sigma_v4 = nn.Linear(16, output_size)

        self.output_size = output_size

    def forward(self, env_state, action_state):
        lstm_out, (h_n, c_n) = self.lstm(env_state)
        cat_layer = torch.cat((lstm_out[:, -1, :], action_state), 1)

        mu = F.leaky_relu(self.hidden_mu1(cat_layer))
        mu = F.leaky_relu(self.hidden_mu2(mu))
        mu = F.leaky_relu(self.hidden_mu3(mu))
        mu = torch.softmax(self.hidden_mu4(mu), dim=1)

        sigma_matrix = F.leaky_relu(self.hidden_sigma_m1(cat_layer))
        sigma_matrix = torch.tanh(self.hidden_sigma_m2(sigma_matrix))

        sigma_vector = F.leaky_relu(self.hidden_sigma_v1(cat_layer))
        sigma_vector = F.leaky_relu(self.hidden_sigma_v2(sigma_vector))
        sigma_vector = F.leaky_relu(self.hidden_sigma_v3(sigma_vector))
        sigma_vector = torch.exp(self.hidden_sigma_v4(sigma_vector))

        return mu, sigma_matrix.reshape((-1, self.output_size, self.output_size)), sigma_vector


class LSTM_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_DQN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_mu1 = nn.Linear(hidden_size + output_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, output_size)

        self.hidden_sigma_m1 = nn.Linear(hidden_size + output_size, 64)
        self.hidden_sigma_m2 = nn.Linear(64, int(output_size * output_size))

        self.hidden_sigma_v1 = nn.Linear(hidden_size + output_size, 64)
        self.hidden_sigma_v2 = nn.Linear(64, 8)
        self.hidden_sigma_v3 = nn.Linear(8, 1)

        self.output_size = output_size

    def forward(self, env_state, action_state):
        lstm_out, (h_n, c_n) = self.lstm(env_state)
        cat_layer = torch.cat((lstm_out[:, -1, :], action_state), 1)

        mu = F.leaky_relu(self.hidden_mu1(cat_layer))
        mu = F.leaky_relu(self.hidden_mu2(mu))
        mu = F.leaky_relu(self.hidden_mu3(mu))
        mu = torch.softmax(self.hidden_mu4(mu), dim=1)

        sigma_matrix = F.leaky_relu(self.hidden_sigma_m1(cat_layer))
        sigma_matrix = self.hidden_sigma_m2(sigma_matrix)

        sigma_vector = F.leaky_relu(self.hidden_sigma_v1(cat_layer))
        sigma_vector = F.leaky_relu(self.hidden_sigma_v2(sigma_vector))
        sigma_vector = self.hidden_sigma_v3(sigma_vector)

        return mu, sigma_matrix.reshape((-1, self.output_size, self.output_size)), sigma_vector


class LSTM_A2C(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_A2C, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_mu1 = nn.Linear(hidden_size + output_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, output_size)

        self.hidden_layer1 = nn.Linear(hidden_size + output_size * 2, 64)
        self.hidden_layer2 = nn.Linear(64, 8)
        self.hidden_layer3 = nn.Linear(8, 1)

    def forward(self, env_state, act_state, action):
        lstm_out, (h_n, c_n) = self.lstm(env_state)
        cat_layer = torch.cat((lstm_out[:, -1, :], act_state), 1)

        mu = F.leaky_relu(self.hidden_mu1(cat_layer))
        mu = F.leaky_relu(self.hidden_mu2(mu))
        mu = F.leaky_relu(self.hidden_mu3(mu))
        mu = torch.softmax(self.hidden_mu4(mu), dim=1)

        cat_layer = torch.cat((cat_layer, action, mu), 1)
        q = F.leaky_relu(self.hidden_layer1(cat_layer))
        q = F.leaky_relu(self.hidden_layer2(q))
        q = self.hidden_layer3(q)
        return mu, q


class ACTOR_QVALUE(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(ACTOR_QVALUE, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_mu1 = nn.Linear(hidden_size + action_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, action_size)

        self.hidden_layer1 = nn.Linear(64 + action_size, 32)
        self.hidden_layer2 = nn.Linear(32, 8)
        self.hidden_layer3 = nn.Linear(8, 1)

    def forward(self, env_state, action_state, action=None, type='actor'):
        lstm_out, (h_n, c_n) = self.lstm(env_state)
        cat_layer = torch.cat((lstm_out[:, -1, :], action_state), 1)
        mu = F.relu(self.hidden_mu1(cat_layer))
        if type == 'actor':
            mu = F.leaky_relu(self.hidden_mu2(mu))
            mu = F.leaky_relu(self.hidden_mu3(mu))
            mu = torch.softmax(self.hidden_mu4(mu), dim=1)
            return mu
        else:
            cat_layer = torch.cat((mu, action), 1)
            q = F.leaky_relu(self.hidden_layer1(cat_layer))
            q = F.leaky_relu(self.hidden_layer2(q))
            q = self.hidden_layer3(q)
            return q
