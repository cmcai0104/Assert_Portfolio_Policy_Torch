import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd import Variable


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


class ACTOR_QVALUE(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super(ACTOR_QVALUE, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_tan1 = nn.Linear(hidden_size, 64)
        self.hidden_tan2 = nn.Linear(64, 16)
        self.hidden_tan3 = nn.Linear(16, action_size)

        self.hidden_sig1 = nn.Linear(hidden_size, 64)
        self.hidden_sig2 = nn.Linear(64, 16)
        self.hidden_sig3 = nn.Linear(16, action_size)

        self.hidden_layer1 = nn.Linear(hidden_size + action_size * 2, 64)
        self.hidden_layer2 = nn.Linear(64, 8)
        self.hidden_layer3 = nn.Linear(8, 1)

    def forward(self, env_state, action_state, action=None, type='actor'):
        lstm_out, (h_n, c_n) = self.lstm(env_state)
        if type == 'actor':
            tan_mu = F.leaky_relu(self.hidden_tan1(lstm_out[:, -1, :]))
            tan_mu = F.leaky_relu(self.hidden_tan2(tan_mu))
            tan_mu = torch.tanh(self.hidden_tan3(tan_mu))

            sig_mu = F.leaky_relu(self.hidden_sig1(lstm_out[:, -1, :]))
            sig_mu = F.leaky_relu(self.hidden_sig2(sig_mu))
            sig_mu = torch.sigmoid(self.hidden_sig3(sig_mu))
            mu = F.softmax(tan_mu * (1 - sig_mu) + (sig_mu) * action_state, dim=1)
            return mu
        else:
            cat_layer = torch.cat((lstm_out[:, -1, :], action_state, action), 1)
            q = F.leaky_relu(self.hidden_layer1(cat_layer))
            q = F.leaky_relu(self.hidden_layer2(q))
            q = self.hidden_layer3(q)
            return q


class ATTN_A2C(nn.Module):
    def __init__(self, input_size, action_size, nhead=2, transformer_layers=1, lstm_layers=1):
        super(ATTN_A2C, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=lstm_layers)
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer1 = nn.TransformerEncoder(encoder_layer1, num_layers=transformer_layers)
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=lstm_layers)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer2 = nn.TransformerEncoder(encoder_layer2, num_layers=transformer_layers)
        self.lstm3 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=lstm_layers)
        encoder_layer3 = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer3 = nn.TransformerEncoder(encoder_layer3, num_layers=transformer_layers)

        self.hidden_tan1 = nn.Linear(input_size, 64)
        self.hidden_tan2 = nn.Linear(64, 16)
        self.hidden_tan3 = nn.Linear(16, action_size)

        self.hidden_sig1 = nn.Linear(input_size, 64)
        self.hidden_sig2 = nn.Linear(64, 16)
        self.hidden_sig3 = nn.Linear(16, action_size)

        self.hidden_layer1 = nn.Linear(input_size + action_size * 2, 64)
        self.hidden_layer2 = nn.Linear(64, 8)
        self.hidden_layer3 = nn.Linear(8, 1)

    def forward(self, env_state, action_state, action=None, type='actor'):
        lstm_out, (h_n, c_n) = self.lstm1(env_state)
        attn_out = self.transformer1(lstm_out)
        lstm_out, (h_n, c_n) = self.lstm2(attn_out)
        attn_out = self.transformer2(lstm_out)
        lstm_out, (h_n, c_n) = self.lstm3(attn_out)
        attn_out = self.transformer3(lstm_out)
        if type == 'actor':
            tan_mu = F.leaky_relu(self.hidden_tan1(attn_out[:, -1, :]))
            tan_mu = F.leaky_relu(self.hidden_tan2(tan_mu))
            tan_mu = torch.tanh(self.hidden_tan3(tan_mu))

            sig_mu = F.leaky_relu(self.hidden_sig1(attn_out[:, -1, :]))
            sig_mu = F.leaky_relu(self.hidden_sig2(sig_mu))
            sig_mu = torch.sigmoid(self.hidden_sig3(sig_mu))
            mu = F.softmax(tan_mu * (1 - sig_mu) + (sig_mu) * action_state, dim=1)
            return mu
        else:
            cat_layer = torch.cat((attn_out[:, -1, :], action_state, action), 1)
            q = F.leaky_relu(self.hidden_layer1(cat_layer))
            q = F.leaky_relu(self.hidden_layer2(q))
            q = self.hidden_layer3(q)
            return q


class ATTN_QLearning(nn.Module):
    def __init__(self, input_size, action_size, nhead=2, transformer_layers=1, lstm_layers=1):
        super(ATTN_QLearning, self).__init__()
        self.lstm1 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=lstm_layers)
        encoder_layer1 = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer1 = nn.TransformerEncoder(encoder_layer1, num_layers=transformer_layers)
        self.lstm2 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=lstm_layers)
        encoder_layer2 = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer2 = nn.TransformerEncoder(encoder_layer2, num_layers=transformer_layers)
        self.lstm3 = nn.LSTM(input_size=input_size, hidden_size=input_size, num_layers=lstm_layers)
        encoder_layer3 = nn.TransformerEncoderLayer(d_model=input_size, nhead=nhead)
        self.transformer3 = nn.TransformerEncoder(encoder_layer3, num_layers=transformer_layers)

        self.hidden_mu1 = nn.Linear(input_size + action_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, action_size)

        self.hidden_sigma_m1 = nn.Linear(input_size + action_size, 64)
        self.hidden_sigma_m2 = nn.Linear(64, int(action_size * action_size))

        self.hidden_sigma_v1 = nn.Linear(input_size + action_size, 64)
        self.hidden_sigma_v2 = nn.Linear(64, 8)
        self.hidden_sigma_v3 = nn.Linear(8, 1)

        self.output_size = action_size

    def forward(self, env_state, action_state):
        lstm_out, (h_n, c_n) = self.lstm1(env_state)
        attn_out = self.transformer1(lstm_out)
        lstm_out, (h_n, c_n) = self.lstm2(attn_out)
        attn_out = self.transformer2(lstm_out)
        lstm_out, (h_n, c_n) = self.lstm3(attn_out)
        attn_out = self.transformer3(lstm_out)
        cat_layer = torch.cat((attn_out[:, -1, :], action_state), 1)

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
