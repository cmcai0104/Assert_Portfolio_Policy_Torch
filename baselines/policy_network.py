import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class DCNN(nn.Module):
    def __init__(self, h, w, outputs):
        super(DCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        def conv2d_size_out(size, kernel_size=5, stride=2):
            return (size - (kernel_size - 1) - 1) // stride + 1

        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))


class LSTM_Dist(nn.Module):
    def __init__(self, input_size, action_size, hidden_size, output_size):
        super(LSTM_Dist, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_mu1 = nn.Linear(hidden_size+action_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, output_size)

        self.hidden_sigma_m1 = nn.Linear(hidden_size+action_size, 64)
        self.hidden_sigma_m2 = nn.Linear(64, int(output_size * output_size))

        self.hidden_sigma_v1 = nn.Linear(hidden_size+action_size, 64)
        self.hidden_sigma_v2 = nn.Linear(64, 32)
        self.hidden_sigma_v3 = nn.Linear(32, 16)
        self.hidden_sigma_v4 = nn.Linear(16, output_size)

        self.output_size = output_size


    def forward(self, env_state, action_state):
        lstm_out, (h_n, c_n) = self.lstm(env_state)
        cat_layer = torch.cat((lstm_out[:,-1,:], action_state), 1)

        mu = F.relu(self.hidden_mu1(cat_layer))
        mu = F.relu(self.hidden_mu2(mu))
        mu = F.relu(self.hidden_mu3(mu))
        mu = torch.softmax(self.hidden_mu4(mu), dim=1)

        sigma_matrix = F.relu(self.hidden_sigma_m1(cat_layer))
        sigma_matrix = torch.tanh(self.hidden_sigma_m2(sigma_matrix))

        sigma_vector = F.relu(self.hidden_sigma_v1(cat_layer))
        sigma_vector = F.relu(self.hidden_sigma_v2(sigma_vector))
        sigma_vector = F.relu(self.hidden_sigma_v3(sigma_vector))
        sigma_vector = torch.exp(self.hidden_sigma_v4(sigma_vector))

        return mu, sigma_matrix.reshape((-1, self.output_size, self.output_size)), sigma_vector


class LSTM_DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_DQN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_mu1 = nn.Linear(hidden_size+output_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, output_size)

        self.hidden_sigma_m1 = nn.Linear(hidden_size+output_size, 64)
        self.hidden_sigma_m2 = nn.Linear(64, int(output_size * output_size))

        self.hidden_sigma_v1 = nn.Linear(hidden_size+output_size, 64)
        self.hidden_sigma_v2 = nn.Linear(64, 8)
        self.hidden_sigma_v3 = nn.Linear(8, 1)

        self.output_size = output_size

    def forward(self, env_state, action_state):
        lstm_out, (h_n, c_n) = self.lstm(env_state)
        cat_layer = torch.cat((lstm_out[:, -1, :], action_state), 1)

        mu = F.relu(self.hidden_mu1(cat_layer))
        mu = F.relu(self.hidden_mu2(mu))
        mu = F.relu(self.hidden_mu3(mu))
        mu = torch.softmax(self.hidden_mu4(mu), dim=1)

        sigma_matrix = F.relu(self.hidden_sigma_m1(cat_layer))
        sigma_matrix = self.hidden_sigma_m2(sigma_matrix)

        sigma_vector = F.relu(self.hidden_sigma_v1(cat_layer))
        sigma_vector = F.relu(self.hidden_sigma_v2(sigma_vector))
        sigma_vector = self.hidden_sigma_v3(sigma_vector)

        return mu, sigma_matrix.reshape((-1, self.output_size, self.output_size)), sigma_vector



class LSTM_A2C(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM_A2C, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_mu1 = nn.Linear(hidden_size+output_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, output_size)

        self.hidden_layer1 = nn.Linear(hidden_size + output_size*2, 64)
        self.hidden_layer2 = nn.Linear(64, 8)
        self.hidden_layer3 = nn.Linear(8,1)

    def forward(self, env_state, action_state):
        lstm_out, (h_n, c_n) = self.lstm(env_state)
        cat_layer = torch.cat((lstm_out[:, -1, :], action_state), 1)

        mu = F.relu(self.hidden_mu1(cat_layer))
        mu = F.relu(self.hidden_mu2(mu))
        mu = F.relu(self.hidden_mu3(mu))
        mu = torch.softmax(self.hidden_mu4(mu), dim=1)

        cat_layer = torch.cat((cat_layer, mu), 1)
        q = F.selu(self.hidden_layer1(cat_layer))
        q = F.selu(self.hidden_layer2(q))
        q = self.hidden_layer3(q)
        return mu, q