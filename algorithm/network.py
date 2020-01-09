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

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))




class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2,
                            bias=True, batch_first=True, dropout=0, bidirectional=False)
        self.hidden_mu1 = nn.Linear(hidden_size, 64)
        self.hidden_mu2 = nn.Linear(64, 32)
        self.hidden_mu3 = nn.Linear(32, 16)
        self.hidden_mu4 = nn.Linear(16, output_size)

        self.hidden_sigma_m1 = nn.Linear(hidden_size, 64)
        self.hidden_sigma_m2 = nn.Linear(64, int(output_size * output_size))

        self.hidden_sigma_v1 = nn.Linear(hidden_size, 64)
        self.hidden_sigma_v2 = nn.Linear(64, 32)
        self.hidden_sigma_v3 = nn.Linear(32, 16)
        self.hidden_sigma_v4 = nn.Linear(16, output_size)

        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.randn(2, 1, self.hidden_size)),
                Variable(torch.randn(2, 1, self.hidden_size)))

    def forward(self, input):
        lstm_out, self.hidden = self.lstm(input, self.hidden)
        mu = F.relu(self.hidden_mu1(lstm_out[:, -1, :]))
        mu = F.relu(self.hidden_mu2(mu))
        mu = F.relu(self.hidden_mu3(mu))
        mu = torch.sigmoid(self.hidden_mu4(mu))

        sigma_matrix = F.relu(self.hidden_sigma_m1(lstm_out[:, -1, :]))
        sigma_matrix = torch.tanh(self.hidden_sigma_m2(sigma_matrix))

        sigma_vector = F.relu(self.hidden_sigma_v1(lstm_out[:, -1, :]))
        sigma_vector = F.relu(self.hidden_sigma_v2(sigma_vector))
        sigma_vector = F.relu(self.hidden_sigma_v3(sigma_vector))
        sigma_vector = torch.exp(self.hidden_sigma_v4(sigma_vector))

        return mu, sigma_matrix.reshape((self.output_size, self.output_size)), sigma_vector