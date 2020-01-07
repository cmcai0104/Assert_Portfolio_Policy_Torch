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
        self.hiddenfc1 = nn.Linear(hidden_size, 64)
        self.hiddenfc2 = nn.Linear(64, 32)
        self.hiddenfc3 = nn.Linear(32, 16)
        self.hiddenfc4 = nn.Linear(16, output_size)
        self.hidden_size = hidden_size
        self.hidden = self.init_hidden()

    def init_hidden(self):
        return (Variable(torch.randn(2, 1, self.hidden_size)),
                Variable(torch.randn(2, 1, self.hidden_size)))

    def forward(self, wordvecs):
        lstm_out, self.hidden = self.lstm(wordvecs, self.hidden)
        tag_space = F.relu(self.hiddenfc1(lstm_out[:, -1, :]))
        tag_space = F.relu(self.hiddenfc2(tag_space))
        tag_space = F.relu(self.hiddenfc3(tag_space))
        # tag_space = F.sigmoid(self.hiddenfc4(tag_space))
        tag_space = torch.sigmoid(self.hiddenfc4(tag_space))
        self.hidden = self.init_hidden()
        return tag_space