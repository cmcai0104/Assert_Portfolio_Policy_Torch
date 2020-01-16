import torch
import torchvision
from torchvision import transforms
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

trainsets = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainsets, batch_size=64, shuffle=True)
testsets = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testsets, batch_size=64, shuffle=False)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.lstm = nn.LSTM(input_size=32*3, hidden_size=128, batch_first=True, num_layers=3)
        self.output = nn.Linear(128, 10)

    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return self.output(out[:,-1,:])

if __name__ == '__main__':
    net = Net()
    criti = nn.CrossEntropyLoss()
    opt = optim.Adam(net.parameters(), lr=0.001)
    for x, y in trainloader:
        x = Variable(x).view(-1, 32, 32*3)
        y = Variable(y)
        out = net(x)
        loss = criti(out, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(loss)


