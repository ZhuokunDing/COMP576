import torch 
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from cifar10 import accuracy
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class RNN(nn.Module):
    def __init__(self, cell='RNN', hidden=64):
        super(RNN, self).__init__()
        if cell == 'RNN':
            self.rnn = nn.RNN(input_size=28, hidden_size=hidden, num_layers=1, batch_first=True)
        elif cell == 'LSTM':
            self.rnn = nn.LSTM(input_size=28, hidden_size=hidden, num_layers=1, batch_first=True)
        elif cell == 'GRU':
            self.rnn = nn.GRU(input_size=28, hidden_size=hidden, num_layers=1, batch_first=True)
        self.fc = nn.Linear(hidden, 10)

    def forward(self, x):
        r_out, _ = self.rnn(x, None)   # None represents zero initial hidden state
        out = self.fc(r_out[:, -1, :])
        return out

def load_data(batch_size):
    # MNIST dataset
    train_dataset = torchvision.datasets.MNIST(root='data/',
                                            train=True, 
                                            transform=transforms.ToTensor(),
                                            download=True)

    test_dataset = torchvision.datasets.MNIST(root='data/',
                                            train=False, 
                                            transform=transforms.ToTensor())

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                            batch_size=batch_size, 
                                            shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                            batch_size=len(test_dataset),
                                            shuffle=False)
    return train_loader, test_loader



def train_rnn(train_loader, test_loader, lr, momentum, report=20, cell='RNN', hidden=64, logdir='results/rnn'):
    writer = SummaryWriter(logdir)
    ntrain = 1000 # per class
    ntest = 100 # per class
    nclass = 10 # number of classes
    imsize = 28
    batchsize = 100
    nsamples = ntrain * nclass

    net = RNN(cell,hidden)
    net.cuda()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    running_loss = 0.0
    # batch_xs = torch.tensor(np.zeros((batchsize, imsize, imsize)), device='cuda', dtype=torch.float32)#setup as [batchsize, width, height, numberOfChannels] and use np.zeros()
    # batch_ys = torch.tensor(np.zeros(batchsize), device='cuda', dtype=torch.long)#setup as [batchsize, the how many classes] 
    test_xs ,test_ys = next(iter(test_loader))
    test_xs = test_xs.cuda()
    test_ys = test_ys.cuda()

    for epoch in range(1):
        for i, (batch_xs, batch_ys) in enumerate(train_loader):
            batch_xs = batch_xs.view(-1, 28, 28).cuda()
            batch_ys = batch_ys.cuda()
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(batch_xs)
            loss = criterion(outputs, batch_ys)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            if (i+1) % report == 0:    # print every {report * batch_size} iterations
                print('[%d, %5d] loss: %.3f' %
                        (epoch + 1, i + 1, running_loss / report))
                writer.add_scalar('training_loss', running_loss / report, epoch * nsamples + (i+1)*batchsize)
                running_loss = 0.0

                train_accuracy = accuracy(net, batch_xs.view(-1, 28, 28).float(), batch_ys)
                print(f'train_accuracy:{train_accuracy:.3f}')
                writer.add_scalar('training_accuracy', train_accuracy, epoch * nsamples + (i+1)*batchsize)

                test_accuracy = accuracy(net, test_xs.view(-1, 28, 28).float(), test_ys)
                print(f'test_accuracy:{test_accuracy:.3f}')
                writer.add_scalar('test_accuracy', test_accuracy, epoch * nsamples + (i+1)*batchsize)


    return net


